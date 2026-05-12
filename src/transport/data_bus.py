"""
DataBus — ZeroMQ-based data plane for high-throughput streaming.

Two patterns:

  1. Orderflow (PUB/SUB)
     - Master PUBs L2/L3 snapshots; multiple consumers (bot, dashboard,
       training) SUBscribe independently.
     - Topic-prefixed: each message is sent as `topic | payload`.

  2. Training batches (PUSH/PULL)
     - Master PUSHes mini-batches. Workers PULL with automatic load balancing.
     - Used by the joint OFT+RL training loop in Phase 3.

This module preserves the same public API for a future Kafka swap (M2 in
INSTITUTIONAL_UPGRADE_PLAN.md). When that migration fires, the public methods
(`publish_orderflow`, `subscribe_orderflow`, `push_batch`, `pull_batch`)
keep their signatures; only the internals change.

Wire format (Phase A6, 2026-05-12):
  byte 0      : version (0x01 = HMAC + msgpack)
  bytes 1..33 : HMAC-SHA256 over msgpack_bytes using ZMQ_BUS_KEY
  bytes 33..  : msgpack-packed payload

Previously the wire format was msgpack with a `\\x00pickle:` fallback for
numpy/torch payloads. The pickle path was a critical RCE vector — any
process that could connect to the local ZMQ ports could forge a payload
and trigger arbitrary code execution on every subscriber. Phase A6 removes
pickle entirely and HMAC-signs every envelope. Callers that need to send
tensors must convert to plain Python (`.tolist()`, dicts) before publish.
"""
from __future__ import annotations

import hmac
import hashlib
import logging
import os
import secrets
import threading
import time
from typing import Callable, Iterable

from .zmq_config import (
    ORDERFLOW_PORT,
    TRAINING_BATCH_PORT,
    CONTROL_FANOUT_PORT,
    bind_addr,
    connect_addr,
)

logger = logging.getLogger("data_bus")

ORDERFLOW_TOPIC_DEFAULT = b"orderflow"

# ── Phase A6: HMAC envelope ─────────────────────────────────────────────────
_ENVELOPE_VERSION = b"\x01"
_HMAC_LEN = 32  # SHA-256


def _load_bus_key() -> bytes:
    """Load the HMAC key for envelope signing.

    Operator sets ZMQ_BUS_KEY in .env (any non-empty string; 32+ bytes
    recommended). All processes that share the data bus MUST share the
    same key — otherwise the subscriber will reject every publisher's
    envelope as forged.

    If the env var is missing, generate an ephemeral 32-byte key at
    module load time AND log a CRITICAL warning. Single-process tests
    will work; cross-process pub/sub will silently drop every message
    because the subscriber's key won't match the publisher's. This
    keeps the security default fail-CLOSED while leaving a clear
    operator path forward.
    """
    raw = os.getenv("ZMQ_BUS_KEY", "")
    if raw:
        # Hash any provided key to a fixed 32-byte derived key. This
        # accepts both short human-typed strings and long base64 keys.
        return hashlib.sha256(raw.encode("utf-8")).digest()
    key = secrets.token_bytes(32)
    logger.critical(
        "ZMQ_BUS_KEY not set in .env — generated an ephemeral key for "
        "this process only. Cross-process pub/sub will FAIL silently "
        "(subscribers reject envelopes signed with a different key). "
        "Add ZMQ_BUS_KEY=<random 32+ char string> to .env on every "
        "machine in the cluster to enable cross-process messaging."
    )
    return key


_BUS_KEY = _load_bus_key()


def _serialize(payload) -> bytes:
    """Pack with msgpack, sign with HMAC-SHA256, wrap in envelope.

    Raises ValueError if msgpack can't serialize the payload (e.g.
    numpy arrays or torch tensors). Callers must convert to plain
    Python (lists / dicts) first — pickle fallback was removed in
    Phase A6 because it permitted RCE on any subscriber.
    """
    try:
        import msgpack
        body = msgpack.packb(payload, use_bin_type=True)
    except (ImportError, TypeError, ValueError) as exc:
        raise ValueError(
            f"data_bus._serialize: msgpack cannot pack {type(payload).__name__} "
            f"({exc}). Convert tensors/arrays to plain Python first "
            f"(e.g. arr.tolist() / tensor.cpu().numpy().tolist())."
        ) from exc
    sig = hmac.new(_BUS_KEY, body, hashlib.sha256).digest()
    return _ENVELOPE_VERSION + sig + body


def _deserialize(blob: bytes):
    """Unwrap envelope, verify HMAC, unpack msgpack.

    Raises ValueError on:
      - version mismatch (legacy / forged envelope)
      - too-short envelope
      - HMAC mismatch (forged or wrong key)
      - msgpack decode failure
    """
    if len(blob) < 1 + _HMAC_LEN:
        raise ValueError(
            f"data_bus._deserialize: envelope too short ({len(blob)} bytes)"
        )
    version = blob[:1]
    if version != _ENVELOPE_VERSION:
        raise ValueError(
            f"data_bus._deserialize: unknown envelope version {version!r}; "
            f"expected {_ENVELOPE_VERSION!r}. This is either a legacy "
            f"(pre-A6, unauthenticated) message or a forged envelope."
        )
    sig_received = blob[1:1 + _HMAC_LEN]
    body = blob[1 + _HMAC_LEN:]
    sig_expected = hmac.new(_BUS_KEY, body, hashlib.sha256).digest()
    if not hmac.compare_digest(sig_received, sig_expected):
        raise ValueError(
            "data_bus._deserialize: HMAC verification failed. The "
            "envelope was either forged or signed with a different "
            "ZMQ_BUS_KEY than this process holds."
        )
    try:
        import msgpack
        return msgpack.unpackb(body, raw=False)
    except Exception as exc:
        raise ValueError(
            f"data_bus._deserialize: msgpack decode failed: {exc}"
        ) from exc


class DataBus:
    """ZeroMQ data plane. Holds sockets for the configured role."""

    def __init__(self, master_host: str | None = None):
        self.master_host = master_host  # None = bind locally; set = connect
        self._ctx = None
        self._sockets: dict[str, object] = {}
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def _ctx_or_open(self):
        if self._ctx is None:
            import zmq
            self._ctx = zmq.Context.instance()
        return self._ctx

    def _socket(self, key: str, sock_type: int, action: str, port: int,
                topic_filter: bytes | None = None):
        """Open or return cached socket. action ∈ {'bind', 'connect'}."""
        with self._lock:
            if key in self._sockets:
                return self._sockets[key]
            import zmq
            ctx = self._ctx_or_open()
            sock = ctx.socket(sock_type)
            if action == "bind":
                sock.bind(bind_addr(port))
                logger.info("[DataBus] bind %s on port %d", key, port)
            else:
                sock.connect(connect_addr(port, self.master_host))
                logger.info("[DataBus] connect %s → %s", key, connect_addr(port, self.master_host))
            if sock_type == zmq.SUB and topic_filter is not None:
                sock.setsockopt(zmq.SUBSCRIBE, topic_filter)
            self._sockets[key] = sock
            return sock

    def close(self) -> None:
        self._stop_flag.set()
        with self._lock:
            for sock in self._sockets.values():
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
            self._sockets.clear()

    # ── Orderflow (PUB/SUB) ─────────────────────────────────────────────────

    def publish_orderflow(self, snapshot: dict, topic: bytes = ORDERFLOW_TOPIC_DEFAULT) -> None:
        """Master-side publish of an L2/L3 snapshot."""
        import zmq
        sock = self._socket("orderflow_pub", zmq.PUB, "bind", ORDERFLOW_PORT)
        sock.send_multipart([topic, _serialize(snapshot)])

    def subscribe_orderflow(
        self,
        callback: Callable[[dict], None],
        topic: bytes = ORDERFLOW_TOPIC_DEFAULT,
        daemon: bool = True,
    ) -> threading.Thread:
        """Worker/consumer-side subscription. Returns the spawned listener thread."""
        import zmq
        sock = self._socket(
            "orderflow_sub", zmq.SUB, "connect", ORDERFLOW_PORT, topic_filter=topic
        )

        def _loop():
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while not self._stop_flag.is_set():
                events = dict(poller.poll(timeout=500))
                if sock in events:
                    try:
                        _topic, blob = sock.recv_multipart()
                        callback(_deserialize(blob))
                    except Exception as exc:
                        logger.warning("[DataBus] orderflow callback error: %s", exc)

        t = threading.Thread(target=_loop, daemon=daemon, name="orderflow-sub")
        t.start()
        return t

    # ── Training batches (PUSH/PULL) ────────────────────────────────────────

    def push_batch(self, batch) -> None:
        """Master-side push of a training batch. Auto-load-balances to PULLers."""
        import zmq
        sock = self._socket("batch_push", zmq.PUSH, "bind", TRAINING_BATCH_PORT)
        sock.send(_serialize(batch))

    def pull_batch(self, timeout_ms: int = 5000):
        """Worker-side pull of one batch. Returns None on timeout."""
        import zmq
        sock = self._socket("batch_pull", zmq.PULL, "connect", TRAINING_BATCH_PORT)
        if sock.poll(timeout_ms) == 0:
            return None
        return _deserialize(sock.recv())

    # ── Control fanout (PUB/SUB, low volume) ───────────────────────────────

    def publish_control(self, message: dict, topic: bytes = b"control") -> None:
        import zmq
        sock = self._socket("control_pub", zmq.PUB, "bind", CONTROL_FANOUT_PORT)
        sock.send_multipart([topic, _serialize(message)])

    def subscribe_control(
        self,
        callback: Callable[[dict], None],
        topic: bytes = b"control",
        daemon: bool = True,
    ) -> threading.Thread:
        import zmq
        sock = self._socket(
            "control_sub", zmq.SUB, "connect", CONTROL_FANOUT_PORT, topic_filter=topic
        )

        def _loop():
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while not self._stop_flag.is_set():
                events = dict(poller.poll(timeout=500))
                if sock in events:
                    try:
                        _topic, blob = sock.recv_multipart()
                        callback(_deserialize(blob))
                    except Exception as exc:
                        logger.warning("[DataBus] control callback error: %s", exc)

        t = threading.Thread(target=_loop, daemon=daemon, name="control-sub")
        t.start()
        return t

    # ── Introspection ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "open_sockets": list(self._sockets.keys()),
                "master_host":  self.master_host or "(local bind)",
                "orderflow_port":      ORDERFLOW_PORT,
                "training_batch_port": TRAINING_BATCH_PORT,
                "control_fanout_port": CONTROL_FANOUT_PORT,
            }


# ─── Singleton helper ─────────────────────────────────────────────────────────

_bus_instance: DataBus | None = None
_bus_lock = threading.Lock()


def get_data_bus(master_host: str | None = None) -> DataBus:
    """Return the process-wide DataBus instance.

    On the master node, leave master_host=None (sockets bind locally).
    On worker nodes, pass master_host="192.168.0.X" to connect.
    """
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = DataBus(master_host=master_host)
    return _bus_instance
