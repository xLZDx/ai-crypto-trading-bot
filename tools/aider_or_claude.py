"""
Aider-with-fallback wrapper.

Per the 2026-05-13 rule update: when Aider's chosen model hits a rate limit
(RPM / TPM / RPD), fall back through a chain of Gemini variants, then to
Claude (if ANTHROPIC_API_KEY is set), and finally signal to the controlling
Claude agent that no remote model could complete the task — at which point
the *direct Claude work* path (per the agents-first rule's carve-out) takes
over.

Usage:
    venv/Scripts/python.exe tools/aider_or_claude.py \\
        --read src/foo.py --read src/bar.py \\
        --message "do X" \\
        target_file.py

Equivalent to:
    aider --model <chain[0]> ... target_file.py
with automatic retry on the rest of the chain on quota errors.

Model fallback order (top → bottom):
    1. gemini/gemini-2.5-pro      (highest quality, lowest quota)
    2. gemini/gemini-2.5-flash    (mid quality, mid quota)
    3. gemini/gemini-2.0-flash    (cheaper, larger quota)
    4. gemini/gemini-2.0-flash-lite (cheapest Gemini)
    5. anthropic/claude-sonnet-4-6  (only if ANTHROPIC_API_KEY set)
    6. EXIT 42 — signal to caller (direct-Claude path takes over)

Quota signals detected in Aider stdout/stderr:
    - "rate limit"
    - "RESOURCE_EXHAUSTED"
    - "quota exceeded"
    - "429"
    - "rate_limit_error"
    - "insufficient_quota"
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Default fallback chain. Override with $AIDER_MODEL_CHAIN (comma-separated).
DEFAULT_CHAIN = [
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "anthropic/claude-sonnet-4-6",
]

# Regex that fires on any rate-limit / quota signal in stdout+stderr.
QUOTA_SIGNAL = re.compile(
    r"(rate[\s_-]?limit"
    r"|RESOURCE_EXHAUSTED"
    r"|quota[\s_-]?exceeded"
    r"|insufficient[\s_-]?quota"
    r"|rate_limit_error"
    r"|GenerateRequestsPerMinute"
    r"|GenerateRequestsPerDay"
    r"|GenerateContentInputTokens"
    r"|\bHTTP[/ ]?1\.[01]\b.{0,30}\b429\b"
    r"|status code:\s*429"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Project root + Aider binary path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIDER_BIN = Path("D:/tools/aider-env/Scripts/aider.exe")

# Exit code emitted when ALL models in the chain fail. The controlling
# Claude agent should catch this and switch to direct-Claude work.
EXIT_ALL_QUOTA_EXHAUSTED = 42


def _build_chain() -> list[str]:
    """Return the model chain, possibly filtered by available API keys.

    - Drops anthropic/* if ANTHROPIC_API_KEY is not set.
    - Drops openai/* if OPENAI_API_KEY is not set.
    - Drops gemini/* if GEMINI_API_KEY is not set (rare).
    """
    raw = os.getenv("AIDER_MODEL_CHAIN") or ",".join(DEFAULT_CHAIN)
    chain = [m.strip() for m in raw.split(",") if m.strip()]
    out = []
    for m in chain:
        provider = m.split("/", 1)[0]
        if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            print(f"[fallback] skipping {m} — ANTHROPIC_API_KEY not set", file=sys.stderr)
            continue
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            print(f"[fallback] skipping {m} — OPENAI_API_KEY not set", file=sys.stderr)
            continue
        if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            print(f"[fallback] skipping {m} — GEMINI_API_KEY not set", file=sys.stderr)
            continue
        out.append(m)
    return out


def _run_aider(model: str, passthrough_args: list[str]) -> tuple[int, str, str]:
    """Run Aider once with the chosen model. Returns (exit_code, stdout, stderr).

    Captures both streams so we can scan for quota signals without losing them.
    Streams are echoed live to the parent stdout/stderr so the operator sees
    progress.
    """
    if not AIDER_BIN.exists():
        return 127, "", f"Aider binary not found at {AIDER_BIN}"

    cmd = [
        str(AIDER_BIN),
        "--model", model,
        "--no-auto-commits",
        "--no-gitignore",
        "--no-show-model-warnings",
        "--yes-always",
    ] + passthrough_args

    print(f"\n[fallback] trying model: {model}", file=sys.stderr)
    print(f"[fallback] cmd: {' '.join(cmd[:6])} ... ({len(cmd)} total args)", file=sys.stderr)

    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    # Drain both streams (no live-tee — keeps logic simple).
    try:
        out, err = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, out or "", (err or "") + "\n[fallback] aider TIMEOUT (>10 min)"
    if out:
        stdout_chunks.append(out)
        sys.stdout.write(out)
    if err:
        stderr_chunks.append(err)
        sys.stderr.write(err)
    return proc.returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    chain = _build_chain()
    if not chain:
        print("[fallback] no API keys for any model in the chain. "
              "Add GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY to .env.",
              file=sys.stderr)
        return EXIT_ALL_QUOTA_EXHAUSTED

    for i, model in enumerate(chain):
        rc, out, err = _run_aider(model, argv)
        combined = out + err
        if rc == 0:
            print(f"\n[fallback] SUCCESS with model {model}", file=sys.stderr)
            return 0
        # Quota or rate-limit signal? Try the next model.
        if QUOTA_SIGNAL.search(combined):
            print(f"[fallback] {model} hit quota/rate limit (rc={rc}). "
                  f"Trying next model in chain…", file=sys.stderr)
            continue
        # Non-quota failure: don't keep cycling — bail with the error.
        print(f"[fallback] {model} failed with non-quota error (rc={rc}). "
              f"Aborting fallback chain to avoid masking the real issue.",
              file=sys.stderr)
        return rc

    # All models in the chain hit quota.
    print(
        "\n[fallback] EVERY model in the chain hit quota / rate limit. "
        "Signaling to the controlling Claude agent (exit 42) so direct-Claude "
        "work can take over per the agents-first rule's carve-out.",
        file=sys.stderr,
    )
    return EXIT_ALL_QUOTA_EXHAUSTED


if __name__ == "__main__":
    raise SystemExit(main())
