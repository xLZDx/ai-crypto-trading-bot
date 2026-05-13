"""Behavioral tests for src/utils/model_integrity.py (Phase A8).

These tests exercise the real sign / verify path against on-disk
files in a tmpdir. No string-matching on source — every assertion
runs the actual code and inspects observable state.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import model_integrity as mi  # noqa: E402


class _BaseIntegrityTest(unittest.TestCase):
    """Common scaffolding: redirect manifest into a per-test tmpdir."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_root = self.tmp.name
        # Point the project root and manifest path inside the tmpdir.
        self._root_patch = mock.patch.object(mi, "_project_root",
                                             return_value=self.tmp_root)
        self._root_patch.start()
        os.makedirs(os.path.join(self.tmp_root, "models"), exist_ok=True)
        # Reset module state (cached key + warning flags).
        mi._reset_for_tests()
        # Clean key env by default; tests opt-in.
        self._env_patch = mock.patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        os.environ.pop("MODEL_MANIFEST_KEY", None)

    def tearDown(self) -> None:
        self._env_patch.stop()
        self._root_patch.stop()
        mi._reset_for_tests()
        self.tmp.cleanup()

    def _make_model(self, name: str, payload: bytes = b"model-bytes-v1") -> str:
        path = os.path.join(self.tmp_root, "models", name)
        with open(path, "wb") as f:
            f.write(payload)
        return path

    def _set_key(self, value: str = "test-secret-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") -> None:
        os.environ["MODEL_MANIFEST_KEY"] = value
        mi._reset_for_tests()


class TestSignVerifyRoundtrip(_BaseIntegrityTest):
    def test_sign_then_verify_passes(self) -> None:
        self._set_key()
        p = self._make_model("a.joblib")
        self.assertTrue(mi.sign_model(p))
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as f:
            manifest = json.load(f)
        self.assertIn("models/a.joblib", manifest["entries"])
        entry = manifest["entries"]["models/a.joblib"]
        self.assertEqual(len(entry["hmac_sha256"]), 64)
        self.assertEqual(entry["size_bytes"], len(b"model-bytes-v1"))
        mi.verify_model_or_raise(p)

    def test_tampered_file_raises(self) -> None:
        self._set_key()
        p = self._make_model("b.joblib", b"original")
        mi.sign_model(p)
        with open(p, "wb") as f:
            f.write(b"tampered!")
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_model_or_raise(p)

    def test_sign_idempotent_updates_hash(self) -> None:
        self._set_key()
        p = self._make_model("c.joblib", b"v1")
        mi.sign_model(p)
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as f:
            h1 = json.load(f)["entries"]["models/c.joblib"]["hmac_sha256"]
        with open(p, "wb") as f:
            f.write(b"v2-different-bytes")
        mi.sign_model(p)
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as f:
            h2 = json.load(f)["entries"]["models/c.joblib"]["hmac_sha256"]
        self.assertNotEqual(h1, h2)
        mi.verify_model_or_raise(p)


class TestVerifyAndLoadBytes(_BaseIntegrityTest):
    """The TOCTOU-free read path: open once, verify, return bytes."""

    def test_returns_bytes_after_sign(self) -> None:
        self._set_key()
        payload = b"\x80\x04pickle-style-bytes-v1"
        p = self._make_model("rt.joblib", payload)
        mi.sign_model(p)
        out = mi.verify_and_load_bytes(p)
        self.assertEqual(out, payload)

    def test_tampered_raises(self) -> None:
        self._set_key()
        p = self._make_model("rt2.joblib", b"good")
        mi.sign_model(p)
        with open(p, "wb") as f:
            f.write(b"evil")
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_and_load_bytes(p)

    def test_missing_file_raises_filenotfound(self) -> None:
        self._set_key()
        with self.assertRaises(FileNotFoundError):
            mi.verify_and_load_bytes(os.path.join(self.tmp_root, "models", "nope.joblib"))

    def test_no_key_returns_bytes_fail_open(self) -> None:
        # No key set: integrity is bypassed but bytes are still returned
        # (so the load path stays functional in dev / bootstrap).
        p = self._make_model("nokey.joblib", b"raw-bytes")
        out = mi.verify_and_load_bytes(p)
        self.assertEqual(out, b"raw-bytes")


class TestPolicyFailOpenAndFailClosed(_BaseIntegrityTest):
    def test_missing_key_bypasses_verify(self) -> None:
        p = self._make_model("d.joblib", b"tampered-but-no-key-set")
        mi.verify_model_or_raise(p)  # must NOT raise

    def test_missing_key_skips_sign(self) -> None:
        p = self._make_model("e.joblib")
        self.assertFalse(mi.sign_model(p))
        manifest_path = os.path.join(self.tmp_root, "models", "manifest.json")
        self.assertFalse(os.path.exists(manifest_path))

    def test_missing_entry_warns_but_allows(self) -> None:
        self._set_key()
        a = self._make_model("a.joblib")
        mi.sign_model(a)
        b = self._make_model("b.joblib", b"untracked-model")
        mi.verify_model_or_raise(b)

    def test_verify_with_missing_file_does_not_raise(self) -> None:
        self._set_key()
        missing = os.path.join(self.tmp_root, "models", "does-not-exist.joblib")
        mi.verify_model_or_raise(missing)  # must NOT raise

    def test_malformed_manifest_entry_FAILS_CLOSED(self) -> None:
        """Phase A8 reviewer feedback: malformed entries must raise, not allow."""
        self._set_key()
        p = self._make_model("f.joblib")
        manifest_path = os.path.join(self.tmp_root, "models", "manifest.json")
        with open(manifest_path, "w") as fp:
            json.dump({"version": 1, "entries": {
                "models/f.joblib": {"hmac_sha256": "abc"}  # short hash
            }}, fp)
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_model_or_raise(p)
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_and_load_bytes(p)

    def test_deleted_manifest_entry_falls_back_to_untracked_allow(self) -> None:
        """After sign, an attacker who can edit the manifest could delete the
        entry — they cannot forge a valid one without the key. That branch
        must reach the 'untracked, allow' path (and warn), not crash."""
        self._set_key()
        p = self._make_model("del.joblib")
        mi.sign_model(p)
        manifest_path = os.path.join(self.tmp_root, "models", "manifest.json")
        with open(manifest_path, "w") as fp:
            json.dump({"version": 1, "entries": {}}, fp)
        mi.verify_model_or_raise(p)  # must NOT raise


class TestSymlinkRejection(_BaseIntegrityTest):
    """Symlinks are refused so a swapped target cannot reach the
    untracked-allow branch via a different manifest key."""

    def _make_symlink(self, link: str, target: str) -> bool:
        """Create a symlink; return False if unsupported (e.g., Win user without privs)."""
        try:
            os.symlink(target, link)
            return True
        except (OSError, NotImplementedError):
            return False

    def test_verify_refuses_symlink(self) -> None:
        self._set_key()
        target = self._make_model("real.joblib", b"signed-bytes")
        mi.sign_model(target)
        link = os.path.join(self.tmp_root, "models", "link.joblib")
        if not self._make_symlink(link, target):
            self.skipTest("symlink creation not permitted on this host")
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_model_or_raise(link)
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_and_load_bytes(link)

    def test_sign_refuses_symlink(self) -> None:
        self._set_key()
        target = self._make_model("real2.joblib", b"x")
        link = os.path.join(self.tmp_root, "models", "link2.joblib")
        if not self._make_symlink(link, target):
            self.skipTest("symlink creation not permitted on this host")
        self.assertFalse(mi.sign_model(link))

    def test_verify_refuses_symlink_mocked(self) -> None:
        """Cross-platform: even when the host can't create real symlinks,
        the policy code path must reject any path where os.path.islink is True."""
        self._set_key()
        p = self._make_model("mock_link.joblib", b"x")
        mi.sign_model(p)
        with mock.patch("os.path.islink", return_value=True):
            with self.assertRaises(mi.ModelIntegrityError):
                mi.verify_model_or_raise(p)
            with self.assertRaises(mi.ModelIntegrityError):
                mi.verify_and_load_bytes(p)

    def test_sign_refuses_symlink_mocked(self) -> None:
        self._set_key()
        p = self._make_model("mock_link2.joblib", b"x")
        with mock.patch("os.path.islink", return_value=True):
            self.assertFalse(mi.sign_model(p))


class TestKeyDerivation(_BaseIntegrityTest):
    def test_different_keys_produce_different_hmacs(self) -> None:
        p = self._make_model("g.joblib", b"same-payload")
        self._set_key("alpha-key-XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        mi.sign_model(p)
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as fp:
            h_alpha = json.load(fp)["entries"]["models/g.joblib"]["hmac_sha256"]

        self._set_key("beta-key-XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        mi.sign_model(p)
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as fp:
            h_beta = json.load(fp)["entries"]["models/g.joblib"]["hmac_sha256"]
        self.assertNotEqual(h_alpha, h_beta)

    def test_wrong_key_makes_existing_signed_model_fail(self) -> None:
        p = self._make_model("h.joblib", b"payload")
        self._set_key("right-key-XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        mi.sign_model(p)
        self._set_key("attacker-key-XXXXXXXXXXXXXXXXXXXXXXXXX")
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_model_or_raise(p)

    def test_short_key_logs_warning(self) -> None:
        """Operator setting a weak key gets a one-shot warning."""
        with self.assertLogs("src.utils.model_integrity", level="WARNING") as cap:
            self._set_key("short")
            # Trigger _load_key
            mi.verify_and_load_bytes(self._make_model("sk.joblib", b"x"))
        self.assertTrue(
            any("only 5 chars" in m or "only" in m for m in cap.output),
            f"expected short-key warning, got: {cap.output}",
        )

    def test_relative_path_normalization(self) -> None:
        """Manifest keys use forward slashes regardless of OS path separator."""
        self._set_key()
        p = self._make_model("i.joblib")
        mi.sign_model(p)
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as fp:
            manifest = json.load(fp)
        self.assertIn("models/i.joblib", manifest["entries"])

    def test_whitespace_only_key_is_treated_as_unset(self) -> None:
        """Reviewer finding: a key set to whitespace must not silently
        re-read env on every call. The sentinel should cache 'absent'."""
        os.environ["MODEL_MANIFEST_KEY"] = "   "
        mi._reset_for_tests()
        # First call surfaces the unset warning.
        with self.assertLogs("src.utils.model_integrity", level="WARNING") as cap:
            self.assertIsNone(mi._load_key())
        self.assertTrue(any("unset" in m for m in cap.output))
        # Second call should NOT re-emit the warning (caching works).
        os.environ["MODEL_MANIFEST_KEY"] = "   "  # still whitespace
        # We cannot assert "no warning" cleanly in assertLogs (it fails on
        # absence), but we can assert _load_key still returns None and
        # _key_warning_emitted is set.
        self.assertIsNone(mi._load_key())
        self.assertTrue(mi._key_warning_emitted)


class TestManifestRobustness(_BaseIntegrityTest):
    def test_corrupt_manifest_falls_back_to_empty(self) -> None:
        self._set_key()
        manifest_path = os.path.join(self.tmp_root, "models", "manifest.json")
        with open(manifest_path, "w") as fp:
            fp.write("{not valid json")
        m = mi._load_manifest()
        self.assertEqual(m["entries"], {})
        p = self._make_model("j.joblib")
        self.assertTrue(mi.sign_model(p))
        mi.verify_model_or_raise(p)

    def test_manifest_persists_across_calls(self) -> None:
        self._set_key()
        a = self._make_model("k.joblib")
        b = self._make_model("l.joblib")
        mi.sign_model(a)
        mi.sign_model(b)
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as fp:
            entries = json.load(fp)["entries"]
        self.assertIn("models/k.joblib", entries)
        self.assertIn("models/l.joblib", entries)

    def test_concurrent_sign_preserves_all_entries(self) -> None:
        """Reviewer finding: confirm two threads signing different models
        don't lose each other's entries. The state_lock + safe_json filelock
        together must serialize the read-modify-write."""
        self._set_key()
        paths = [self._make_model(f"m{i}.joblib", b"x" * (i + 1)) for i in range(10)]

        errors: list[BaseException] = []
        def _worker(p: str) -> None:
            try:
                mi.sign_model(p)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(p,)) for p in paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errors, [], f"sign_model raised under concurrency: {errors}")

        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as fp:
            entries = json.load(fp)["entries"]
        for i in range(10):
            self.assertIn(f"models/m{i}.joblib", entries,
                          f"concurrent sign dropped entry m{i}: {sorted(entries)}")


class TestEnforcementMode(_BaseIntegrityTest):
    """Phase 1b (2026-05-14): MODEL_HMAC_ENFORCEMENT knob.

    Default is "enforce" (existing fail-closed behavior — covered above).
    "warn" lets a mismatch through with a CRITICAL log (diagnostics only).
    "off" skips the HMAC computation entirely on every verify call.
    Malformed manifest entries ALWAYS raise regardless of mode.
    """

    def _set_mode(self, mode: str) -> None:
        os.environ[mi._ENFORCEMENT_ENV] = mode
        mi._reset_for_tests()

    def test_default_mode_is_enforce(self) -> None:
        """No env var set -> mode should be 'enforce'."""
        os.environ.pop(mi._ENFORCEMENT_ENV, None)
        mi._reset_for_tests()
        self.assertEqual(mi._load_enforcement_mode(), mi._MODE_ENFORCE)

    def test_invalid_mode_falls_back_to_enforce_with_warning(self) -> None:
        with self.assertLogs("src.utils.model_integrity", level="WARNING") as cap:
            self._set_mode("bogus-value")
            mode = mi._load_enforcement_mode()
        self.assertEqual(mode, mi._MODE_ENFORCE)
        self.assertTrue(any("bogus-value" in m for m in cap.output),
                        f"expected invalid-mode warning: {cap.output}")

    def test_warn_mode_emits_one_shot_critical_alert(self) -> None:
        """warn-mode should warn the operator they're in diagnostic mode."""
        with self.assertLogs("src.utils.model_integrity", level="CRITICAL") as cap:
            self._set_mode("warn")
            mi._load_enforcement_mode()
        self.assertTrue(
            any("warn" in m and "diagnostics only" in m for m in cap.output),
            f"expected warn-mode alert, got: {cap.output}",
        )

    def test_warn_mode_allows_load_on_tampered_file(self) -> None:
        """Tampered file in warn mode -> bytes returned, CRITICAL logged, no raise."""
        self._set_key()
        p = self._make_model("w1.joblib", b"original")
        mi.sign_model(p)
        with open(p, "wb") as f:
            f.write(b"tampered!!!")
        self._set_mode("warn")
        self._set_key()  # re-set key after _reset_for_tests
        with self.assertLogs("src.utils.model_integrity", level="CRITICAL") as cap:
            out = mi.verify_and_load_bytes(p)
        self.assertEqual(out, b"tampered!!!", "warn mode must still return the tampered bytes")
        self.assertTrue(
            any("HMAC mismatch" in m and "allowing load anyway" in m for m in cap.output),
            f"expected warn-mode mismatch log: {cap.output}",
        )

    def test_warn_mode_verify_or_raise_does_not_raise(self) -> None:
        self._set_key()
        p = self._make_model("w2.joblib", b"original")
        mi.sign_model(p)
        with open(p, "wb") as f:
            f.write(b"tampered2")
        self._set_mode("warn")
        self._set_key()
        mi.verify_model_or_raise(p)  # must NOT raise

    def test_off_mode_skips_hmac_on_tampered_file(self) -> None:
        """off-mode -> no HMAC computed, no log, bytes returned regardless."""
        self._set_key()
        p = self._make_model("o1.joblib", b"original")
        mi.sign_model(p)
        with open(p, "wb") as f:
            f.write(b"tampered3")
        self._set_mode("off")
        self._set_key()
        out = mi.verify_and_load_bytes(p)
        self.assertEqual(out, b"tampered3")
        # verify_model_or_raise also short-circuits
        mi.verify_model_or_raise(p)

    def test_off_mode_still_allows_sign_model(self) -> None:
        """off applies only to verify path; sign_model still updates the manifest."""
        self._set_mode("off")
        self._set_key()
        p = self._make_model("o2.joblib", b"payload")
        self.assertTrue(mi.sign_model(p))
        with open(os.path.join(self.tmp_root, "models", "manifest.json")) as fp:
            manifest = json.load(fp)
        self.assertIn("models/o2.joblib", manifest["entries"])

    def test_malformed_entry_raises_even_in_off_mode(self) -> None:
        """Corruption signals always raise — `off` does NOT bypass malformed-entry detection.
        Completes the policy matrix alongside test_malformed_entry_raises_even_in_warn_mode."""
        self._set_mode("off")
        self._set_key()
        p = self._make_model("malformed_off.joblib", b"x")
        manifest_path = os.path.join(self.tmp_root, "models", "manifest.json")
        with open(manifest_path, "w") as fp:
            json.dump({"version": 1, "entries": {
                "models/malformed_off.joblib": {"hmac_sha256": "abc"}  # short hash
            }}, fp)
        # off skips HMAC computation but still validates manifest entry shape,
        # so a corrupted/tampered manifest entry still raises.
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_model_or_raise(p)
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_and_load_bytes(p)

    def test_enforce_mode_still_raises_on_mismatch(self) -> None:
        """Sanity: explicitly setting enforce keeps the existing fail-closed."""
        self._set_key()
        p = self._make_model("e1.joblib", b"original")
        mi.sign_model(p)
        with open(p, "wb") as f:
            f.write(b"tampered_e")
        self._set_mode("enforce")
        self._set_key()
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_and_load_bytes(p)

    def test_malformed_entry_raises_even_in_warn_mode(self) -> None:
        """Malformed manifest entries are corruption — they always raise."""
        self._set_mode("warn")
        self._set_key()
        p = self._make_model("malformed_warn.joblib", b"x")
        manifest_path = os.path.join(self.tmp_root, "models", "manifest.json")
        with open(manifest_path, "w") as fp:
            json.dump({"version": 1, "entries": {
                "models/malformed_warn.joblib": {"hmac_sha256": "abc"}  # short hash
            }}, fp)
        with self.assertRaises(mi.ModelIntegrityError):
            mi.verify_model_or_raise(p)

    def test_mode_is_cached_after_first_read(self) -> None:
        """Mode read once per process; env mutations after first read are ignored."""
        self._set_mode("warn")
        first = mi._load_enforcement_mode()
        self.assertEqual(first, "warn")
        os.environ[mi._ENFORCEMENT_ENV] = "enforce"
        second = mi._load_enforcement_mode()
        self.assertEqual(second, "warn", "mode must be cached; env changes need _reset_for_tests")


if __name__ == "__main__":
    unittest.main()
