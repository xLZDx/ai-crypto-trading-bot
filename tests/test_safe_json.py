"""F1 — Behavioral tests for src/utils/safe_json.py.

Every test invokes the real function and asserts on observable outcomes
(file contents, return values, error propagation).  No string-matching.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.safe_json import read_json, write_json  # noqa: E402


class TestWriteJson(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "data.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_write_and_read_roundtrip(self) -> None:
        payload = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
        write_json(self.path, payload)
        result = read_json(self.path)
        self.assertEqual(result, payload)

    def test_write_creates_parent_dirs(self) -> None:
        deep = os.path.join(self.tmp.name, "a", "b", "c", "file.json")
        write_json(deep, {"x": 1})
        self.assertTrue(os.path.isfile(deep))
        self.assertEqual(read_json(deep), {"x": 1})

    def test_write_is_atomic_no_tmp_file_left(self) -> None:
        write_json(self.path, {"done": True})
        tmp_files = [f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")]
        self.assertEqual(tmp_files, [], f"leftover .tmp files: {tmp_files}")

    def test_write_overwrites_previous_content(self) -> None:
        write_json(self.path, {"v": 1})
        write_json(self.path, {"v": 2})
        self.assertEqual(read_json(self.path)["v"], 2)

    def test_write_produces_valid_json(self) -> None:
        write_json(self.path, [1, "two", None, True])
        with open(self.path, encoding="utf-8") as f:
            parsed = json.load(f)
        self.assertEqual(parsed, [1, "two", None, True])

    def test_write_handles_unicode(self) -> None:
        write_json(self.path, {"emoji": "✓", "cjk": "日本語"})
        result = read_json(self.path)
        self.assertEqual(result["emoji"], "✓")
        self.assertEqual(result["cjk"], "日本語")


class TestReadJson(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "data.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_read_missing_file_returns_default(self) -> None:
        result = read_json(self.path, default={"fallback": True})
        self.assertEqual(result, {"fallback": True})

    def test_read_missing_file_default_none(self) -> None:
        self.assertIsNone(read_json(self.path))

    def test_read_corrupt_json_returns_default(self) -> None:
        with open(self.path, "w") as f:
            f.write("{not valid json")
        result = read_json(self.path, default={"safe": True})
        self.assertEqual(result, {"safe": True})

    def test_read_valid_file(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"hello": "world"}, f)
        self.assertEqual(read_json(self.path), {"hello": "world"})


class TestConcurrentWriters(unittest.TestCase):
    """Concurrent write_json calls must not corrupt the file or leave
    partial/intermediate content.  The filelock + atomic rename guarantees
    every read sees a complete, valid JSON document."""

    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "shared.json")
        # Seed with initial value
        write_json(self.path, {"counter": 0})

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_10_concurrent_writers_all_succeed(self) -> None:
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                write_json(self.path, {"counter": i, "writer": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [], f"write_json raised under concurrency: {errors}")
        # File must still be valid JSON after all writes
        result = read_json(self.path)
        self.assertIsInstance(result, dict)
        self.assertIn("counter", result)
        self.assertIn("writer", result)

    def test_concurrent_writes_leave_no_tmp_files(self) -> None:
        def worker(i: int) -> None:
            write_json(self.path, {"i": i})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        tmp_files = [f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")]
        self.assertEqual(tmp_files, [], f"orphaned .tmp files after concurrent writes: {tmp_files}")

    def test_reader_never_sees_partial_content(self) -> None:
        """While 5 threads hammer write_json, a concurrent reader must always
        get back a fully-parseable JSON dict — never a truncated / corrupt file.

        The file is pre-seeded so read_json never has a legitimate reason to
        return None (the file always exists); any None return during the write
        storm means a torn read was silently caught and swallowed — that is a
        failure, not a benign miss.
        """
        read_errors: list[str] = []
        done_event = threading.Event()
        # Pre-seed the file so the reader always has a valid starting point.
        write_json(self.path, {"seq": -1, "padding": "seed"})

        def reader() -> None:
            while not done_event.is_set():
                data = read_json(self.path, default=None)
                if data is None:
                    # File exists but was unreadable — torn read or broken atomicity.
                    read_errors.append("read_json returned None (file should always be parseable)")
                elif not isinstance(data, dict):
                    read_errors.append(f"unexpected type: {type(data)}")
                time.sleep(0.001)

        def writer(i: int) -> None:
            for _ in range(20):
                write_json(self.path, {"seq": i, "padding": "x" * 200})

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        writers = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in writers:
            t.start()
        for t in writers:
            t.join(timeout=15)
        done_event.set()
        reader_thread.join(timeout=2)

        self.assertEqual(read_errors, [], f"reader saw malformed data: {read_errors}")


class TestAtomicRename(unittest.TestCase):
    """write_json must use os.replace so the original is never visible as
    zero-length or partially written, even on Windows where os.replace
    is the atomic-rename primitive."""

    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "atomic.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_original_intact_if_write_fails(self) -> None:
        """If the serialization step raises (e.g. unserializable object),
        the original file must be left untouched."""
        original = {"original": True}
        write_json(self.path, original)

        with self.assertRaises(Exception):
            write_json(self.path, object())  # not JSON-serializable

        result = read_json(self.path)
        self.assertEqual(result, original)

    def test_subsequent_writes_always_produce_valid_json(self) -> None:
        """Repeated writes to the same path must always leave a valid JSON file."""
        for i in range(20):
            write_json(self.path, {"iteration": i, "data": "x" * 50})
        result = read_json(self.path)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["iteration"], 19)


if __name__ == "__main__":
    unittest.main()
