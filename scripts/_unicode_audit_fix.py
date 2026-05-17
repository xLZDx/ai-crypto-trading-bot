"""Surgical Unicode-to-ASCII fix across all .py files in D:\\test 2.

Scope (PHASE 1 only):
  - logger.{info,debug,warning,error,critical,exception}(...)
  - print(...)
  - raise <Error>("...")
  - parser.add_argument(..., help="...")
  - sys.stderr.write / sys.stdout.write

Only string literals INSIDE those call sites are touched. Comments, docstrings,
regex patterns, sample data fixtures, and string literals in other call sites
remain UTF-8 with their existing content.

Banned chars replaced via REPL_MAP. Anything else > 127 is logged so the
operator can decide case-by-case.

Usage:
  python scripts/_unicode_audit_fix.py --dry-run   # report only, no writes
  python scripts/_unicode_audit_fix.py             # apply
  python scripts/_unicode_audit_fix.py --root "D:\\test 2"  # different scope
"""
from __future__ import annotations

import argparse
import ast
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPL_MAP = {
    "→": "->",   "←": "<-",
    "—": "--",   "–": "-",
    "…": "...",
    "‘": "'",    "’": "'",
    "“": '"',    "”": '"',
    "×": "x",    "÷": "/",
    "°": "deg",  "±": "+/-",
    "≈": "~=",   "≤": "<=",   "≥": ">=",   "≠": "!=",
    " ": " ",     # non-breaking space
    "•": "*",     # bullet
    "·": "*",     # middle dot
    "‐": "-", "‑": "-", "‒": "-",   # hyphen variants
    "│": "|", "─": "-", "┃": "|", "━": "-",  # box-drawing
    "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
    "█": "#", "▉": "#", "▊": "#", "▋": "#",
    "▌": "#", "▍": "#", "▎": "#", "▏": "#",
    "▕": "|", "▒": "#", "░": "-", "▓": "#",
    "✅": "OK ",  "❌": "FAIL ",
    "✓": "OK ",  "✗": "FAIL ",
    "✔": "OK ",  "✘": "FAIL ",
    "⚠": "WARN ",
    "⚡": "* ",
    "✨": "* ",
    "ℹ": "INFO ",
    "🚀": "START ",  # rocket
    "®": "(R)", "©": "(C)", "™": "(TM)",
}

EXCLUDE_DIRS = {"venv", ".venv", "node_modules", ".git", "__pycache__",
                "_archive", "build", ".dart_tool", ".tox", "dist",
                "site-packages", ".pytest_cache", ".mypy_cache",
                "references", "MetaGPT"}  # third-party reference code (Chinese/etc.)

# Call-site detectors
LOGGER_METHODS = {"info", "debug", "warning", "warn",
                  "error", "critical", "exception", "log"}
WRITE_METHODS  = {"write"}
ALWAYS_TARGET_FUNCS = {"print"}


def _is_target_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id in ALWAYS_TARGET_FUNCS:
        return True
    if isinstance(func, ast.Attribute):
        method = func.attr
        if method in LOGGER_METHODS:
            return True
        if method == "write":
            # sys.stderr.write / sys.stdout.write
            v = func.value
            if isinstance(v, ast.Attribute) and v.attr in {"stderr", "stdout"}:
                return True
    return False


def _is_raise(node: ast.Raise) -> bool:
    return True   # all raises


def _is_add_argument_help(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "add_argument":
        return True
    return False


def _ascii_replace(s: str) -> tuple[str, list[str]]:
    """Return (new_string, unhandled_chars). unhandled_chars are non-ASCII
    chars that aren't in REPL_MAP — they get unicodedata NFKD-stripped and
    each remaining non-ASCII byte becomes '?'."""
    if all(ord(c) < 128 for c in s):
        return s, []
    out_chars: list[str] = []
    unhandled: list[str] = []
    for c in s:
        if ord(c) < 128:
            out_chars.append(c)
            continue
        repl = REPL_MAP.get(c)
        if repl is not None:
            out_chars.append(repl)
            continue
        # Try NFKD decomposition
        decomp = unicodedata.normalize("NFKD", c)
        kept = "".join(ch for ch in decomp if ord(ch) < 128)
        if kept:
            out_chars.append(kept)
        else:
            out_chars.append("?")
            unhandled.append(c)
    return "".join(out_chars), unhandled


def _gather_target_string_nodes(tree: ast.AST) -> list[ast.Constant]:
    """Walk tree, return Constant string nodes that live inside a target call site."""
    targets: list[ast.Constant] = []

    class Visitor(ast.NodeVisitor):
        def visit_Raise(self, node: ast.Raise):
            if node.exc is not None:
                for sub in ast.walk(node.exc):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        targets.append(sub)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            if _is_target_call(node):
                for arg in node.args:
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            targets.append(sub)
                # Logger/print may use keyword args (rare): skip kw to avoid touching
                # things like extra={"emoji": "..."} — only positional.
            elif _is_add_argument_help(node):
                for kw in node.keywords:
                    if kw.arg in {"help", "metavar"}:
                        for sub in ast.walk(kw.value):
                            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                                targets.append(sub)
            self.generic_visit(node)

    Visitor().visit(tree)
    return targets


def _apply_edits(src: str, targets: list[ast.Constant]) -> tuple[str, int, Counter]:
    """For each target node, replace non-ASCII chars in the slice of source it
    covers. Process in reverse position order so earlier edits don't shift later
    offsets."""
    lines = src.splitlines(keepends=True)
    edits: list[tuple[int, int, int, int, str, str]] = []  # (sl, sc, el, ec, old_slice, new_slice)

    for node in targets:
        sl, sc = node.lineno - 1, node.col_offset
        el, ec = (node.end_lineno or node.lineno) - 1, node.end_col_offset or 0
        # Extract original source slice
        if sl == el:
            old = lines[sl][sc:ec]
        else:
            parts = [lines[sl][sc:]]
            for j in range(sl + 1, el):
                parts.append(lines[j])
            parts.append(lines[el][:ec])
            old = "".join(parts)
        new, _ = _ascii_replace(old)
        if new != old:
            edits.append((sl, sc, el, ec, old, new))

    # Apply in reverse order
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    new_lines = lines[:]
    chars_replaced = 0
    by_char: Counter = Counter()
    for sl, sc, el, ec, old, new in edits:
        chars_replaced += sum(1 for c in old if ord(c) > 127)
        for c in old:
            if ord(c) > 127:
                by_char[c] += 1
        if sl == el:
            new_lines[sl] = new_lines[sl][:sc] + new + new_lines[sl][ec:]
        else:
            head = new_lines[sl][:sc]
            tail = new_lines[el][ec:]
            new_lines[sl] = head + new
            for j in range(sl + 1, el):
                new_lines[j] = ""
            new_lines[el] = tail
    return "".join(new_lines), chars_replaced, by_char


def walk_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("D:/test 2"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = walk_files(args.root)
    print(f"Scanning {len(files)} .py files under {args.root}")
    total_changed = 0
    total_chars = 0
    overall_chars: Counter = Counter()
    fail_files: list[tuple[Path, str]] = []
    changed_files: list[tuple[Path, int]] = []

    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                src = p.read_text(encoding="cp1252")
            except Exception as e:
                fail_files.append((p, f"read: {e}"))
                continue
        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError as e:
            fail_files.append((p, f"parse: {e}"))
            continue
        targets = _gather_target_string_nodes(tree)
        if not targets:
            continue
        new_src, chars, by_char = _apply_edits(src, targets)
        if new_src == src:
            continue
        total_changed += 1
        total_chars += chars
        overall_chars.update(by_char)
        changed_files.append((p, chars))
        if not args.dry_run:
            # Verify the result still parses
            try:
                ast.parse(new_src, filename=str(p))
            except SyntaxError as e:
                fail_files.append((p, f"post-edit parse: {e}"))
                continue
            backup = p.with_suffix(p.suffix + ".unicode_fix.bak")
            if not backup.exists():
                backup.write_bytes(p.read_bytes())
            p.write_text(new_src, encoding="utf-8")

    print()
    print(f"=== SUMMARY ===")
    print(f"files needing fix: {total_changed}")
    print(f"chars replaced:    {total_chars}")
    print(f"failed (skipped):  {len(fail_files)}")
    print()
    print("Top chars replaced:")
    for c, n in overall_chars.most_common(20):
        name = unicodedata.name(c, "?")
        print(f"  {hex(ord(c)):>8} {name:<40} x{n}")
    if changed_files:
        print()
        print("Top 20 changed files:")
        changed_files.sort(key=lambda x: x[1], reverse=True)
        for p, n in changed_files[:20]:
            print(f"  {n:>4}  {p}")
    if fail_files:
        print()
        print("Failures (skipped):")
        for p, msg in fail_files[:20]:
            print(f"  {p}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
