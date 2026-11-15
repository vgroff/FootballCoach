"""After scripts/split_module.py: rewrite `from <source> import a, b` statements
across the repo so each name that moved is imported from its new module.

Usage: python scripts/fix_moved_imports.py SOURCE_MODULE MOVED.json [--dry-run]
e.g.   python scripts/fix_moved_imports.py footballcoach.ai.ppo.ppo_trainer plan.moved.json
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

CR, LF = chr(13), chr(10)


def main() -> None:
    source, moved_path = sys.argv[1], Path(sys.argv[2])
    dry = "--dry-run" in sys.argv
    pkg = source.rsplit(".", 1)[0]
    moved = json.loads(moved_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path("src").resolve()))
    src_module = importlib.import_module(source)
    gone = {n: h for n, h in moved.items() if not hasattr(src_module, n)}
    changed_files = 0
    for path in [*Path("src").rglob("*.py"), *Path("tests").rglob("*.py"), *Path("scripts").rglob("*.py")]:
        raw = path.read_bytes().decode("utf-8")
        crlf = CR + LF in raw
        text = raw.replace(CR + LF, LF)
        if source not in text:
            continue
        tree = ast.parse(text)
        lines = text.split(LF)
        edits = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module == source):
                continue
            moving = [a for a in node.names if a.name in gone]
            if not moving:
                continue
            staying = [a for a in node.names if a.name not in gone]
            indent = lines[node.lineno - 1][: len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
            tail = lines[node.end_lineno - 1]
            comment = tail[tail.index("#"):] if "#" in tail and tail.rstrip().endswith(tail[tail.index("#"):].rstrip()) and ")" not in tail[tail.index("#"):] else ""

            def fmt(module: str, aliases: list[ast.alias]) -> str:
                names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in aliases)
                return f"{indent}from {module} import {names}"

            new = []
            if staying:
                new.append(fmt(source, staying))
            by_home: dict[str, list[ast.alias]] = {}
            for a in moving:
                by_home.setdefault(gone[a.name], []).append(a)
            for home, aliases in by_home.items():
                new.append(fmt(f"{pkg}.{home}", aliases))
            if comment:
                new[0] += "  " + comment
            edits.append((node.lineno - 1, node.end_lineno, new))
        if not edits:
            continue
        for start, end, new in sorted(edits, reverse=True):
            lines[start:end] = new
        changed_files += 1
        print(f"{path}: {len(edits)} import(s) rewritten")
        if not dry:
            path.write_bytes((LF.join(lines).replace(LF, CR + LF) if crlf else LF.join(lines)).encode("utf-8"))
    print(f"{changed_files} file(s)")


if __name__ == "__main__":
    main()
