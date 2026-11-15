"""Move methods of a class (and the module-level helpers only they use) into
mixin modules, verbatim. Built to split ai/ppo/ppo_trainer.py's PPOTrainer
without changing behaviour; see scripts/golden_snapshot.py for the check.

Usage: python scripts/split_module.py PLAN.json [--dry-run]

PLAN.json:
{
  "source": "src/footballcoach/ai/ppo/ppo_trainer.py",
  "package": "footballcoach.ai.ppo",
  "class": "PPOTrainer",
  "common": "trainer_common",
  "mixins": {"checkpoint_io": {"class": "CheckpointIOMixin", "doc": "...", "methods": ["..."]}}
}

Each module-level helper goes to the one module that uses it (the source
module counts as a user when the class's remaining methods, other helpers
staying there, or anything else in the source uses it), or to the common
module when several modules do. Imports are rebuilt from the names each piece
of moved code actually uses.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

CR, LF = chr(13), chr(10)


def read_text(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes().decode("utf-8")
    return raw.replace(CR + LF, LF), CR + LF in raw


def write_text(path: Path, text: str, crlf: bool) -> None:
    path.write_bytes((text.replace(LF, CR + LF) if crlf else text).encode("utf-8"))


def loads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def bound_names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    return []


def spans(lines: list[str], body: list[ast.stmt], floor: int) -> list[tuple[int, int]]:
    """(start, end) 0-based inclusive line spans, each starting right after the
    previous statement so the comments above a statement travel with it."""
    out = []
    prev_end = floor
    for node in body:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        start = prev_end
        while start < first and not lines[start].strip():
            start += 1
        out.append((start, node.end_lineno - 1))
        prev_end = node.end_lineno
    return out


def import_lines(imports: list[ast.stmt], needed: set[str]) -> list[str]:
    out = []
    for node in imports:
        if isinstance(node, ast.ImportFrom):
            keep = [a for a in node.names if (a.asname or a.name) in needed]
            if keep:
                names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
                out.append(f"from {'.' * node.level}{node.module or ''} import {names}")
        else:
            keep = [a for a in node.names if (a.asname or a.name.split(".")[0]) in needed]
            for a in keep:
                out.append(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
    return out


def main() -> None:
    plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    dry = "--dry-run" in sys.argv
    src_path = Path(plan["source"])
    text, crlf = read_text(src_path)
    lines = text.split(LF)
    tree = ast.parse(text)
    pkg = plan["package"]
    src_mod = src_path.stem
    common = plan["common"]

    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    imported = {(a.asname or a.name.split(".")[0]) for n in imports for a in n.names}
    helpers = [n for n in tree.body if bound_names(n) and not (isinstance(n, ast.ClassDef) and n.name == plan["class"])]
    helper_of = {name: node for node in helpers for name in bound_names(node)}
    module_spans = dict(zip(map(id, tree.body), spans(lines, tree.body, 0)))

    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == plan["class"])
    cls_body_start = cls.body[0].lineno - 1
    method_spans = dict(zip(map(id, cls.body), spans(lines, cls.body, cls_body_start)))
    methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    home_of_method = {}
    for mod, spec in plan["mixins"].items():
        for m in spec["methods"]:
            if m not in methods:
                sys.exit(f"unknown method {m}")
            if m in home_of_method:
                sys.exit(f"{m} planned twice")
            home_of_method[m] = mod

    problems = []
    for m, mod in home_of_method.items():
        used = loads(methods[m])
        for bad in (plan["class"], *[n.name for n in tree.body if isinstance(n, ast.ClassDef)]):
            if bad in used and bad not in helper_of:
                problems.append(f"{m} -> {mod} refers to {bad}")
        if plan["class"] in used:
            problems.append(f"{m} -> {mod} refers to {plan['class']} by name")

    # Users of each helper: modules whose moved/remaining code loads it.
    users: dict[str, set[str]] = {name: set() for name in helper_of}
    staying_class_nodes = [n for n in cls.body if not (isinstance(n, ast.FunctionDef) and n.name in home_of_method)]
    for node in staying_class_nodes:
        for name in loads(node) & users.keys():
            users[name].add(src_mod)
    for m, mod in home_of_method.items():
        for name in loads(methods[m]) & users.keys():
            users[name].add(mod)
    other_top = [n for n in tree.body if not bound_names(n) and not isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in other_top:
        for name in loads(node) & users.keys():
            users[name].add(src_mod)
    for name in plan.get("keep_in_source", []):
        users[name].add(src_mod)
    # Helpers used by helpers: propagate until stable.
    home: dict[str, str] = {}
    for _ in range(50):
        changed = False
        for name, node in helper_of.items():
            u = users[name]
            new_home = (next(iter(u)) if len(u) == 1 else (src_mod if not u else common))
            if home.get(name) != new_home:
                home[name] = new_home
                changed = True
        for name, node in helper_of.items():
            for dep in loads(node) & users.keys():
                if dep != name and home[name] not in users[dep]:
                    users[dep].add(home[name])
                    changed = True
        if not changed:
            break
    # A common helper that depends on a helper living elsewhere would import back into it.
    for name, h in home.items():
        if h == common:
            for dep in loads(helper_of[name]) & helper_of.keys():
                if home[dep] not in (common,):
                    users[dep].add(common)
    for _ in range(50):
        changed = False
        for name in helper_of:
            u = users[name]
            new_home = next(iter(u)) if len(u) == 1 else (src_mod if not u else common)
            if home[name] != new_home:
                home[name] = new_home
                changed = True
        if not changed:
            break
    # Moved code may only use names it can import: other helpers or the source's
    # imports. A class still defined in the source (e.g. the split class itself)
    # would have to be imported back from it, which is a cycle.
    staying_classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)} - {
        name for name in helper_of if home.get(name) not in (None, src_mod)
    }
    def locally_imported(node: ast.AST) -> set[str]:
        return {(a.asname or a.name.split(".")[0]) for n in ast.walk(node)
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}

    for name, node in helper_of.items():
        if home[name] != src_mod:
            for bad in loads(node) & staying_classes - locally_imported(node):
                problems.append(f"helper {name} -> {home[name]} refers to {bad}, which stays in {src_mod}")
    if src_mod in {home[n] for n in helper_of} and any(home[d] == src_mod for n in helper_of if home[n] != src_mod for d in loads(helper_of[n]) & helper_of.keys()):
        problems.append("a moved helper depends on a helper that stays in the source module")

    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  ", p)
        sys.exit(1)

    moved_names = {n for n, h in home.items() if h != src_mod}
    print("helper homes (moved):")
    for name in sorted(moved_names):
        print(f"   {name} -> {home[name]}  (users: {sorted(users[name])})")

    def build(mod: str, helper_nodes: list[ast.stmt], method_nodes: list[ast.stmt], spec: dict | None) -> str:
        code_nodes = helper_nodes + method_nodes
        needed = set().union(*[loads(n) for n in code_nodes]) if code_nodes else set()
        own = {name for n in helper_nodes for name in bound_names(n)}
        if method_nodes:
            for n in method_nodes:
                needed |= {d.id for dec in getattr(n, "decorator_list", []) for d in ast.walk(dec) if isinstance(d, ast.Name)}
        out = []
        doc = spec["doc"] if spec else plan.get("common_doc", "Helpers shared by the PPOTrainer mixin modules.")
        out.append(f'"""{doc}"""')
        out.append("from __future__ import annotations")
        out.append("")
        out.extend(import_lines(imports, needed - own))
        from_elsewhere: dict[str, list[str]] = {}
        for name in sorted((needed - own) & helper_of.keys()):
            from_elsewhere.setdefault(home[name], []).append(name)
        for other, names in sorted(from_elsewhere.items()):
            if other == mod:
                continue
            if other == src_mod:
                sys.exit(f"{mod} would import {names} back from {src_mod}")
            out.append(f"from {pkg}.{other} import {', '.join(names)}")
        out.append("")
        for n in helper_nodes:
            s, e = module_spans[id(n)]
            out.append("")
            out.extend(lines[s:e + 1])
        if method_nodes:
            out.append("")
            out.append("")
            out.append(f"class {spec['class']}:")
            out.append(f'    """{spec["class_doc"]}"""')
            for n in method_nodes:
                s, e = method_spans[id(n)]
                out.append("")
                out.extend(lines[s:e + 1])
        return LF.join(out).rstrip() + LF

    new_files: dict[str, str] = {}
    targets = list(plan["mixins"]) + [common]
    for mod in targets:
        helper_nodes = [n for n in helpers if any(home[b] == mod for b in bound_names(n))]
        method_nodes = [methods[m] for m in plan["mixins"].get(mod, {}).get("methods", [])]
        method_nodes.sort(key=lambda n: n.lineno)
        if not helper_nodes and not method_nodes:
            continue
        new_files[mod] = build(mod, helper_nodes, method_nodes, plan["mixins"].get(mod))

    # Rewrite the source: drop moved helpers and methods, add mixin bases and imports.
    drop: list[tuple[int, int]] = []
    for n in helpers:
        if any(home[b] != src_mod for b in bound_names(n)):
            drop.append(module_spans[id(n)])
    for m in home_of_method:
        drop.append(method_spans[id(methods[m])])
    keep_line = [True] * len(lines)
    for s, e in drop:
        for i in range(s, e + 1):
            keep_line[i] = False
    remaining_text = LF.join(l for l, k in zip(lines, keep_line) if k)
    rem_tree = ast.parse(remaining_text)
    rem_cls = next(n for n in rem_tree.body if isinstance(n, ast.ClassDef) and n.name == plan["class"])
    rem_needed = loads(rem_tree)

    mixin_classes = [(mod, spec["class"]) for mod, spec in plan["mixins"].items()]
    rem_lines = remaining_text.split(LF)
    cls_line = rem_cls.lineno - 1
    bases = [spec_class for _, spec_class in mixin_classes] + [ast.unparse(b) for b in rem_cls.bases]
    header = rem_lines[cls_line]
    assert header.startswith(f"class {plan['class']}"), header
    rem_lines[cls_line] = f"class {plan['class']}({', '.join(bases)}):"

    new_imports = [f"from {pkg}.{mod} import {klass}" for mod, klass in mixin_classes]
    by_home: dict[str, list[str]] = {}
    for name in sorted(rem_needed & moved_names):
        by_home.setdefault(home[name], []).append(name)
    for h, names in sorted(by_home.items()):
        new_imports.append(f"from {pkg}.{h} import {', '.join(names)}")
    last_import = max(n.end_lineno for n in rem_tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    rem_lines[last_import:last_import] = new_imports
    final_text = LF.join(rem_lines)

    # Drop now-unused imports from the source (names nothing left there loads).
    final_tree = ast.parse(final_text)
    used_now = loads(final_tree) | {n.value.id for n in ast.walk(final_tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    final_lines = final_text.split(LF)
    for node in sorted([n for n in final_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))], key=lambda n: -n.lineno):
        keep = [a for a in node.names if (a.asname or a.name.split(".")[0]) in used_now or (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
        if len(keep) == len(node.names):
            continue
        if not keep:
            replacement = []
        elif isinstance(node, ast.ImportFrom):
            replacement = [f"from {'.' * node.level}{node.module or ''} import {', '.join(a.name + (f' as {a.asname}' if a.asname else '') for a in keep)}"]
        else:
            replacement = [f"import {a.name}" + (f" as {a.asname}" if a.asname else "") for a in keep]
        final_lines[node.lineno - 1:node.end_lineno] = replacement
    final_text = LF.join(final_lines)

    print("new modules:", {m: t.count(LF) for m, t in new_files.items()})
    print(f"{src_path.name}: {len(lines)} -> {final_text.count(LF) + 1} lines")
    if dry:
        return
    for mod, body in new_files.items():
        path = src_path.with_name(f"{mod}.py")
        if path.exists() and mod != common:
            sys.exit(f"{path} already exists")
        if path.exists():
            existing, _ = read_text(path)
            sys.exit(f"{path} already exists; merge needed")
        write_text(path, body, crlf)
    write_text(src_path, final_text, crlf)
    moved_report = {name: home[name] for name in sorted(moved_names)}
    Path(sys.argv[1]).with_suffix(".moved.json").write_text(json.dumps(moved_report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
