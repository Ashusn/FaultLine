import subprocess
from pathlib import Path

import yaml

from .common import ROOT


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        out[prefix[:-1]] = d
    else:
        out[prefix[:-1]] = d
    return out


def head_version(path):
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not top:
        return None
    rel = Path(path).resolve().relative_to(Path(top).resolve()).as_posix()
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True)
    return yaml.safe_load(r.stdout) if r.returncode == 0 else None


def compare(old, new):
    a, b = flatten(old or {}), flatten(new or {})
    changes = []
    for k in sorted(set(a) | set(b)):
        if k.startswith("_"):
            continue
        if a.get(k, "<missing>") != b.get(k, "<missing>"):
            changes.append((k, a.get(k, "<missing>"), b.get(k, "<missing>")))
    return changes


def run(paths):
    total = 0
    for p in paths:
        new = yaml.safe_load(Path(p).read_text())
        old = head_version(p)
        if old is None:
            print(f"{p}: not in git yet, nothing to compare")
            continue
        ch = compare(old, new)
        total += len(ch)
        print(f"{p}: {len(ch)} changed values against HEAD")
        for k, o, n in ch:
            print(f"  {k}: {o} -> {n}")
    if total:
        print("Read every line above. If any change is not one you meant, run: git checkout -- <config>")
    return total
