import hashlib

from .schema import group_key

SPLITS = ["train", "val", "test", "test_ood_fault", "test_ood_app", "test_faultline"]


def _bucket(key, seed):
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_incidents(incidents, cfg):
    s = cfg["split"]
    seed = cfg.get("seed", 0)
    hold_fault = set(s.get("holdout_faults") or [])
    hold_app = set(s.get("holdout_apps") or [])
    dedicated = set(s.get("dedicated_sources") or [])
    val_frac, test_frac = float(s["val"]), float(s["test"])
    out = {k: [] for k in SPLITS}
    for inc in incidents:
        lab = inc["label"]
        if inc["source"] in dedicated:
            out["test_faultline"].append(inc)
            continue
        if inc["app"] in hold_app:
            out["test_ood_app"].append(inc)
            continue
        if lab["fault"] in hold_fault:
            out["test_ood_fault"].append(inc)
            continue
        u = _bucket(group_key(inc), seed)
        if u < test_frac:
            out["test"].append(inc)
        elif u < test_frac + val_frac:
            out["val"].append(inc)
        else:
            out["train"].append(inc)
    return out


def leak_report(splits, cfg):
    s = cfg["split"]
    groups = {k: {group_key(i) for i in v} for k, v in splits.items()}
    ids = {k: {i["incident_id"] for i in v} for k, v in splits.items()}
    problems = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if ids[a] & ids[b]:
                problems.append(f"incident ids shared by {a} and {b}")
            if groups[a] & groups[b] and "test_faultline" not in (a, b):
                problems.append(f"groups shared by {a} and {b}: {sorted(groups[a] & groups[b])[:3]}")
    train_faults = {i["label"]["fault"] for i in splits["train"]}
    train_apps = {i["app"] for i in splits["train"]}
    for src in s.get("dedicated_sources") or []:
        if any(i["source"] == src for n in ("train", "val") for i in splits[n]):
            problems.append(f"dedicated source {src} found in train or val")
    for f in s.get("holdout_faults") or []:
        if f in train_faults:
            problems.append(f"holdout fault {f} found in train")
    for a in s.get("holdout_apps") or []:
        if a in train_apps:
            problems.append(f"holdout app {a} found in train")
    sizes = {k: len(v) for k, v in splits.items()}
    return {"ok": not problems, "problems": problems, "sizes": sizes}
