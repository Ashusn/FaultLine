import json
import time
from pathlib import Path

import numpy as np

from ..common import read_json, read_jsonl, set_seed, wdir, wpath, write_json
from ..metrics import summarize
from ..signals import FEATURES, rule_rank, service_features

KEYS = ["top1", "top3", "fault_acc"]


def _incidents(cfg, split, n=None):
    from ..data import load_split

    return load_split(cfg, split, n)


def _global(signals):
    g = {}
    for a in signals.get("anomalies", []):
        k = a.get("signal", "other")
        g[k] = g.get(k, 0.0) + float(a.get("severity", 0) or 0)
    return g


def fault_features(signals, services, svc):
    f = service_features(signals, services)[svc]
    g = _global(signals)
    return f + [g.get(k.replace("_chg", ""), 0.0) for k in FEATURES if k.endswith("_chg")]


def train(cfg, name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib

    m = cfg["models"][name]
    set_seed(cfg.get("seed", 0))
    X, y, Xf, yf = [], [], [], []
    for inc in _incidents(cfg, "train"):
        feats = service_features(inc["signals"], inc["services"])
        for s, v in feats.items():
            X.append(v)
            y.append(1 if s == inc["label"]["service"] else 0)
        Xf.append(fault_features(inc["signals"], inc["services"], inc["label"]["service"]))
        yf.append(inc["label"]["fault"])
    params = {"max_iter": int(m.get("max_iter", 300)), "learning_rate": float(m.get("lr", 0.05)), "max_leaf_nodes": int(m.get("max_leaf_nodes", 31)), "random_state": int(cfg.get("seed", 0))}
    svc_model = HistGradientBoostingClassifier(class_weight="balanced", **params).fit(np.asarray(X), np.asarray(y))
    fault_model = HistGradientBoostingClassifier(**params).fit(np.asarray(Xf), np.asarray(yf))
    out = wdir(cfg, "models", name)
    joblib.dump(svc_model, out / "service_ranker.joblib")
    joblib.dump(fault_model, out / "fault_classifier.joblib")
    write_json(out / "config.json", {"type": "faultline_classifier", "features": FEATURES, "faults": sorted(set(yf)), "params": params, "train_incidents": len(yf)})
    return {"local": str(out), "training": None}


class Classifier:
    def __init__(self, path):
        import joblib

        p = Path(path)
        self.svc = joblib.load(p / "service_ranker.joblib")
        self.fault = joblib.load(p / "fault_classifier.joblib")
        self.meta = read_json(p / "config.json")

    def predict(self, signals, services):
        feats = service_features(signals, services)
        names = list(feats)
        probs = self.svc.predict_proba(np.asarray([feats[s] for s in names]))[:, 1]
        order = [names[i] for i in np.argsort(-probs, kind="stable")]
        fp = self.fault.predict_proba(np.asarray([fault_features(signals, services, order[0])]))[0]
        faults = [str(self.fault.classes_[i]) for i in np.argsort(-fp)]
        return {"services": order, "service_scores": {names[i]: float(probs[i]) for i in range(len(names))}, "faults": faults[:3], "fault_scores": {str(c): float(p) for c, p in zip(self.fault.classes_, fp)}}


def _signals_for(cfg, inc, source):
    if source == "reference":
        return inc["signals"]
    p = wpath(cfg, "results", f"{source}_samples_index.json")
    if not p.exists():
        idx = {}
        for f in sorted(Path(wpath(cfg, "results", "x").parent).glob(f"{source}_*_samples.jsonl")):
            for r in read_jsonl(f):
                idx[r["id"]] = r.get("pred")
        write_json(p, idx)
    return read_json(p).get(inc["incident_id"]) or {"anomalies": []}


def evaluate(cfg, name, ref, split, n=None, source="reference"):
    incs = _incidents(cfg, split, n)
    if not incs:
        return None
    clf = None if ref == "rule" else Classifier(ref)
    items = []
    t = time.time()
    for inc in incs:
        sig = _signals_for(cfg, inc, source)
        if clf:
            p = clf.predict(sig, inc["services"])
            order, fault = p["services"], p["faults"][0]
        else:
            order, fault = rule_rank(sig, inc["services"]), None
        items.append({"id": inc["incident_id"], "top1": 1.0 if order[0] == inc["label"]["service"] else 0.0, "top3": 1.0 if inc["label"]["service"] in order[:3] else 0.0, "fault_acc": (1.0 if fault == inc["label"]["fault"] else 0.0) if fault else None, "fault": inc["label"]["fault"]})
    ms = (time.time() - t) * 1000 / len(incs)
    s = summarize(items, KEYS, seed=cfg.get("seed", 0))
    s["ms_per_incident"] = round(ms, 3)
    by = {}
    for it in items:
        by.setdefault(it["fault"], []).append(it["top1"])
    s["top1_by_fault"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())}
    return {"items": items, "summary": s}


def study(cfg, name, ref):
    out = {}
    sources = ["reference"] + [x for x in cfg["models"][name].get("study_sources", []) if list(Path(wpath(cfg, "results", "x").parent).glob(f"{x}_*_samples.jsonl"))]
    for split in cfg["eval"]["full_splits"]:
        row = {}
        for src in sources:
            for label, r in (("rule", "rule"), ("classifier", ref)):
                res = evaluate(cfg, name, r, split, source=src)
                if res:
                    row[f"{label}_on_{src}"] = {k: res["summary"].get(k) for k in ("top1", "top3", "fault_acc", "ms_per_incident")}
        if row:
            out[split] = row
    write_json(wpath(cfg, "results", f"{name}_study.json"), out)
    return out
