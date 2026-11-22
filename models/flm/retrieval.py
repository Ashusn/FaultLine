import re
from pathlib import Path

from .common import ROOT
from .render import signals_query
from .schema import FAULT_KINDS

HEAD = re.compile(r"^####\s+(fault|service|app)\s+([A-Za-z0-9_.\-]+)\s*$", re.I)
CODE_START = re.compile(r"^(func |def |class |export function |public |private )")


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def runbook_docs(dirs):
    docs = []
    for d in dirs:
        d = _resolve(d)
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.md")):
            cur, buf = None, []

            def flush():
                if cur and buf:
                    kind, value = cur
                    text = "\n".join(buf).strip()
                    if text:
                        docs.append({"id": f"runbook:{f.stem}:{kind}:{value}", "kind": f"runbook_{kind}", "meta": {kind: value}, "text": f"{kind} {value}\n{text}"})

            for line in f.read_text().splitlines():
                m = HEAD.match(line)
                if m:
                    flush()
                    cur, buf = (m.group(1).lower(), m.group(2).lower()), []
                elif cur:
                    buf.append(line)
            flush()
    return docs


def code_docs(dirs, services, max_lines=60, exts=(".go", ".py", ".ts", ".java")):
    docs = []
    known = sorted(services, key=len, reverse=True)
    for d in dirs:
        d = _resolve(d)
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if f.suffix not in exts or not f.is_file() or "test" in f.name:
                continue
            rel = f.relative_to(d).as_posix().lower()
            svc = next((s for s in known if s in rel), None)
            if not svc:
                continue
            lines = f.read_text(errors="ignore").splitlines()
            starts = [i for i, l in enumerate(lines) if CODE_START.match(l)] or [0]
            starts.append(len(lines))
            for a, b in zip(starts, starts[1:]):
                for s in range(a, b, max_lines):
                    chunk = "\n".join(lines[s : min(b, s + max_lines)]).strip()
                    if len(chunk) > 40:
                        docs.append({"id": f"code:{rel}:{s + 1}", "kind": "code", "meta": {"service": svc, "path": rel}, "text": f"file {rel}\n{chunk}"})
    return docs


def incident_doc(inc):
    lab = inc["label"]
    return {
        "id": f"incident:{inc['incident_id']}",
        "kind": "incident",
        "meta": {"service": lab["service"], "fault": lab["fault"], "app": inc["app"]},
        "text": f"past incident in {inc['app']}: root cause {lab['fault']} in {lab['service']}. symptoms: {signals_query(inc['signals'], k=6)}",
    }


def relevant(inc, corpus, max_code=3):
    lab = inc["label"]
    out, code = [], []
    for d in corpus:
        m = d["meta"]
        if d["kind"] == "runbook_fault" and m.get("fault") == lab["fault"]:
            out.append(d["id"])
        elif d["kind"] == "runbook_service" and m.get("service") == lab["service"]:
            out.append(d["id"])
        elif d["kind"] == "incident" and m.get("service") == lab["service"] and m.get("fault") == lab["fault"] and m.get("app") == inc["app"] and d["id"] != f"incident:{inc['incident_id']}":
            out.append(d["id"])
        elif d["kind"] == "code" and m.get("service") == lab["service"]:
            code.append(d["id"])
    return out + code[:max_code]


def build(cfg, splits):
    r = cfg["retrieval"]
    services = sorted({s for v in splits.values() for i in v for s in i["services"]})
    corpus = runbook_docs(r["runbook_dirs"]) + code_docs(r.get("code_dirs", []), services)
    corpus += [incident_doc(i) for i in splits["train"]]
    seen, uniq = set(), []
    for d in corpus:
        if d["id"] not in seen:
            seen.add(d["id"])
            uniq.append(d)
    corpus = uniq
    queries = {}
    for name, incs in splits.items():
        rows = []
        for inc in incs:
            rel = relevant(inc, corpus, r.get("max_code_positives", 3))
            if not rel:
                continue
            rows.append({"id": f"q:{inc['incident_id']}", "incident_id": inc["incident_id"], "text": signals_query(inc["signals"], k=r.get("query_anomalies", 5)), "relevant": rel, "fault": inc["label"]["fault"], "service": inc["label"]["service"]})
        queries[name] = rows
    pairs = []
    by_id = {d["id"]: d for d in corpus}
    for q in queries["train"]:
        for did in q["relevant"][: r.get("max_positives_per_query", 4)]:
            pairs.append({"anchor": q["text"], "positive": by_id[did]["text"], "qid": q["id"], "did": did})
    return corpus, queries, pairs


def check_runbooks(corpus):
    faults = {d["meta"].get("fault") for d in corpus if d["kind"] == "runbook_fault"}
    return sorted(set(FAULT_KINDS) - faults)
