import os
import re

from .common import log
from .schema import FAULT_KINDS

PROMPT = (
    "You are an on call engineer. Rewrite these incident symptoms as one short question you would type into a search box "
    "while investigating. Keep every service name and number you use exactly as given. Do not guess or name the root cause."
)

BANNED = sorted({f for k in FAULT_KINDS if k not in ("latency", "error") for f in (k, k.replace("_", " "))} | {"root cause", "caused by", "because"})


def client(cfg):
    from openai import OpenAI

    s = cfg["retrieval"]["synthetic"]
    return OpenAI(base_url=s["base_url"], api_key=os.environ.get(s.get("api_key_env", "SYNTH_API_KEY"), "none"))


def keep(text, services):
    t = text.strip().strip('"')
    if not (20 <= len(t) <= 400):
        return None
    low = t.lower()
    if not any(s.lower() in low for s in services):
        return None
    if any(re.search(rf"\b{re.escape(b)}\b", low) for b in BANNED):
        return None
    return t


def paraphrase(cfg, queries, services):
    s = cfg["retrieval"]["synthetic"]
    c = client(cfg)
    out, seen, dropped = [], set(), 0
    for q in queries:
        for _ in range(int(s.get("per_query", 1))):
            r = c.chat.completions.create(
                model=s["model"],
                temperature=float(s.get("temperature", 0.8)),
                max_tokens=120,
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": q["text"]}],
            )
            t = keep(r.choices[0].message.content or "", services)
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append({**q, "id": f"{q['id']}:syn{len(out)}", "text": t, "synthetic": True})
            else:
                dropped += 1
    log("synthetic queries kept", len(out), "dropped", dropped)
    return out
