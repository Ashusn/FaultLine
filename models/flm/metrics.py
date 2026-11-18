import math
import random


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def mrr_at_k(ranked, relevant, k):
    rel = set(relevant)
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(values, n=1000, alpha=0.05, seed=0):
    values = list(values)
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    k = len(values)
    stats = sorted(mean(values[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return [round(stats[int(alpha / 2 * n)], 4), round(stats[min(n - 1, int((1 - alpha / 2) * n))], 4)]


def paired_delta_ci(a, b, n=1000, alpha=0.05, seed=0):
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    d = [x - y for x, y in zip(a, b)]
    return round(mean(d), 4), bootstrap_ci(d, n, alpha, seed)


def summarize(items, keys, ci=True, seed=0):
    out = {}
    for k in keys:
        vals = [float(it[k]) for it in items if it.get(k) is not None]
        out[k] = round(mean(vals), 4)
        if ci:
            out[f"{k}_ci95"] = bootstrap_ci(vals, seed=seed)
    out["n"] = len(items)
    return out
