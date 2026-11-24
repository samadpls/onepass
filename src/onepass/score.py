"""Typed Decisions metrics, matching LocalLLaMA/typed-decisions bench/score_with_jobs.py."""
import math


def score_one(probabilities, gold):
    """(correct, KL from gold, Brier, confidence) for one decision."""
    opts = list(gold["probabilities"])
    g = [float(gold["probabilities"][o]) for o in opts]
    if gold["type"] == "noul":
        probabilities = {{"yes": "true", "no": "false"}.get(k, k): v for k, v in probabilities.items()}
    p = [float(probabilities.get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return p.index(max(p)) == g.index(max(g)), kl, brier, max(p)


def summarize(rows, n_bins=10):
    """rows: (correct, kl, brier, confidence) tuples -> the four benchmark numbers."""
    n = len(rows)
    hits, total, conf = [0] * n_bins, [0] * n_bins, [0.0] * n_bins
    for ok, _, _, c in rows:
        b = min(int(c * n_bins), n_bins - 1)
        total[b] += 1
        hits[b] += int(ok)
        conf[b] += c
    ece = sum(total[b] / n * abs(hits[b] / total[b] - conf[b] / total[b]) for b in range(n_bins) if total[b])
    return {
        "accuracy": sum(r[0] for r in rows) / n,
        "kl_from_gold": sum(r[1] for r in rows) / n,
        "brier": sum(r[2] for r in rows) / n,
        "ece": ece,
        "n": n,
    }
