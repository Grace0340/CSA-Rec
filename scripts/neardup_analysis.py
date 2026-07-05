"""Quantify the near-duplicate behaviour the case study exposes qualitatively.

The CSA-Rec content pathway rewards items that are semantically close to what a
user already engaged with, which occasionally surfaces near-identical products
(the "Memory Card" case in the paper). This script measures how large that
surface is, and what a simple novelty post-filter would remove. It is purely
offline: it needs only the reproducible split (from data.py) and the cached
frozen semantics (.npy), so it runs with numpy alone once the encoder cache
exists.

It reports, per catalog:
  * catalog near-duplication -- fraction of cold items whose nearest WARM
    neighbour has cosine similarity above each threshold theta;
  * user-owned near-duplication -- fraction of held-out cold positives that are
    near-duplicates (cos > theta) of an item already in the user's history,
    i.e. the recommendations a novelty post-filter would suppress;
  * the mean nearest-neighbour similarity of the cold pool.

Usage (run where the Amazon data + semantic cache live, e.g. the GPU box):
  python scripts/neardup_analysis.py --config configs/default.yaml \
      --data.source amazon --data.category Video_Games --semantic.method sbert \
      --seed 2026 --out runs/neardup_Video_Games_s2026.json
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csarec.utils import load_config
from csarec.data import load_data
from csarec.semantic import get_semantic

THRESHOLDS = [0.80, 0.90, 0.95]


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def main():
    cfg = load_config()
    out_path = cfg.dget("out", "neardup.json")
    data = load_data(cfg)
    sem = get_semantic(cfg, data, device="cpu")
    sem = _norm(np.asarray(sem, dtype=np.float32))

    warm = np.where(data.warm_mask)[0]
    cold = np.where(data.cold_mask)[0]
    W = sem[warm]  # (nw, d)

    # ---- catalog near-duplication: cold -> nearest warm similarity ----
    max_sim = np.empty(len(cold), dtype=np.float32)
    chunk = 2048
    for s in range(0, len(cold), chunk):
        block = sem[cold[s:s + chunk]] @ W.T          # (c, nw)
        max_sim[s:s + chunk] = block.max(axis=1)
    catalog = {f"theta>{t}": float((max_sim > t).mean()) for t in THRESHOLDS}
    mean_nn = float(max_sim.mean())

    # ---- user-owned near-duplication among cold test positives ----
    owned_hits = {t: 0 for t in THRESHOLDS}
    n_cold_pos = 0
    for u, items in data.test_cold.items():
        hist = list(data.user_pos_train.get(u, ()))
        if not hist:
            n_cold_pos += len(items)
            continue
        H = sem[np.asarray(hist, dtype=np.int64)]      # (h, d)
        for ci in items:
            sims = sem[ci] @ H.T
            mx = float(sims.max()) if sims.size else 0.0
            for t in THRESHOLDS:
                if mx > t:
                    owned_hits[t] += 1
            n_cold_pos += 1
    owned = {f"theta>{t}": (owned_hits[t] / max(n_cold_pos, 1)) for t in THRESHOLDS}

    result = {
        "dataset": cfg.dget("data.category", "na"),
        "seed": cfg.dget("seed"),
        "num_cold_items": int(len(cold)),
        "num_cold_test_positives": int(n_cold_pos),
        "cold_pool_mean_nn_similarity": mean_nn,
        "catalog_near_dup_fraction": catalog,
        "user_owned_near_dup_fraction": owned,
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
