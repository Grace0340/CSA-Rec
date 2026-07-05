"""Data loading, k-core filtering, and the item cold-start split.

Cold-start protocol:
  * A fraction `cold_ratio` of items are designated COLD and have ALL their
    interactions removed from training (zero-interaction items).
  * Warm items use leave-one-out: last interaction -> test, 2nd-last -> valid.
  * Every interaction with a cold item becomes a cold test positive.
  * We evaluate warm / cold / all separately.
"""
import os
import gzip
import json
import numpy as np


class RecData:
    def __init__(self):
        self.num_users = 0
        self.num_items = 0
        self.train_pairs = None          # np.ndarray (N, 2): (user, item)
        self.user_pos_train = {}         # user -> set(item)
        self.valid = {}                  # user -> [item] (warm held-out)
        self.test_warm = {}              # user -> [item]
        self.test_cold = {}              # user -> [item]
        self.warm_mask = None            # np.bool_ (num_items,)
        self.cold_mask = None            # np.bool_ (num_items,)
        self.item_text = None            # list[str] or None
        self.item_image = None           # list[str|None] or None
        self.semantic = None             # optional precomputed (num_items, d)


def _kcore_filter(inters, min_u, min_i, passes=10):
    """inters: list of (u, i, t). Iteratively drop low-degree users/items."""
    for _ in range(passes):
        uc, ic = {}, {}
        for u, i, _t in inters:
            uc[u] = uc.get(u, 0) + 1
            ic[i] = ic.get(i, 0) + 1
        keep = [(u, i, t) for (u, i, t) in inters
                if uc[u] >= min_u and ic[i] >= min_i]
        if len(keep) == len(inters):
            break
        inters = keep
    return inters


def build_from_interactions(inters, item_text_raw, cold_ratio, min_u, min_i, seed,
                            cold_split="random"):
    """inters: list of (user_raw, item_raw, timestamp).
    item_text_raw: dict item_raw -> text (or None).

    cold_split:
      "random"   -- draw the cold items uniformly at random (default protocol).
      "temporal" -- designate the most RECENTLY introduced items as cold, i.e.
                    items whose first appearance in the log falls in the latest
                    `cold_ratio` quantile. This mirrors real deployment, where
                    the items lacking history are the newly listed ones rather
                    than a random subset of the catalog.
    """
    rng = np.random.default_rng(seed)
    inters = _kcore_filter(inters, min_u, min_i)
    if not inters:
        raise ValueError("No interactions left after k-core filtering.")

    users = sorted({u for u, _i, _t in inters})
    items = sorted({i for _u, i, _t in inters})
    uid = {u: k for k, u in enumerate(users)}
    iid = {i: k for k, i in enumerate(items)}
    n_users, n_items = len(users), len(items)

    data = RecData()
    data.num_users, data.num_items = n_users, n_items
    data._item_ids = items  # new-index -> original item id (for aligning features)
    data._user_ids = users  # new-index -> original user id (for split export)

    # item text aligned to new index
    if item_text_raw is not None:
        data.item_text = [item_text_raw.get(items[k], "") for k in range(n_items)]

    # designate cold items
    n_cold = max(1, int(round(cold_ratio * n_items)))
    if cold_split == "temporal":
        # first-seen timestamp per (reindexed) item; newest items become cold.
        first_seen = np.full(n_items, -np.inf)
        for u, i, t in inters:
            k = iid[i]
            if t > first_seen[k]:
                first_seen[k] = t
        # break ties deterministically with a seeded jitter, then take the top.
        jitter = rng.random(n_items) * 1e-6
        order = np.argsort(first_seen + jitter)      # ascending: oldest first
        cold_idx = set(order[-n_cold:].tolist())     # newest n_cold items
    elif cold_split == "random":
        cold_idx = set(rng.choice(n_items, size=n_cold, replace=False).tolist())
    else:
        raise ValueError(f"unknown cold_split: {cold_split}")
    warm_mask = np.ones(n_items, dtype=bool)
    for c in cold_idx:
        warm_mask[c] = False
    data.warm_mask = warm_mask
    data.cold_mask = ~warm_mask

    # group per user, sorted by time
    per_user = {}
    for u, i, t in inters:
        per_user.setdefault(uid[u], []).append((t, iid[i]))
    for u in per_user:
        per_user[u].sort()

    train_pairs = []
    for u, seq in per_user.items():
        warm_seq = [i for _t, i in seq if warm_mask[i]]
        cold_seq = [i for _t, i in seq if not warm_mask[i]]
        if cold_seq:
            data.test_cold[u] = list(dict.fromkeys(cold_seq))  # dedup, keep order
        if len(warm_seq) >= 3:
            data.test_warm[u] = [warm_seq[-1]]
            data.valid[u] = [warm_seq[-2]]
            train_items = warm_seq[:-2]
        elif len(warm_seq) == 2:
            data.test_warm[u] = [warm_seq[-1]]
            train_items = warm_seq[:-1]
        else:
            train_items = warm_seq
        for i in train_items:
            train_pairs.append((u, i))
        data.user_pos_train[u] = set(train_items)

    data.train_pairs = np.asarray(train_pairs, dtype=np.int64)
    if data.train_pairs.size == 0:
        raise ValueError("Empty training set; relax cold_ratio / k-core.")
    return data


# --------------------------------------------------------------------------- #
# Synthetic data (dependency-free smoke test; CPU friendly)
# --------------------------------------------------------------------------- #
def make_synthetic(cfg):
    rng = np.random.default_rng(cfg.seed)
    U = cfg.dget("data.synth_users", 800)
    I = cfg.dget("data.synth_items", 500)
    d = cfg.dget("data.synth_dim", 32)
    density = cfg.dget("data.synth_density", 0.03)

    user_f = rng.normal(size=(U, d))
    item_f = rng.normal(size=(I, d))
    logits = user_f @ item_f.T
    probs = 1.0 / (1.0 + np.exp(-logits))
    inters = []
    n_per_user = max(3, int(density * I))
    for u in range(U):
        p = probs[u] / probs[u].sum()
        picks = rng.choice(I, size=n_per_user, replace=False, p=p)
        for rank, i in enumerate(picks):
            inters.append((u, int(i), rank))  # rank as pseudo-timestamp

    data = build_from_interactions(
        inters, None,
        cold_ratio=cfg.dget("data.cold_ratio", 0.1),
        min_u=cfg.dget("data.min_user_inter", 5),
        min_i=cfg.dget("data.min_item_inter", 5),
        seed=cfg.seed,
        cold_split=cfg.dget("data.cold_split", "random"),
    )
    # inject a semantic signal correlated with the (true) item factors, so that
    # cold-start is actually learnable in the smoke test.
    sem_dim = cfg.dget("semantic.dim", 64)
    proj = rng.normal(size=(d, sem_dim))
    sem_full = item_f @ proj + 0.1 * rng.normal(size=(item_f.shape[0], sem_dim))
    # item_f is indexed by ORIGINAL item id; reindex to the (filtered) new order.
    sem = sem_full[np.asarray(data._item_ids, dtype=np.int64)]
    data.semantic = sem.astype(np.float32)
    return data


# --------------------------------------------------------------------------- #
# Amazon Reviews 2023 (McAuley Lab). Download instructions in README.
# Expected local layout:
#   <root>/amazon/<Category>/<Category>.jsonl[.gz]        (reviews)
#   <root>/amazon/<Category>/meta_<Category>.jsonl[.gz]   (item metadata)
# --------------------------------------------------------------------------- #
def _open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _find(root, *names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    return None


def load_amazon(cfg):
    cat = cfg.dget("data.category", "Beauty")
    root = os.path.join(cfg.dget("data.root", "./data"), "amazon", cat)
    rev = _find(root, f"{cat}.jsonl.gz", f"{cat}.jsonl")
    meta = _find(root, f"meta_{cat}.jsonl.gz", f"meta_{cat}.jsonl")
    if rev is None:
        raise FileNotFoundError(
            f"Review file not found under {root}. See README for the download step."
        )

    inters = []
    with _open_maybe_gz(rev) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = r.get("user_id")
            i = r.get("parent_asin") or r.get("asin")
            t = r.get("timestamp", 0)
            rating = r.get("rating", 5.0)
            if u is None or i is None:
                continue
            if rating is not None and float(rating) < 4.0:
                continue  # implicit positive = rating >= 4
            inters.append((u, i, int(t)))

    item_text = {}
    item_image = {}
    if meta is not None:
        with _open_maybe_gz(meta) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = m.get("parent_asin") or m.get("asin")
                if key is None:
                    continue
                parts = [m.get("title", "")]
                feats = m.get("features") or []
                desc = m.get("description") or []
                if isinstance(feats, list):
                    parts += [str(x) for x in feats]
                if isinstance(desc, list):
                    parts += [str(x) for x in desc]
                item_text[key] = " ".join(p for p in parts if p)[:2000]
                imgs = m.get("images") or []
                if isinstance(imgs, list) and imgs:
                    first = imgs[0]
                    if isinstance(first, dict):
                        item_image[key] = first.get("large") or first.get("hi_res") or first.get("thumb")

    data = build_from_interactions(
        inters, item_text if item_text else None,
        cold_ratio=cfg.dget("data.cold_ratio", 0.1),
        min_u=cfg.dget("data.min_user_inter", 5),
        min_i=cfg.dget("data.min_item_inter", 5),
        seed=cfg.seed,
        cold_split=cfg.dget("data.cold_split", "random"),
    )
    if item_image:
        # align image list to reindexed items via item_text order is not kept;
        # images are optional and only used by the CLIP encoder path.
        data.item_image = None  # populated by semantic.py if use_image and available
        data._raw_item_image = item_image  # keep raw for optional encoder use
    return data


def load_data(cfg):
    src = cfg.dget("data.source", "synthetic")
    if src == "synthetic":
        return make_synthetic(cfg)
    if src == "amazon":
        return load_amazon(cfg)
    if src == "mind":
        raise NotImplementedError(
            "MIND loader is left as an extension; see README. Amazon/synthetic are ready."
        )
    raise ValueError(f"unknown data.source: {src}")


# --------------------------------------------------------------------------- #
# Split export -- let external SOTA baselines reuse the EXACT cold/warm split.
# --------------------------------------------------------------------------- #
def export_split(data, out_dir):
    """Dump the reproducible split as TSV + JSON so any baseline repo can align
    to the same user/item indexing, k-core filtering, and cold/warm partition.

    Files written to out_dir:
      item_map.tsv   item_index \t original_item_id \t is_cold
      user_map.tsv   user_index \t original_user_id
      train.tsv      user_index \t item_index        (warm interactions only)
      valid.tsv      user_index \t item_index         (warm leave-one-out)
      test_warm.tsv  user_index \t item_index
      test_cold.tsv  user_index \t item_index         (zero-interaction items)
      split_meta.json  counts + config-independent summary
    """
    os.makedirs(out_dir, exist_ok=True)
    item_ids = getattr(data, "_item_ids", list(range(data.num_items)))
    user_ids = getattr(data, "_user_ids", list(range(data.num_users)))

    with open(os.path.join(out_dir, "item_map.tsv"), "w", encoding="utf-8") as f:
        f.write("item_index\titem_id\tis_cold\n")
        for k in range(data.num_items):
            f.write(f"{k}\t{item_ids[k]}\t{int(data.cold_mask[k])}\n")

    with open(os.path.join(out_dir, "user_map.tsv"), "w", encoding="utf-8") as f:
        f.write("user_index\tuser_id\n")
        for k in range(data.num_users):
            f.write(f"{k}\t{user_ids[k]}\n")

    def _dump_pairs(path, pairs):
        with open(path, "w", encoding="utf-8") as f:
            f.write("user_index\titem_index\n")
            for u, i in pairs:
                f.write(f"{int(u)}\t{int(i)}\n")

    def _dump_dict(path, d):
        with open(path, "w", encoding="utf-8") as f:
            f.write("user_index\titem_index\n")
            for u, items in d.items():
                for i in items:
                    f.write(f"{int(u)}\t{int(i)}\n")

    _dump_pairs(os.path.join(out_dir, "train.tsv"), data.train_pairs)
    _dump_dict(os.path.join(out_dir, "valid.tsv"), data.valid)
    _dump_dict(os.path.join(out_dir, "test_warm.tsv"), data.test_warm)
    _dump_dict(os.path.join(out_dir, "test_cold.tsv"), data.test_cold)

    meta = {
        "num_users": int(data.num_users),
        "num_items": int(data.num_items),
        "num_cold_items": int(data.cold_mask.sum()),
        "num_warm_items": int(data.warm_mask.sum()),
        "num_train_interactions": int(len(data.train_pairs)),
        "num_test_warm_users": len(data.test_warm),
        "num_test_cold_users": len(data.test_cold),
        "num_valid_users": len(data.valid),
    }
    with open(os.path.join(out_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
