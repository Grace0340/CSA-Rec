"""Full-ranking evaluation: Recall@K and NDCG@K over warm / cold / all splits."""
import numpy as np
import torch


def _dcg_at_k(hits):
    # hits: (b, k) binary in rank order
    k = hits.shape[1]
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=hits.device).float())
    return (hits * disc).sum(dim=1)


@torch.no_grad()
def evaluate(model, data, split_dict, ks, device, candidates="all", chunk=1024):
    """candidates: which item pool to rank against.
       "warm" -> rank among warm items (in-matrix eval)
       "cold" -> rank among cold items (cold-start item eval)
       "all"  -> rank against the full catalog
    """
    if not split_dict:
        return {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}
    user_all, item_final = model.get_all_eval()
    maxk = min(max(ks), item_final.shape[0])
    users = list(split_dict.keys())
    cold_items = torch.tensor(np.where(data.cold_mask)[0], device=device)
    warm_items = torch.tensor(np.where(data.warm_mask)[0], device=device)

    agg = {f"recall@{k}": 0.0 for k in ks}
    agg.update({f"ndcg@{k}": 0.0 for k in ks})
    n_eval = 0

    for s in range(0, len(users), chunk):
        batch_users = users[s:s + chunk]
        U = torch.tensor(batch_users, device=device)
        scores = user_all[U] @ item_final.t()             # (b, I)

        # mask training positives
        for bi, u in enumerate(batch_users):
            pos = data.user_pos_train.get(u)
            if pos:
                scores[bi, list(pos)] = float("-inf")
        if candidates == "warm" and cold_items.numel() > 0:
            scores[:, cold_items] = float("-inf")
        elif candidates == "cold" and warm_items.numel() > 0:
            scores[:, warm_items] = float("-inf")

        topk = torch.topk(scores, maxk, dim=1).indices        # (b, maxk)

        for bi, u in enumerate(batch_users):
            gt = set(split_dict[u])
            if not gt:
                continue
            rec = topk[bi].tolist()
            hit_flags = torch.tensor([1.0 if it in gt else 0.0 for it in rec],
                                     device=device).unsqueeze(0)
            for k in ks:
                hk = hit_flags[:, :k]
                n_hit = hk.sum().item()
                agg[f"recall@{k}"] += n_hit / min(len(gt), k)
                dcg = _dcg_at_k(hk).item()
                ideal = _dcg_at_k(torch.ones(1, min(len(gt), k), device=device)).item()
                agg[f"ndcg@{k}"] += dcg / ideal if ideal > 0 else 0.0
            n_eval += 1

    for key in agg:
        agg[key] = agg[key] / max(n_eval, 1)
    return agg


@torch.no_grad()
def evaluate_beyond(model, data, split_dict, ks, device, semantic,
                    candidates="cold", chunk=1024, max_users=None, seed=0):
    """Beyond-accuracy metrics on the recommended lists:

      coverage@K  -- fraction of the candidate item pool that appears in at
                     least one user's top-K (catalog coverage).
      novelty@K   -- mean self-information -log2(p_i) of recommended items,
                     where p_i is the train popularity share; higher = more
                     long-tail / less obvious recommendations.
      diversity@K -- mean intra-list dissimilarity 1 - avg pairwise cosine of
                     recommended items' semantic vectors.
    """
    empty = {f"coverage@{k}": 0.0 for k in ks}
    empty.update({f"novelty@{k}": 0.0 for k in ks})
    empty.update({f"diversity@{k}": 0.0 for k in ks})
    if not split_dict:
        return empty

    user_all, item_final = model.get_all_eval()
    maxk = min(max(ks), item_final.shape[0])
    users = list(split_dict.keys())
    if max_users is not None and len(users) > max_users:
        # deterministic subsample keeps coverage/novelty/diversity cheap on the
        # largest catalogs without changing the reported numbers materially.
        rng = np.random.default_rng(seed)
        users = [users[i] for i in rng.choice(len(users), size=max_users, replace=False)]
    cold_items = torch.tensor(np.where(data.cold_mask)[0], device=device)
    warm_items = torch.tensor(np.where(data.warm_mask)[0], device=device)

    # popularity (train interaction share) for novelty
    pop = np.zeros(data.num_items, dtype=np.float64)
    for _u, i in data.train_pairs:
        pop[i] += 1.0
    total = max(pop.sum(), 1.0)
    p = np.clip(pop / total, 1e-12, 1.0)
    self_info = torch.tensor(-np.log2(p), dtype=torch.float32, device=device)

    sem_n = torch.nn.functional.normalize(
        torch.tensor(semantic, dtype=torch.float32, device=device), dim=1)

    pool_size = (int(cold_items.numel()) if candidates == "cold"
                 else int(warm_items.numel()) if candidates == "warm"
                 else data.num_items)
    seen = {k: set() for k in ks}
    nov = {k: 0.0 for k in ks}
    div = {k: 0.0 for k in ks}
    n_eval = 0

    for s in range(0, len(users), chunk):
        batch_users = users[s:s + chunk]
        U = torch.tensor(batch_users, device=device)
        scores = user_all[U] @ item_final.t()
        for bi, u in enumerate(batch_users):
            pos = data.user_pos_train.get(u)
            if pos:
                scores[bi, list(pos)] = float("-inf")
        if candidates == "warm" and cold_items.numel() > 0:
            scores[:, cold_items] = float("-inf")
        elif candidates == "cold" and warm_items.numel() > 0:
            scores[:, warm_items] = float("-inf")
        topk = torch.topk(scores, maxk, dim=1).indices  # (b, maxk)

        for bi, u in enumerate(batch_users):
            if not split_dict[u]:
                continue
            rec = topk[bi]
            for k in ks:
                rk = rec[:k]
                seen[k].update(rk.tolist())
                nov[k] += float(self_info[rk].mean())
                v = sem_n[rk]
                sim = v @ v.t()
                off = (sim.sum() - torch.diagonal(sim).sum()) / max(k * (k - 1), 1)
                div[k] += float(1.0 - off)
            n_eval += 1

    out = {}
    for k in ks:
        out[f"coverage@{k}"] = len(seen[k]) / max(pool_size, 1)
        out[f"novelty@{k}"] = nov[k] / max(n_eval, 1)
        out[f"diversity@{k}"] = div[k] / max(n_eval, 1)
    return out
