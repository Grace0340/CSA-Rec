"""CSA-Rec training / evaluation entry point.

Usage:
  python -m csarec.train --config configs/default.yaml
  python -m csarec.train --config configs/default.yaml --data.source amazon --data.category Beauty
  python -m csarec.train --config configs/default.yaml --model.use_csia false   # ablation
"""
import os
import json
import time
import copy
import numpy as np
import torch

from .utils import load_config, set_seed, pick_device, ensure_dir
from .data import load_data
from .semantic import get_semantic
from .models import CSARec, build_norm_adj
from .metrics import evaluate, evaluate_beyond


def merge_all(test_warm, test_cold):
    out = {}
    for u, items in test_warm.items():
        out.setdefault(u, []).extend(items)
    for u, items in test_cold.items():
        out.setdefault(u, []).extend(items)
    return out


def sample_negatives(users, warm_items, user_pos_train, rng):
    neg = rng.choice(warm_items, size=len(users))
    for k, u in enumerate(users):
        pos = user_pos_train.get(int(u), ())
        tries = 0
        while neg[k] in pos and tries < 10:
            neg[k] = rng.choice(warm_items)
            tries += 1
    return neg


def run():
    cfg = load_config()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    print(f"[cfg] device={device} source={cfg.dget('data.source')} "
          f"backbone={cfg.dget('model.backbone')} "
          f"use_semantic={cfg.dget('model.use_semantic')} "
          f"use_csia={cfg.dget('model.use_csia')} use_cpg={cfg.dget('model.use_cpg')}")

    data = load_data(cfg)
    print(f"[data] users={data.num_users} items={data.num_items} "
          f"train={len(data.train_pairs)} cold_items={int(data.cold_mask.sum())} "
          f"test_warm_users={len(data.test_warm)} test_cold_users={len(data.test_cold)}")

    semantic = get_semantic(cfg, data, device)
    print(f"[semantic] method={cfg.dget('semantic.method')} shape={tuple(semantic.shape)}")

    ks = cfg.dget("train.topk", [10, 20])
    baseline = cfg.dget("baseline", "none")
    if baseline and baseline != "none":
        from .baselines import build_baseline
        print(f"[baseline] {baseline} (training-free)")
        model = build_baseline(baseline, data, semantic, device, seed=cfg.seed)
        evaluate_and_save(cfg, data, model, device, ks, semantic)
        return

    method = cfg.dget("method", "none")
    if method and method != "none":
        from .coldbaselines import run_cold_baseline, LEARNED_BASELINES
        if method not in LEARNED_BASELINES:
            raise ValueError(f"unknown method: {method} (expected one of {LEARNED_BASELINES})")
        print(f"[method] learned cold-start baseline: {method}")
        model = run_cold_baseline(cfg, data, semantic, device, method)
        evaluate_and_save(cfg, data, model, device, ks, semantic)
        return

    norm_adj = build_norm_adj(data.train_pairs, data.num_users, data.num_items, device) \
        if cfg.dget("model.backbone") == "lightgcn" else None
    model = CSARec(cfg, data, semantic, norm_adj, device).to(device)
    print(f"[model] trainable_params={model.trainable_params():,} "
          f"adapter_params={model.adapter_params():,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.dget("train.lr", 1e-3),
                           weight_decay=cfg.dget("train.weight_decay", 1e-4))

    pairs = data.train_pairs
    warm_items = np.where(data.warm_mask)[0]
    rng = np.random.default_rng(cfg.seed)
    ks = cfg.dget("train.topk", [10, 20])
    lam = cfg.dget("model.csia_lambda", 0.5)
    bs = cfg.dget("train.batch_size", 2048)
    test_all = merge_all(data.test_warm, data.test_cold)

    best_valid, best_state, bad = -1.0, None, 0
    for epoch in range(1, cfg.dget("train.epochs", 50) + 1):
        model.train()
        perm = rng.permutation(len(pairs))
        tot, tot_bpr, tot_csia = 0.0, 0.0, 0.0
        for s in range(0, len(pairs), bs):
            batch = pairs[perm[s:s + bs]]
            u = torch.tensor(batch[:, 0], device=device)
            pos = torch.tensor(batch[:, 1], device=device)
            neg_np = sample_negatives(batch[:, 0], warm_items, data.user_pos_train, rng)
            neg = torch.tensor(neg_np, device=device)

            user_all, item_all = model.propagate()
            bpr = model.bpr_loss(user_all, item_all, u, pos, neg)
            anchor = torch.unique(pos)
            csia = model.csia_loss(item_all, anchor)
            loss = bpr + lam * csia

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); tot_bpr += bpr.item(); tot_csia += float(csia)

        if epoch % cfg.dget("train.eval_every", 5) == 0 or epoch == 1:
            model.eval()
            valid = evaluate(model, data, data.valid, ks, device, candidates="warm")
            key = f"ndcg@{max(ks)}"
            msg = f"[ep {epoch}] loss={tot:.3f} bpr={tot_bpr:.3f} csia={tot_csia:.3f} " \
                  f"valid_{key}={valid[key]:.4f}"
            print(msg)
            if valid[key] > best_valid:
                best_valid = valid[key]
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= cfg.dget("train.patience", 5):
                    print(f"[early stop] no improvement for {bad} evals")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    evaluate_and_save(cfg, data, model, device, ks, semantic)


def evaluate_and_save(cfg, data, model, device, ks, semantic=None):
    test_all = merge_all(data.test_warm, data.test_cold)
    results = {
        "warm": evaluate(model, data, data.test_warm, ks, device, candidates="warm"),
        "cold": evaluate(model, data, data.test_cold, ks, device, candidates="cold"),
        "all": evaluate(model, data, test_all, ks, device, candidates="all"),
    }
    # --- efficiency ---
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        _ = model.get_all_eval()
    if device == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 5 * 1000
    efficiency = {
        "trainable_params": model.trainable_params(),
        "adapter_params": model.adapter_params(),
        "adapter_ratio": model.adapter_params() / max(model.trainable_params(), 1),
        "full_inference_ms": round(latency_ms, 2),
        "fusion_beta": float(model.beta.detach()) if getattr(model, "use_fusion", False)
                       and hasattr(model, "beta_raw") else None,
    }

    # --- beyond-accuracy metrics (coverage / novelty / diversity) ---
    beyond = {}
    if semantic is not None and cfg.dget("eval.beyond", True):
        cap = cfg.dget("eval.beyond_max_users", 20000)
        try:
            beyond = {
                "cold": evaluate_beyond(model, data, data.test_cold, ks, device,
                                        semantic, candidates="cold",
                                        max_users=cap, seed=cfg.seed),
            }
        except Exception as e:
            print(f"[warn] beyond-accuracy metrics skipped: {e}")

    out_dir = ensure_dir(os.path.join(cfg.dget("output.dir", "./runs"),
                                      cfg.dget("output.tag", "run")))
    payload = {"config": cfg, "metrics": results, "efficiency": efficiency,
               "beyond": beyond}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n==== FINAL ====")
    for split in ("warm", "cold", "all"):
        row = " ".join(f"{k}={v:.4f}" for k, v in results[split].items())
        print(f"{split:>4}: {row}")
    print(f"efficiency: {efficiency}")
    print(f"[saved] {os.path.join(out_dir, 'metrics.json')}")

    _write_rationale_demo(model, data, out_dir, device)


@torch.no_grad()
def _write_rationale_demo(model, data, out_dir, device, n=5):
    """Template rationales for a few cold items -> nearest warm neighbours in the
    aligned space. Demonstrates the interpretability module without an LLM call."""
    user_all, item_final = model.get_all_eval()
    cold_ids = np.where(data.cold_mask)[0][:n]
    warm_ids = np.where(data.warm_mask)[0]
    lines = []
    for c in cold_ids:
        cvec = item_final[c:c + 1]
        sims = torch.nn.functional.cosine_similarity(cvec, item_final[warm_ids])
        top = warm_ids[torch.topk(sims, min(3, len(warm_ids))).indices.cpu().numpy()]
        def _txt(idx):
            if data.item_text is not None:
                return (data.item_text[idx] or f"item#{idx}")[:60]
            return f"item#{idx}"
        neigh = "; ".join(_txt(int(t)) for t in top)
        lines.append(f"Cold item [{_txt(int(c))}] is recommended because it is "
                     f"semantically close to items you engaged with: {neigh}.")
    with open(os.path.join(out_dir, "rationales.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(out_dir, 'rationales.txt')}")


if __name__ == "__main__":
    run()
