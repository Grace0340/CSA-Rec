"""Learned cold-start baselines under a *unified* backbone and split.

The main paper compares CSA-Rec against training-free references (Content-kNN,
Popularity, Random) and the pure collaborative backbone. Reviewers rightly ask
for *learned* cold-start methods. Re-running the original authors' code under a
zero-interaction protocol they were not designed for is error-prone and unfair,
so we instead re-implement the representative mechanisms inside our own pipeline:
every baseline shares the identical LightGCN backbone, the identical cold/warm
split, the identical frozen encoder, and the identical full-ranking evaluator,
so any difference isolates the content->collaborative mechanism itself.

Implemented mechanisms (selected via ``--method``):
  * ``content2emb`` -- regress warm items' trained ID embeddings from content
        (MSE). The classic "content tower" cold-start baseline (DeepMusic /
        MetaEmbedding family). No contrastive term, no neighbour aggregation.
  * ``dropoutnet``  -- DropoutNet [Volkovs et al., NeurIPS'17]: an item tower
        that sees [collaborative, content] with the collaborative input randomly
        dropped, trained to reconstruct the ID embedding; at cold time the
        collaborative input is absent and content carries the prediction.
  * ``clcrec``      -- CLCRec [Wei et al., MM'21]: a contrastive objective that
        maximises agreement between an item's content embedding and its
        collaborative embedding; the content embedding is used directly for cold
        items. This is the closest learned baseline to CSA-Rec and isolates the
        incremental value of our CPG neighbour aggregation and adaptive fusion,
        which CLCRec does not have.

All three produce a cold item representation from content, scale-match it to the
warm ID-embedding norm exactly as CSA-Rec does, and rank with the collaborative
user embedding. None of them use the adaptive fusion gate (a CSA-Rec component).
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import CSARec, build_norm_adj
from .metrics import evaluate


LEARNED_BASELINES = ("content2emb", "dropoutnet", "clcrec")


class _BaselineScorer:
    """Adapts a precomputed (user, item) representation to the evaluator API."""

    def __init__(self, user_repr, item_repr, n_head_params):
        self._u = user_repr
        self._i = item_repr
        self._n = int(n_head_params)
        self.use_fusion = False

    @torch.no_grad()
    def get_all_eval(self):
        return self._u, self._i

    def trainable_params(self):
        return self._n

    def adapter_params(self):
        return self._n

    def eval(self):
        return self

    def train(self, mode=True):
        return self


# --------------------------------------------------------------------------- #
# Stage 1: train the shared collaborative backbone (pure BPR, no semantics).
# --------------------------------------------------------------------------- #
def _backbone_cache_path(cfg):
    root = cfg.dget("data.root", "./data")
    src = cfg.dget("data.source", "synthetic")
    cat = cfg.dget("data.category", "na")
    bb = cfg.dget("model.backbone", "lightgcn")
    cr = cfg.dget("data.cold_ratio", 0.1)
    split = cfg.dget("data.cold_split", "random")
    seed = cfg.dget("seed", 0)
    d = cfg.dget("model.emb_dim", 64)
    L = cfg.dget("model.n_layers", 2)
    return os.path.join(root, f"bbemb_{src}_{cat}_{bb}_{split}_cr{cr}_d{d}_L{L}_seed{seed}.npz")


def _train_backbone(cfg, data, semantic, device):
    from .train import sample_negatives  # lazy: avoids a circular import

    # The learned baselines all share one frozen CF backbone; cache it so the
    # three heads for a given (category, seed) train it only once.
    cache = _backbone_cache_path(cfg)
    if cfg.dget("train.cache_backbone", True) and os.path.exists(cache):
        z = np.load(cache)
        if z["item"].shape[0] == data.num_items and z["user"].shape[0] == data.num_users:
            print(f"[coldbaseline] reuse cached backbone {os.path.basename(cache)}")
            return (torch.tensor(z["user"], device=device),
                    torch.tensor(z["item"], device=device))

    # The Cfg dict subclass is not deepcopy-friendly, so toggle use_semantic on
    # the live config to build a pure CF backbone, then restore it afterwards.
    _orig_use_sem = cfg.dget("model.use_semantic", True)
    cfg.dset("model.use_semantic", False)  # pure LightGCN/MF, no adapters
    norm_adj = build_norm_adj(data.train_pairs, data.num_users, data.num_items, device) \
        if cfg.dget("model.backbone") == "lightgcn" else None
    bb = CSARec(cfg, data, semantic, norm_adj, device).to(device)
    cfg.dset("model.use_semantic", _orig_use_sem)  # restore for later stages/save
    opt = torch.optim.Adam(bb.parameters(), lr=cfg.dget("train.lr", 1e-2),
                           weight_decay=cfg.dget("train.weight_decay", 0.0))

    pairs = data.train_pairs
    warm_items = np.where(data.warm_mask)[0]
    rng = np.random.default_rng(cfg.seed)
    bs = cfg.dget("train.batch_size", 2048)
    ks = cfg.dget("train.topk", [10, 20])
    best_valid, best_state, bad = -1.0, None, 0
    for epoch in range(1, cfg.dget("train.epochs", 50) + 1):
        bb.train()
        perm = rng.permutation(len(pairs))
        for s in range(0, len(pairs), bs):
            batch = pairs[perm[s:s + bs]]
            u = torch.tensor(batch[:, 0], device=device)
            pos = torch.tensor(batch[:, 1], device=device)
            neg = torch.tensor(sample_negatives(batch[:, 0], warm_items,
                                                data.user_pos_train, rng), device=device)
            user_all, item_all = bb.propagate()
            loss = bb.bpr_loss(user_all, item_all, u, pos, neg)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % cfg.dget("train.eval_every", 5) == 0 or epoch == 1:
            bb.eval()
            valid = evaluate(bb, data, data.valid, ks, device, candidates="warm")
            key = f"ndcg@{max(ks)}"
            print(f"[bb ep {epoch}] valid_{key}={valid[key]:.4f}")
            if valid[key] > best_valid:
                best_valid, best_state, bad = valid[key], copy.deepcopy(bb.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.dget("train.patience", 5):
                    print(f"[bb early stop] ep {epoch}")
                    break
    if best_state is not None:
        bb.load_state_dict(best_state)
    bb.eval()
    with torch.no_grad():
        user_all, item_all = bb.propagate()
    user_all, item_all = user_all.detach(), item_all.detach()
    if cfg.dget("train.cache_backbone", True):
        os.makedirs(cfg.dget("data.root", "./data"), exist_ok=True)
        np.savez(cache, user=user_all.cpu().numpy(), item=item_all.cpu().numpy())
    return user_all, item_all


# --------------------------------------------------------------------------- #
# Stage 2: content heads.
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, din, d, hidden=None):
        super().__init__()
        hidden = hidden or d
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class _DropoutTower(nn.Module):
    """Item tower that consumes [collaborative, content]; the collaborative half
    is randomly dropped during training and absent (zero) for cold items."""

    def __init__(self, sem_dim, d, hidden=None):
        super().__init__()
        hidden = hidden or d
        self.sem_proj = nn.Linear(sem_dim, d)
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, e_cf, s, drop_cf=False):
        sp = self.sem_proj(s)
        if drop_cf:
            e_cf = torch.zeros_like(e_cf)
        return self.net(torch.cat([e_cf, sp], dim=-1))


def _fit_content2emb(head, sem, E_i, warm_idx, device, epochs, lr, bs, rng):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    tgt = E_i[warm_idx].detach()
    sw = sem[warm_idx]
    for ep in range(epochs):
        perm = rng.permutation(len(warm_idx))
        for s in range(0, len(warm_idx), bs):
            b = perm[s:s + bs]
            pred = head(sw[b])
            loss = F.mse_loss(pred, tgt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def _fit_clcrec(head, sem, E_i, warm_idx, device, epochs, lr, bs, rng, tau=0.2):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    e_id = F.normalize(E_i[warm_idx].detach(), dim=-1)
    sw = sem[warm_idx]
    for ep in range(epochs):
        perm = rng.permutation(len(warm_idx))
        for s in range(0, len(warm_idx), bs):
            b = perm[s:s + bs]
            hb = F.normalize(head(sw[b]), dim=-1)
            eb = e_id[b]
            logits = hb @ eb.t() / tau
            labels = torch.arange(logits.size(0), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def _fit_dropoutnet(tower, sem, E_i, warm_idx, device, epochs, lr, bs, rng, p_drop=0.5):
    opt = torch.optim.Adam(tower.parameters(), lr=lr)
    tgt = E_i[warm_idx].detach()
    ecf = E_i[warm_idx].detach()
    sw = sem[warm_idx]
    for ep in range(epochs):
        perm = rng.permutation(len(warm_idx))
        for s in range(0, len(warm_idx), bs):
            b = perm[s:s + bs]
            # per-sample collaborative dropout
            keep = (torch.rand(len(b), 1, device=device) > p_drop).float()
            pred = tower.net(torch.cat([ecf[b] * keep, tower.sem_proj(sw[b])], dim=-1))
            loss = F.mse_loss(pred, tgt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return tower


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_cold_baseline(cfg, data, semantic, device, kind):
    """Train the backbone + a content head; return a scorer for the evaluator."""
    d = int(cfg.dget("model.emb_dim", 64))
    epochs = int(cfg.dget("train.baseline_head_epochs", 30))
    lr = float(cfg.dget("train.baseline_head_lr", 1e-3))
    bs = int(cfg.dget("train.batch_size", 2048))
    rng = np.random.default_rng(cfg.seed)

    print(f"[coldbaseline] {kind}: stage-1 backbone")
    E_u, E_i = _train_backbone(cfg, data, semantic, device)

    sem = F.normalize(torch.tensor(semantic, dtype=torch.float32, device=device), dim=1)
    sem_dim = sem.shape[1]
    warm_idx = torch.tensor(np.where(data.warm_mask)[0], dtype=torch.long, device=device)

    print(f"[coldbaseline] {kind}: stage-2 content head ({epochs} epochs)")
    if kind == "content2emb":
        head = _MLP(sem_dim, d).to(device)
        _fit_content2emb(head, sem, E_i, warm_idx, device, epochs, lr, bs, rng)
        cold_repr = head(sem).detach()
        n_params = sum(p.numel() for p in head.parameters())
    elif kind == "clcrec":
        head = _MLP(sem_dim, d).to(device)
        _fit_clcrec(head, sem, E_i, warm_idx, device, epochs, lr, bs, rng,
                    tau=float(cfg.dget("model.csia_tau", 0.2)))
        cold_repr = head(sem).detach()
        n_params = sum(p.numel() for p in head.parameters())
    elif kind == "dropoutnet":
        tower = _DropoutTower(sem_dim, d).to(device)
        _fit_dropoutnet(tower, sem, E_i, warm_idx, device, epochs, lr, bs, rng)
        zeros = torch.zeros(data.num_items, d, device=device)
        cold_repr = tower.net(torch.cat([zeros, tower.sem_proj(sem)], dim=-1)).detach()
        n_params = sum(p.numel() for p in tower.parameters())
    else:
        raise ValueError(f"unknown learned baseline: {kind}")

    # scale-match cold pseudo-embeddings to the mean warm norm (same as CSA-Rec)
    warm_mask_t = torch.tensor(data.warm_mask, dtype=torch.bool, device=device)
    cold_mask_t = torch.tensor(data.cold_mask, dtype=torch.bool, device=device)
    warm_norm = E_i[warm_mask_t].norm(dim=1).mean().clamp(min=1e-6)
    cold_repr = F.normalize(cold_repr, dim=1) * warm_norm
    final = torch.where(cold_mask_t.unsqueeze(1), cold_repr, E_i)
    return _BaselineScorer(E_u, final, n_params)
