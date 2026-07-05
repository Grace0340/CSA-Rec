"""CSA-Rec model: a collaborative backbone (LightGCN/MF) plus the three
trainable modules that constitute the contribution:

  * SAA  -- Semantic Alignment Adapter (low-rank projection of frozen semantics)
  * CSIA -- Contrastive Semantic-ID Alignment loss (aligns SAA output with ID emb)
  * CPG  -- Cold-start Pseudo-Embedding Generator (content -> collaborative space)

Ablation switches (model.*): use_semantic, saa_type, use_csia, use_cpg.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(train_pairs, n_users, n_items, device):
    """Symmetric normalized adjacency of the user-item bipartite graph."""
    u = train_pairs[:, 0]
    i = train_pairs[:, 1] + n_users
    rows = np.concatenate([u, i])
    cols = np.concatenate([i, u])
    n = n_users + n_items
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    dinv = np.power(np.maximum(deg, 1.0), -0.5)
    vals = dinv[rows] * dinv[cols]
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to(device)


def _make_saa(saa_type, sem_dim, d, rank):
    if saa_type == "lowrank":
        return nn.Sequential(nn.Linear(sem_dim, rank), nn.GELU(), nn.Linear(rank, d))
    if saa_type == "linear":
        return nn.Linear(sem_dim, d)
    if saa_type == "mlp":
        return nn.Sequential(nn.Linear(sem_dim, d), nn.GELU(), nn.Linear(d, d))
    raise ValueError(f"unknown saa_type: {saa_type}")


class CSARec(nn.Module):
    def __init__(self, cfg, data, semantic, norm_adj, device):
        super().__init__()
        m = cfg.model
        self.backbone = m.get("backbone", "lightgcn")
        self.n_layers = int(m.get("n_layers", 2))
        self.d = int(m.get("emb_dim", 64))
        self.use_semantic = bool(m.get("use_semantic", True))
        self.use_csia = bool(m.get("use_csia", True))
        self.use_cpg = bool(m.get("use_cpg", True))
        self.use_fusion = bool(m.get("use_fusion", True))
        self.tau = float(m.get("csia_tau", 0.2))
        self.device = device
        self.n_users = data.num_users
        self.n_items = data.num_items
        self.norm_adj = norm_adj

        self.user_emb = nn.Embedding(self.n_users, self.d)
        self.item_emb = nn.Embedding(self.n_items, self.d)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        sem = torch.tensor(semantic, dtype=torch.float32)
        self.register_buffer("sem", sem)
        self.register_buffer("cold_mask", torch.tensor(data.cold_mask, dtype=torch.bool))
        self.register_buffer("warm_mask", torch.tensor(data.warm_mask, dtype=torch.bool))

        self.saa = _make_saa(m.get("saa_type", "lowrank"), sem.shape[1], self.d,
                             int(m.get("saa_rank", 16)))
        # CPG: content-guided neighbour aggregation of warm ID embeddings.
        # For each (cold) item we aggregate the trained ID embeddings of its
        # semantic nearest-neighbour WARM items, then refine with a small linear.
        if self.use_semantic and self.use_cpg:
            self.cpg_proj = nn.Linear(self.d, self.d)
            self._build_neighbors(
                semantic, data.warm_mask, device,
                K=int(m.get("cpg_neighbors", 10)),
                temp=float(m.get("cpg_temp", 0.1)),
                cache_path=self._nbr_cache_path(cfg),
            )

        # Adaptive collaborative-content fusion.  The final score adds a direct
        # semantic-match term  beta * <user profile, item semantics>  to the
        # collaborative score.  beta is learnable (softplus, >=0): on dense data
        # it stays small (collaborative dominates); on sparse data it grows so
        # the model falls back to content matching. This term subsumes a pure
        # content-kNN scorer, so CSA-Rec is content-kNN + collaborative refinement.
        if self.use_semantic and self.use_fusion:
            sem_n = F.normalize(sem, dim=1)
            self.register_buffer("sem_norm", sem_n)
            tp = torch.as_tensor(data.train_pairs, dtype=torch.long)
            prof = torch.zeros(self.n_users, sem_n.shape[1])
            cnt = torch.zeros(self.n_users)
            prof.index_add_(0, tp[:, 0], sem_n[tp[:, 1]])
            cnt.index_add_(0, tp[:, 0], torch.ones(tp.shape[0]))
            prof = prof / cnt.clamp(min=1.0).unsqueeze(1)
            self.register_buffer("user_profile", F.normalize(prof, dim=1))
            self.beta_raw = nn.Parameter(torch.tensor(0.0))

    # --- backbone propagation ------------------------------------------------
    def propagate(self):
        if self.backbone == "mf":
            return self.user_emb.weight, self.item_emb.weight
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[:self.n_users], out[self.n_users:]

    # --- neighbour precompute for CPG ---------------------------------------
    def _nbr_cache_path(self, cfg):
        root = cfg.dget("data.root", "./data")
        src = cfg.dget("data.source", "synthetic")
        cat = cfg.dget("data.category", "na")
        cr = cfg.dget("data.cold_ratio", 0.1)
        seed = cfg.dget("seed", 0)
        K = cfg.dget("model.cpg_neighbors", 10)
        # The neighbour table depends on the ENCODER (which items are similar)
        # and on the COLD-SPLIT policy (which items are warm candidates), so both
        # must be part of the cache key to avoid reusing a stale table.
        method = cfg.dget("semantic.method", "hash")
        mtag = "".join(c if c.isalnum() else "-" for c in
                       str(cfg.dget("semantic.model_name", ""))).strip("-") \
            if method in ("sbert", "clip") else method
        split = cfg.dget("data.cold_split", "random")
        os.makedirs(root, exist_ok=True)
        return os.path.join(
            root, f"{src}_{cat}_nbr_{mtag}_{split}_K{K}_cr{cr}_seed{seed}.npz")

    def _build_neighbors(self, semantic_np, warm_mask_np, device, K, temp, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path)
            self.register_buffer("nbr_idx", torch.tensor(z["idx"], dtype=torch.long))
            self.register_buffer("nbr_w", torch.tensor(z["w"], dtype=torch.float32))
            return
        sem = F.normalize(torch.tensor(semantic_np, dtype=torch.float32, device=device), dim=1)
        warm_idx = torch.tensor(np.where(warm_mask_np)[0], dtype=torch.long, device=device)
        nwarm = int(warm_idx.numel())
        K = max(1, min(K, nwarm - 1))
        wsem = sem[warm_idx]
        warm_pos = torch.full((self.n_items,), -1, dtype=torch.long, device=device)
        warm_pos[warm_idx] = torch.arange(nwarm, device=device)
        nbr_idx = torch.empty(self.n_items, K, dtype=torch.long, device=device)
        nbr_w = torch.empty(self.n_items, K, dtype=torch.float32, device=device)
        chunk = 512
        for s in range(0, self.n_items, chunk):
            e = min(s + chunk, self.n_items)
            sims = sem[s:e] @ wsem.t()                     # (c, nwarm)
            wp = warm_pos[torch.arange(s, e, device=device)]
            valid = wp >= 0
            if valid.any():
                ridx = torch.nonzero(valid, as_tuple=True)[0]
                sims[ridx, wp[valid]] = float("-inf")       # exclude self
            vals, idx = sims.topk(K, dim=1)
            nbr_idx[s:e] = warm_idx[idx]
            nbr_w[s:e] = torch.softmax(vals / temp, dim=1)
        self.register_buffer("nbr_idx", nbr_idx)
        self.register_buffer("nbr_w", nbr_w)
        if cache_path:
            np.savez(cache_path, idx=nbr_idx.cpu().numpy(), w=nbr_w.cpu().numpy())

    # --- content -> collaborative encoder (the cold-item pathway) ------------
    # The SAME function is trained by CSIA and used at eval, otherwise the eval
    # module would corrupt the aligned representation. Neighbour ID embeddings
    # are detached so CSIA trains the adapters, not the collaborative backbone.
    def content_embed(self, item_all, idx=None):
        s = self.sem if idx is None else self.sem[idx]
        h = self.saa(s)
        if self.use_cpg and hasattr(self, "nbr_idx"):
            nb = self.nbr_idx if idx is None else self.nbr_idx[idx]   # (m, K)
            nw = self.nbr_w if idx is None else self.nbr_w[idx]       # (m, K)
            neigh = item_all[nb].detach()                            # (m, K, d)
            p = (nw.unsqueeze(-1) * neigh).sum(dim=1)                # (m, d)
            h = h + self.cpg_proj(p)
        return h

    # --- item representations used at eval (with cold override) --------------
    def item_eval_repr(self, item_all):
        if not self.use_semantic:
            return item_all
        cold_repr = self.content_embed(item_all)           # (I, d) via SAA(+CPG)
        # Scale-match cold pseudo-embeddings to the warm ID-embedding norm, so that
        # warm and cold items compete on the same scale during full ranking.
        warm_norm = item_all[self.warm_mask].norm(dim=1).mean().clamp(min=1e-6)
        cold_repr = F.normalize(cold_repr, dim=1) * warm_norm
        final = torch.where(self.cold_mask.unsqueeze(1), cold_repr, item_all)
        return final

    @property
    def beta(self):
        return F.softplus(self.beta_raw)

    # --- losses --------------------------------------------------------------
    def bpr_loss(self, user_all, item_all, u, pos, neg):
        ue, pe, ne = user_all[u], item_all[pos], item_all[neg]
        x_pos = (ue * pe).sum(-1)
        x_neg = (ue * ne).sum(-1)
        if self.use_semantic and self.use_fusion:
            pu = self.user_profile[u]
            b = self.beta
            x_pos = x_pos + b * (pu * self.sem_norm[pos]).sum(-1)
            x_neg = x_neg + b * (pu * self.sem_norm[neg]).sum(-1)
        return -F.logsigmoid(x_pos - x_neg).mean()

    def csia_loss(self, item_all, warm_items):
        if not (self.use_semantic and self.use_csia):
            return torch.zeros((), device=self.device)
        # Stop-grad on the collaborative target: CSIA aligns the semantic adapter
        # INTO the ID space, it must not drag the ID embeddings toward semantics
        # (that corrupts the collaborative signal and stalls BPR).
        e_id = F.normalize(item_all[warm_items].detach(), dim=-1)
        e_sem = F.normalize(self.content_embed(item_all, warm_items), dim=-1)
        logits = e_sem @ e_id.t() / self.tau
        labels = torch.arange(logits.size(0), device=self.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    @torch.no_grad()
    def get_all_eval(self):
        user_all, item_all = self.propagate()
        item_repr = self.item_eval_repr(item_all)
        if self.use_semantic and self.use_fusion:
            # Concatenate so a single dot product reproduces the fused score:
            #   <user_all, item_repr> + beta * <user_profile, sem_norm>.
            b = self.beta
            user_all = torch.cat([user_all, b * self.user_profile], dim=1)
            item_repr = torch.cat([item_repr, self.sem_norm], dim=1)
        return user_all, item_repr

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def adapter_params(self):
        n = sum(p.numel() for p in self.saa.parameters())
        if hasattr(self, "cpg_proj"):
            n += sum(p.numel() for p in self.cpg_proj.parameters())
        return n
