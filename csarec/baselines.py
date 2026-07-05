"""Training-free baselines that plug into metrics.evaluate().

Each baseline exposes get_all_eval() -> (user_repr, item_repr) so that the same
full-ranking evaluator (warm / cold / all) scores them identically to CSA-Rec.

  content_knn : user = mean of interacted items' frozen semantics;
                item = its frozen semantics; score = cosine similarity.
                This is the pure content-based cold-start baseline -- the key
                control showing that ALIGNING semantics to the collaborative
                space (CSA-Rec) beats naive semantic matching.
  pop         : popularity (train interaction count). Cold items have count 0.
  random      : random Gaussian embeddings (reference lower bound).
"""
import numpy as np
import torch
import torch.nn.functional as F


class _Scorer:
    def __init__(self, user_repr, item_repr):
        self._u = user_repr
        self._i = item_repr

    @torch.no_grad()
    def get_all_eval(self):
        return self._u, self._i

    def trainable_params(self):
        return 0

    def adapter_params(self):
        return 0

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def build_baseline(kind, data, semantic, device, seed=2026):
    n_users, n_items = data.num_users, data.num_items
    if kind == "content_knn":
        sem = F.normalize(torch.tensor(semantic, dtype=torch.float32, device=device), dim=1)
        prof = torch.zeros(n_users, sem.shape[1], device=device)
        cnt = torch.zeros(n_users, 1, device=device)
        for u, items in data.user_pos_train.items():
            if not items:
                continue
            idx = torch.tensor(list(items), dtype=torch.long, device=device)
            prof[u] = sem[idx].sum(dim=0)
            cnt[u] = len(items)
        prof = prof / cnt.clamp(min=1.0)
        prof = F.normalize(prof, dim=1)
        return _Scorer(prof, sem)

    if kind == "pop":
        pop = np.zeros(n_items, dtype=np.float32)
        for _u, i in data.train_pairs:
            pop[i] += 1.0
        item_repr = torch.tensor(pop, device=device).view(n_items, 1)
        user_repr = torch.ones(n_users, 1, device=device)
        return _Scorer(user_repr, item_repr)

    if kind == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        u = torch.randn(n_users, 32, generator=g).to(device)
        i = torch.randn(n_items, 32, generator=g).to(device)
        return _Scorer(u, i)

    raise ValueError(f"unknown baseline: {kind}")
