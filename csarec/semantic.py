"""Frozen semantic embeddings for items.

The backbone is FROZEN by design: we compute item embeddings once, cache them,
and never back-propagate into the encoder. This is the deployment-cost claim of
CSA-Rec -- only the light adapter (SAA) and CPG are trained downstream.

Methods:
  * hash  -- deterministic pseudo-embedding from text (no deps; for smoke tests)
  * sbert -- sentence-transformers text encoder (default for real data)
  * clip  -- CLIP text (+ optional image) encoder via transformers
"""
import os
import hashlib
import numpy as np


def _hash_embed(texts, dim, seed=2026):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for k, t in enumerate(texts):
        h = hashlib.sha256((str(seed) + "::" + (t or f"item_{k}")).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        out[k] = rng.normal(size=dim).astype(np.float32)
    return out


def _sbert_embed(texts, model_name, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        [t if t else " " for t in texts],
        batch_size=256, convert_to_numpy=True,
        show_progress_bar=True, normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def _clip_embed(texts, images, model_name, device, use_image):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = model_name if "clip" in model_name.lower() else "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    feats = []
    bs = 128
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            batch_txt = [t if t else " " for t in texts[s:s + bs]]
            inp = proc(text=batch_txt, return_tensors="pt",
                       padding=True, truncation=True).to(device)
            tfeat = model.get_text_features(**inp)
            tfeat = torch.nn.functional.normalize(tfeat, dim=-1)
            if use_image and images is not None:
                from PIL import Image
                imgs = []
                for path in images[s:s + bs]:
                    try:
                        imgs.append(Image.open(path).convert("RGB"))
                    except Exception:
                        imgs.append(Image.new("RGB", (224, 224)))
                iin = proc(images=imgs, return_tensors="pt").to(device)
                ifeat = model.get_image_features(**iin)
                ifeat = torch.nn.functional.normalize(ifeat, dim=-1)
                feat = torch.nn.functional.normalize(tfeat + ifeat, dim=-1)
            else:
                feat = tfeat
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def _sanitize(name):
    """Filesystem-safe short tag for an encoder model name."""
    return "".join(c if c.isalnum() else "-" for c in str(name)).strip("-")


def get_semantic(cfg, data, device):
    # synthetic already carries an injected semantic signal
    if data.semantic is not None:
        return data.semantic

    method = cfg.dget("semantic.method", "hash")
    root = cfg.dget("data.root", "./data")
    src = cfg.dget("data.source", "synthetic")
    cat = cfg.dget("data.category", "na")
    # Cache key must include the CATEGORY and the ENCODER model, otherwise
    # different categories (or a stronger encoder swapped in for the scaling
    # study) silently collide on the same .npy file.
    mtag = _sanitize(cfg.dget("semantic.model_name", "")) if method in ("sbert", "clip") else ""
    parts = [p for p in (src, cat, method, mtag) if p]
    cache_path = os.path.join(root, "_".join(parts) + "_sem.npy")
    if cfg.dget("semantic.cache", True) and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == data.num_items:
            return arr

    texts = data.item_text if data.item_text is not None else [""] * data.num_items
    if method == "hash":
        emb = _hash_embed(texts, cfg.dget("semantic.dim", 64), cfg.seed)
    elif method == "sbert":
        emb = _sbert_embed(texts, cfg.dget("semantic.model_name",
                                           "sentence-transformers/all-MiniLM-L6-v2"), device)
    elif method == "clip":
        images = getattr(data, "item_image", None)
        emb = _clip_embed(texts, images, cfg.dget("semantic.model_name", "openai/clip-vit-base-patch32"),
                          device, cfg.dget("semantic.use_image", False))
    else:
        raise ValueError(f"unknown semantic.method: {method}")

    if cfg.dget("semantic.cache", True):
        os.makedirs(root, exist_ok=True)
        np.save(cache_path, emb)
    return emb
