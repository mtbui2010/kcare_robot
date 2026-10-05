"""Telling look-alike products apart (신라면 vs 너구리 vs 짜파게티 ...).

The open-vocabulary detector finds "instant noodle" packs well but cannot tell
brands apart, and the VLMs misread the small labels with confidence (it read
辛라면 as 불닭볶음면, "high"). So a product named in ``configs/products/
products.json`` is found in two steps:

  1. detect its generic ``class`` ("instant noodle") — every pack on the shelf;
  2. compare each pack's crop with the product reference photos (SigLIP image
     embeddings via visionserve) and keep the pack that matches this product
     best — only when the match is close enough (``min_sim``) and clearly ahead
     of every other product (``min_margin``).

Otherwise the product counts as not found. The robot is built for blind users,
so it must not hand over a pack it is unsure of, and it does not ask back.
"""
import dataclasses
import json
import os
import re

import cv2
import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'products')
_CATALOG = os.path.join(_DIR, 'products.json')
_cache = {'mtime': None, 'cat': None, 'refs': {}}


def _norm(s: str) -> str:
    return re.sub(r'\s+', '', str(s)).lower()


def _catalog() -> dict:
    try:
        mtime = os.path.getmtime(_CATALOG)
    except OSError:
        return {}
    if _cache['mtime'] != mtime:
        with open(_CATALOG, encoding='utf-8') as f:
            _cache.update(mtime=mtime, cat=json.load(f), refs={})
    return _cache['cat'] or {}


def lookup(name: str):
    """(canonical product name, its entry, match settings) or None."""
    cat = _catalog()
    want = _norm(name)
    for key, entry in (cat.get('products') or {}).items():
        if want == _norm(key) or want in (_norm(a) for a in entry.get('aliases', [])):
            return key, entry, cat.get('match', {})
    return None


def _embed(client, rgb, model) -> np.ndarray:
    v = np.asarray(client.predict(model=model, image=np.ascontiguousarray(rgb)).embeddings[0], dtype=float)
    return v / (np.linalg.norm(v) or 1.0)


def _inner(rgb, frac: float):
    """The middle `frac` of a crop. Packs stand tilted and overlapping, so a full
    box also holds the edge of its neighbour; matching on full boxes called a
    신라면 an 안성탕면 with a "sure" margin, twice. The middle 80% did not."""
    if frac >= 1.0:
        return rgb
    h, w = rgb.shape[:2]
    dx, dy = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    return rgb[dy:h - dy, dx:w - dx]


def _ref_embeddings(client, model, inner: float = 1.0) -> dict:
    """{product: (n_refs, dim)} for every product, cached until the catalogue changes."""
    key = (model, inner)
    if key not in _cache['refs']:
        out = {}
        for name, entry in (_catalog().get('products') or {}).items():
            vs = []
            for f in entry.get('refs', []):
                im = cv2.imread(os.path.join(_DIR, f))
                if im is not None:
                    vs.append(_embed(client, _inner(im[..., ::-1], inner), model))
            if vs:
                out[name] = np.stack(vs)
        _cache['refs'][key] = out
    return _cache['refs'][key]


def _inside(a, b, frac=0.6) -> bool:
    """Box a ([x, y, w, h]) lies mostly inside box b."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy > frac * a[2] * a[3]


def _single_items(boxes) -> list:
    """Indices of boxes around one item: drops a box around two or more others
    (a whole shelf compartment), a box that is part of a bigger item box, and a
    box much wider than the rest (two items merged into one box)."""
    group = [sum(_inside(o, b) for j, o in enumerate(boxes) if j != i) >= 2 for i, b in enumerate(boxes)]
    keep = []
    for i, b in enumerate(boxes):
        if group[i]:
            continue
        if any(not group[j] and _inside(b, o) and o[2] * o[3] > b[2] * b[3]
               for j, o in enumerate(boxes) if j != i):
            continue
        keep.append(i)
    # A box much wider than the other packs is two packs the detector merged
    # (seen with "instant noodle" on two leaning packs); its centre is the gap
    # between them, so it must not be matched or grasped.
    if len(keep) >= 3:
        widths = sorted(boxes[i][2] for i in keep)
        med = widths[len(widths) // 2]
        keep = [i for i in keep if boxes[i][2] <= 1.6 * med]
    return keep


def _crop(rgb, box, pad=4):
    x, y, w, h = (int(v) for v in box)
    return rgb[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]


def classify(client, rgb, res, settings: dict) -> list:
    """One report per single-pack detection — {index, box, best, sim, margin,
    sure} — embedding each pack once, whatever number of products is asked for."""
    model = settings.get('model', 'siglip-image')
    min_sim = float(settings.get('min_sim', 0.80))
    min_margin = float(settings.get('min_margin', 0.03))
    inner = float(settings.get('inner', 1.0))
    refs = _ref_embeddings(client, model, inner)
    dets = list(res.detections)
    report = []
    for i in _single_items([d.bbox for d in dets]):
        crop = _inner(_crop(rgb, dets[i].bbox), inner)
        if crop.size == 0 or not refs:
            continue
        v = _embed(client, crop, model)
        scores = sorted(((float((m @ v).max()), k) for k, m in refs.items()), reverse=True)
        best, name = scores[0]
        margin = best - (scores[1][0] if len(scores) > 1 else 0.0)
        report.append({'index': i, 'box': [round(float(b)) for b in dets[i].bbox], 'best': name,
                       'sim': round(best, 3), 'margin': round(margin, 3),
                       'sure': best >= min_sim and margin >= min_margin})
    return report


def subset(res, indices, label: str):
    """`res` with only `indices`, relabelled `label` (masks kept aligned)."""
    dets = list(res.detections)
    aligned = len(res.masks) == len(dets)
    return dataclasses.replace(
        res,
        detections=[dataclasses.replace(dets[i], cls=label) for i in indices],
        masks=[res.masks[i] for i in indices] if aligned else [],
    )


def match(client, rgb, res, product: str, settings: dict, label: str = None):
    """`res` restricted to the detections that are `product`, relabelled
    `label` (the name the caller asked for, so select_target_object(cls=label)
    keeps them), plus a report of every pack's best match for the log."""
    report = classify(client, rgb, res, settings)
    kept = [r['index'] for r in report if r['sure'] and r['best'] == product]
    return subset(res, kept, label or product), report


single_items = _single_items
