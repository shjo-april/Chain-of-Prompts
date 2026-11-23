# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

"""
Hero GIF, step 1 (GPU): run SAM 3 with and without Chain-of-Prompts on every scene of scenes.json.

For each scene this records
  - CoP from the scene's clicks (one per cell type, or one per image), with every propagated point;
  - the same clicks without CoP (plain SAM 3 with the same overlap guard and NMS);
  - per-cell clicking: the same clicks first, then one click on every remaining ground-truth cell
    (its centroid, in ground-truth id order), with the exact AJI as the clicks accumulate;
and writes <out>/<key>/{cop_cell.json, maps.npz}, which build_gif.py turns into the GIF.

    python hero_gif/run_models.py --ckpt ./checkpoints/sam3                        # writes outputs/hero_gif
    python hero_gif/run_models.py --ckpt ./checkpoints/sam3 --out hero_gif/results  # replaces the recorded run

Ground truth is used only to place the per-cell clicks and to score AJI. The LIVECell pair ships
with the repository. CoNSeP is not redistributable, so put its mask at the path given in
scenes.json first (see DATASETS.md). Each scene is compared with the recorded run in results/.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core.sam import ImageSAM
from core.cop import (ChainOfPrompts, hierarchical_similarity_gating, farthest_point_selection, _deduplicate,
                      sam_predict_best, safe_mask_insert, img_to_feat, feat_to_img)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default=os.path.join(ROOT, 'checkpoints', 'sam3'), help='SAM 3 checkpoint folder')
ap.add_argument('--out', default=os.path.join(ROOT, 'outputs', 'hero_gif'))
args = ap.parse_args()
RECORDED = os.path.join(HERE, 'results')


def load_instance_mask(path):
    """Instance-id map from a .npy file, a 2D PNG, or an RGB-encoded PNG (one id per distinct color)."""
    raw = np.load(path) if path.endswith('.npy') else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw.ndim == 3:
        b, g, r = (raw[:, :, c].astype(np.int64) for c in range(3))
        enc = r * 65536 + g * 256 + b
        _, inv = np.unique(enc, return_inverse=True)
        return inv.reshape(enc.shape).astype(np.int32)
    return raw.astype(np.int32)


def aji(pred, gt):
    """Aggregated Jaccard Index of a predicted instance map against the ground truth."""
    matched, inter, union = set(), 0, 0
    for gid in sorted(np.unique(gt))[1:]:
        g = gt == gid
        best_ji, best = 0, -1
        for pid in np.unique(pred[g]):
            if pid == 0:
                continue
            p = pred == pid
            ji = np.sum(np.logical_and(p, g)) / (np.sum(np.logical_or(p, g)) + 1e-8)
            if ji > best_ji:
                best_ji, best = ji, pid
        if best > 0:
            p = pred == best
            inter += np.sum(np.logical_and(g, p)); union += np.sum(np.logical_or(g, p)); matched.add(best)
        else:
            union += np.sum(g)
    fp = sum(int(np.sum(pred == pid)) for pid in sorted(np.unique(pred))[1:] if pid not in matched)
    return round(float(inter / (union + fp + 1e-8)), 4)


def fpr_logged(cop, start):
    """Same loop as core.cop.farthest_prompt_recursion, also returning the iteration that found each point."""
    allr, it_of = np.zeros((0, 2), np.int64), []
    used, cur = np.zeros((0, 2), np.int64), np.array(start, np.int64)
    for k in range(cop.max_iterations):
        rel = hierarchical_similarity_gating(cop.f_high, cop.f_low, cur, min_area=cop.min_area)
        new = _deduplicate(rel, allr)
        used = np.vstack([used, cur.reshape(1, 2)])
        if len(new) == 0:
            break
        allr = np.vstack([allr, new]); it_of += [k + 1] * len(new)
        nxt = farthest_point_selection(allr, used)
        if nxt is None:
            break
        cur = nxt
    return allr, it_of


def run_logged(cop, img, clicks):
    """Same order and rules as ChainOfPrompts.segment_steps, logging every prompt.

    Returns the label map, {instance id: click index}, and the events (type = click index, iter = FPR
    iteration or -1 for the user click, x, y, ok = mask inserted, user).
    """
    cop.encode(img)
    per_click = []
    for ti, (x, y) in enumerate(clicks):
        pts, its = fpr_logged(cop, img_to_feat(x, y, cop.iw, cop.ih, cop.fw, cop.fh))
        per_click.append((ti, pts, its))
    h, w = img.shape[:2]
    lab, score, areas, counter = np.zeros((h, w), np.int32), np.full((h, w), -1.0, np.float32), {}, 0
    click_of, seen, ev = {}, np.zeros((0, 2), np.int32), []
    for ti, pts, its in per_click:
        ux, uy = clicks[ti]
        m, s = sam_predict_best(cop.model, ux, uy)
        ok, counter = safe_mask_insert(m, s, lab, score, counter, areas, cop.nms_iou)
        if ok:
            click_of[counter] = ti
        ev.append(dict(type=ti, iter=-1, x=int(ux), y=int(uy), ok=bool(ok), user=True))
        ip = (np.array([feat_to_img(int(px), int(py), cop.fw, cop.fh, cop.iw, cop.ih) for px, py in pts], np.int32)
              if len(pts) else np.zeros((0, 2), np.int32))
        keep = np.ones(len(ip), bool)
        if len(seen) and len(ip):
            keep = np.sqrt(((ip[:, None, :].astype(np.float32) - seen[None].astype(np.float32)) ** 2).sum(2)).min(1) > 5.0
        for (px, py), it in zip(ip[keep], np.array(its)[keep]):
            m, s = sam_predict_best(cop.model, int(px), int(py))
            ok, counter = safe_mask_insert(m, s, lab, score, counter, areas, cop.nms_iou)
            if ok:
                click_of[counter] = ti
            ev.append(dict(type=ti, iter=int(it), x=int(px), y=int(py), ok=bool(ok), user=False))
        if keep.any():
            seen = np.vstack([seen, ip[keep]]) if len(seen) else ip[keep]
    return lab, click_of, ev


def clicks_only(cop, img, clicks):
    """The same clicks without CoP: one SAM 3 mask per click, same overlap guard and NMS."""
    cop.encode(img)
    lab, score, areas, c = np.zeros(img.shape[:2], np.int32), np.full(img.shape[:2], -1.0, np.float32), {}, 0
    for x, y in clicks:
        m, s = sam_predict_best(cop.model, x, y)
        _, c = safe_mask_insert(m, s, lab, score, c, areas, cop.nms_iou)
    return lab


def per_instance_curve(cop, img, gt, clicks, cop_aji):
    """Per-cell clicking: the CoP clicks first, then the centroid of every other ground-truth cell in id order.

    The overlap guard makes the result depend on the click order, so the order is fixed to the ground-truth
    ids. Returns the clicks, the exact AJI after about every N/60 clicks, the first click count that reaches
    the CoP AJI, and the final label map.
    """
    hit = {int(gt[y, x]) for x, y in clicks} - {0}
    pts = list(clicks)
    for i in np.unique(gt):
        if i and int(i) not in hit:
            ys, xs = np.where(gt == i)
            pts.append((int(round(xs.mean())), int(round(ys.mean()))))
    cop.encode(img)
    lab, score, areas, c = np.zeros(gt.shape, np.int32), np.full(gt.shape, -1.0, np.float32), {}, 0
    step = max(1, len(pts) // 60)
    marks = set(range(step, len(pts) + 1, step)) | {len(pts), len(clicks)}
    log, curve, match = [], [], None
    for k, (x, y) in enumerate(pts, 1):
        m, s = sam_predict_best(cop.model, x, y)
        ok, c = safe_mask_insert(m, s, lab, score, c, areas, cop.nms_iou)
        log.append(dict(x=x, y=y, ok=bool(ok), label=int(c) if ok else 0))
        if k in marks:
            v = aji(lab, gt); curve.append([k, v])
            if match is None and v >= cop_aji:
                match = k
    return dict(clicks=log, curve=curve, match_clicks=match), lab


def differences(doc, maps, key):
    """What differs from the recorded run of this scene (empty list = reproduced exactly)."""
    ref = json.load(open(os.path.join(RECORDED, key, 'cop_cell.json'), encoding='utf-8'))
    ref_maps = np.load(os.path.join(RECORDED, key, 'maps.npz'))
    diff = [f'{k} map' for k in maps if not np.array_equal(maps[k], ref_maps[k])]
    for k in doc:
        if doc[k] != ref.get(k):
            r = ref['result']
            diff.append(k + (f" (recorded AJI {r['aji']} / {r['clicks_only']['aji']} / {r['per_instance']['aji']})" if k == 'result' else ''))
    return diff


model = ImageSAM(args.ckpt, 'sam3')
cop = ChainOfPrompts(model)
for sc in json.load(open(os.path.join(HERE, 'scenes.json'), encoding='utf-8'))['scenes']:
    key, gt_path = sc['key'], os.path.join(ROOT, sc['gt'])
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f'{key}: ground-truth mask not found at {sc["gt"]} (see DATASETS.md)')
    img, gt = cv2.imread(os.path.join(ROOT, sc['img']), cv2.IMREAD_COLOR), load_instance_mask(gt_path)
    clicks = [(int(x), int(y)) for x, y in sc['clicks']]

    lab, click_of, ev = run_logged(cop, img, clicks)
    ref, _, _ = ChainOfPrompts(model).segment(img, clicks)
    co = clicks_only(cop, img, clicks)
    res = dict(key=key, img=sc['img'], clicks=[list(c) for c in clicks], same_as_cop_segment=bool(np.array_equal(ref, lab)),
               n_inst=int(lab.max()), n_gt=int((np.unique(gt) > 0).sum()), aji=aji(lab, gt), n_prompts=len(ev),
               clicks_only=dict(n_inst=int(co.max()), aji=aji(co, gt)))
    pi, pl = per_instance_curve(cop, img, gt, clicks, res['aji'])
    res['per_instance'] = dict(n_clicks=len(pi['clicks']), n_inst=int(pl.max()), aji=pi['curve'][-1][1], match_clicks=pi['match_clicks'])
    doc = dict(result=res, events=ev, per_instance=pi)
    if sc.get('seed_clicks'):                                      # the other clicks behind the range quoted in the GIF footnote
        doc['seed_runs'] = [dict(seed=i, click=[x, y], aji=aji(cop.segment(img, [(x, y)])[0], gt))
                            for i, (x, y) in enumerate(sc['seed_clicks'])]
    maps = dict(cop=lab, click_of=np.array([click_of.get(i, -1) for i in range(int(lab.max()) + 1)]), per_cell=pl)

    diff = differences(doc, maps, key) if os.path.exists(os.path.join(RECORDED, key, 'cop_cell.json')) else None
    d = os.path.join(args.out, key); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, 'maps.npz'), **maps)
    with open(os.path.join(d, 'cop_cell.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(doc, f, indent=1); f.write('\n')
    print(f"{key}: w/ CoP {len(clicks)} click(s) AJI {res['aji']} ({res['n_inst']} of {res['n_gt']} cells) | "
          f"same clicks w/o CoP AJI {res['clicks_only']['aji']} | every cell, {res['per_instance']['n_clicks']} clicks AJI {res['per_instance']['aji']}")
    if 'seed_runs' in doc:
        print('  seed clicks AJI:', ' '.join(f"{r['aji']:.4f}" for r in doc['seed_runs']))
    if diff is not None:
        print('  recorded run reproduced exactly' if not diff else '  differs from the recorded run in: ' + '; '.join(diff))
