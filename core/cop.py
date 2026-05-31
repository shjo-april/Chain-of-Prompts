# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

"""
Chain-of-Prompts (CoP) — training-free group interaction for cell instance segmentation.

Public demo core. Given a frozen SAM encoder and one user click *per cell type*, CoP
recursively expands that single click into every same-type instance, then decodes each
discovered point into an instance mask. No training, no GT, no per-instance clicking.

Pipeline (matches the paper):
    1. Encode image once with the frozen SAM encoder        -> encode()
    2. Hierarchical Similarity Gating (HSG)                 -> hierarchical_similarity_gating()
    3. Farthest Prompt Recursion (FPR)                      -> farthest_prompt_recursion()
    4. Training-free mask decoding (guard + NMS)            -> decode_points()

End-to-end entry point:
    cop = ChainOfPrompts(model)          # model = core.sam.ImageSAM(..., 'sam3')
    label_map, overlay, info = cop.segment(cv_image, clicks_xy)

This module intentionally contains NO evaluation / benchmark code (AJI, PQ, baselines,
plotting). Those live in the benchmark release.
"""

import cv2
import torch
import numpy as np
import torch.nn.functional as F


# ============================================================
# Coordinate conversion (image <-> feature grid)
# ============================================================
def img_to_feat(ix, iy, iw, ih, fw, fh):
    return (int(np.clip(np.floor(ix / iw * fw), 0, fw - 1)),
            int(np.clip(np.floor(iy / ih * fh), 0, fh - 1)))


def feat_to_img(px, py, fw, fh, iw, ih):
    return (int(np.clip(np.floor(px / fw * iw), 0, iw - 1)),
            int(np.clip(np.floor(py / fh * ih), 0, ih - 1)))


# ============================================================
# Feature preprocessing
# ============================================================
def preprocess_features(feat: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
    """L2-normalize a (D, H, W) feature map and optionally resize to target_size."""
    feat = feat.to(dtype=torch.float32)
    feat = F.normalize(feat, p=2, dim=0)
    if target_size is not None and (feat.shape[1] != target_size[0] or feat.shape[2] != target_size[1]):
        feat = F.interpolate(feat.unsqueeze(0), size=target_size,
                             mode='bilinear', align_corners=False).squeeze(0)
        feat = F.normalize(feat, p=2, dim=0)
    return feat


# ============================================================
# Step 2: Hierarchical Similarity Gating (HSG)
# ============================================================
@torch.no_grad()
def hierarchical_similarity_gating(f_high, f_low, prompt_fxy, min_area=10):
    """
    Turn a single feature-space prompt into a set of reliable same-type points.

    1. High-res similarity:  sim = <prompt_feat_high, F_high>, clamp[0,1]
    2. Non-parametric gate:  foreground = sim > (mean + std)
    3. Connected components: weighted centroid per component (area >= min_area) -> candidates
    4. Low-res gating:       keep candidates whose F_low similarity > (mean + std)

    Returns reliable points in feature coordinates, shape (N, 2) as (x, y).
    """
    _, fh, fw = f_high.shape
    px = int(np.clip(int(prompt_fxy[0]), 0, fw - 1))
    py = int(np.clip(int(prompt_fxy[1]), 0, fh - 1))

    with torch.amp.autocast('cuda', dtype=torch.float32, enabled=False):
        prompt_feat = f_high[:, py, px]
        sim = torch.einsum('d, dhw -> hw', prompt_feat, f_high)
        sim = torch.clamp(sim, 0.0, 1.0)
        thresh = sim.mean() + sim.std()
        fg_gpu = sim > thresh

    fg_map = fg_gpu.cpu().numpy().astype(np.uint8)
    sim_np = sim.float().cpu().numpy()
    num_labels, labels = cv2.connectedComponents(fg_map, connectivity=8)

    candidates = []
    for cid in range(1, num_labels):
        comp = labels == cid
        if comp.sum() < min_area:
            continue
        ys, xs = np.where(comp)
        w = sim_np[ys, xs]
        cy = int(np.round(np.average(ys, weights=w)))
        cx = int(np.round(np.average(xs, weights=w)))
        candidates.append((cx, cy))
    candidates = np.array(candidates, dtype=np.int64) if candidates else np.zeros((0, 2), dtype=np.int64)

    if len(candidates) == 0:
        return candidates

    # --- Low-res gating ---
    with torch.amp.autocast('cuda', dtype=torch.float32, enabled=False):
        prompt_feat_low = f_low[:, py, px]
        cand_x = torch.from_numpy(candidates[:, 0]).long().to(f_low.device).clamp(0, fw - 1)
        cand_y = torch.from_numpy(candidates[:, 1]).long().to(f_low.device).clamp(0, fh - 1)
        scores_low = torch.einsum('d, dn -> n', prompt_feat_low, f_low[:, cand_y, cand_x])
        scores_np = scores_low.cpu().numpy()
    thr_low = scores_np.mean() + scores_np.std()
    rel_mask = scores_np > thr_low
    return candidates[rel_mask] if rel_mask.any() else np.zeros((0, 2), dtype=np.int64)


# ============================================================
# Step 3: Farthest Prompt Recursion (FPR)
# ============================================================
def _deduplicate(new_pts, existing, min_dist=3.0):
    if len(new_pts) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if len(existing) == 0:
        return new_pts
    diffs = new_pts.astype(np.float32)[:, None, :] - existing.astype(np.float32)[None, :, :]
    min_dists = np.sqrt((diffs ** 2).sum(axis=2)).min(axis=1)
    keep = min_dists > min_dist
    return new_pts[keep] if keep.any() else np.zeros((0, 2), dtype=np.int64)


def farthest_point_selection(reliable, used):
    """Pick the reliable point farthest (image-space) from all previously used prompts."""
    if len(reliable) == 0:
        return None
    diffs = reliable.astype(np.float32)[:, None, :] - used.astype(np.float32)[None, :, :]
    min_dists = np.sqrt((diffs ** 2).sum(axis=2)).min(axis=1)
    return reliable[np.argmax(min_dists)]


@torch.no_grad()
def farthest_prompt_recursion(f_high, f_low, init_prompt_fxy,
                              max_iterations=30, min_area=10, dedup_dist=3.0):
    """
    Recursively expand a single feature-space click into whole-image coverage.

    Repeats { HSG -> dedup new points -> re-prompt the farthest uncovered point }
    until no new points are discovered (convergence) or max_iterations is reached.

    Returns reliable points in feature coordinates, shape (N, 2) as (x, y).
    """
    all_reliable = np.zeros((0, 2), dtype=np.int64)
    used_prompts = np.zeros((0, 2), dtype=np.int64)
    current = np.array(init_prompt_fxy, dtype=np.int64)

    for _ in range(max_iterations):
        reliable = hierarchical_similarity_gating(f_high, f_low, current, min_area=min_area)
        new_points = _deduplicate(reliable, all_reliable, min_dist=dedup_dist)
        used_prompts = np.vstack([used_prompts, current.reshape(1, 2)])

        if len(new_points) == 0:
            break
        all_reliable = np.vstack([all_reliable, new_points])

        nxt = farthest_point_selection(all_reliable, used_prompts)
        if nxt is None:
            break
        current = nxt

    return all_reliable


# ============================================================
# Step 4: Training-free mask decoding (SAM + guard + NMS)
# ============================================================
def sam_predict_best(model, ix, iy):
    """Prompt SAM at (ix, iy); return (best-score mask, score).

    Uses the public ImageSAM API; the image must already be encoded
    (ImageSAM caches its state from the preceding encode_image() call).
    """
    masks, scores = model.predict(
        point_coords=[(int(ix), int(iy))], point_labels=[1], postprocessing=None)
    # multimask_output=False -> a single (best) mask is returned.
    return masks[0].astype(np.bool_), float(scores[0])


def safe_mask_insert(mask, score, label_map, score_map, inst_counter,
                     inst_areas=None, nms_iou=0.5):
    """
    Insert mask into label_map.
      Guard: reject if it overlaps 2+ existing instances.
      NMS:   if nms_iou > 0, reject near-duplicates (IoU > nms_iou with any existing).
    Existing instances are never modified. Returns (inserted: bool, inst_counter).
    """
    mask_area = int(mask.sum())
    if mask_area == 0:
        return False, inst_counter

    overlap = label_map[mask]
    ids, counts = np.unique(overlap, return_counts=True)
    fg = ids > 0
    ids, counts = ids[fg], counts[fg]

    if len(ids) >= 2:  # guard
        return False, inst_counter

    if nms_iou > 0 and inst_areas is not None and len(ids) == 1:
        eid, inter = int(ids[0]), int(counts[0])
        earea = inst_areas.get(eid, 0)
        if earea > 0 and inter / (mask_area + earea - inter + 1e-8) > nms_iou:
            return False, inst_counter

    inst_counter += 1
    update = mask & (score > score_map)
    label_map[update] = inst_counter
    score_map[update] = score
    if inst_areas is not None:
        inst_areas[inst_counter] = mask_area
    return True, inst_counter


# ============================================================
# Visualization
# ============================================================
_MOD, _M = 2 ** 24, 11400719


def encode_mask(label_map, color_format='rgb'):
    """Deterministic per-id color encoding of an instance label map (for saving)."""
    flat = label_map.flatten().astype(np.int64)
    x = (flat * _M) % _MOD
    x[flat == 0] = 0
    r = ((x >> 16) & 0xFF).astype(np.uint8)
    g = ((x >> 8) & 0xFF).astype(np.uint8)
    b = (x & 0xFF).astype(np.uint8)
    H, W = label_map.shape
    if color_format == 'bgr':
        return np.stack([b, g, r], axis=-1).reshape(H, W, 3)
    return np.stack([r, g, b], axis=-1).reshape(H, W, 3)


def colorize_instance_map(label_map, cv_image, alpha=0.35, seed=42):
    """Alpha-blend random per-instance colors over the image."""
    overlay = cv_image.copy().astype(np.float32)
    n = int(label_map.max())
    if n == 0:
        return overlay.astype(np.uint8)
    rng = np.random.default_rng(seed)
    palette = rng.integers(60, 240, size=(n + 1, 3)).astype(np.float32)
    palette[0] = 0
    for i in range(1, n + 1):
        m = label_map == i
        if m.any():
            overlay[m] = (1 - alpha) * overlay[m] + alpha * palette[i]
    return overlay.astype(np.uint8)


# ============================================================
# End-to-end Chain-of-Prompts
# ============================================================
class ChainOfPrompts:
    """
    Wrap a frozen SAM (e.g. core.sam.ImageSAM(..., 'sam3')) into the CoP pipeline.

    Usage:
        from core.sam import ImageSAM
        from core.cop import ChainOfPrompts
        cop = ChainOfPrompts(ImageSAM('<ckpt_dir>/sam3', 'sam3'))
        label_map, overlay, info = cop.segment(cv_image, clicks_xy=[(x1, y1), (x2, y2)])

    `clicks_xy` is one (x, y) click per cell type, in image pixel coordinates.
    """

    def __init__(self, model, max_iterations=30, min_area=10, nms_iou=0.5):
        self.model = model
        self.max_iterations = max_iterations
        self.min_area = min_area
        self.nms_iou = nms_iou

    @torch.no_grad()
    def encode(self, cv_image):
        """Encode a BGR image once; cache multi-scale features + sizes.

        ImageSAM.encode_image returns the frozen backbone FPN as a list of
        (1, C, H, W) tensors; [0] is high-resolution, [2] is low-resolution.
        It also caches the predictor state internally for later predict() calls.
        """
        feats = self.model.encode_image(cv_image)
        f_high = feats[0][0].to(dtype=torch.float32)
        f_low = feats[2][0].to(dtype=torch.float32)
        self.ih, self.iw = cv_image.shape[:2]
        self.fh, self.fw = f_high.shape[1], f_high.shape[2]
        self.f_high = f_high
        with torch.amp.autocast('cuda', dtype=torch.float32, enabled=False):
            self.f_low = preprocess_features(f_low, target_size=(self.fh, self.fw))
        return self

    @torch.no_grad()
    def propagate(self, click_xy):
        """One user click (image coords) -> propagated points (image coords)."""
        fx, fy = img_to_feat(*click_xy, self.iw, self.ih, self.fw, self.fh)
        feat_pts = farthest_prompt_recursion(
            self.f_high, self.f_low, (fx, fy),
            max_iterations=self.max_iterations, min_area=self.min_area)
        if len(feat_pts) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return np.array([feat_to_img(int(px), int(py), self.fw, self.fh, self.iw, self.ih)
                         for px, py in feat_pts], dtype=np.int32)

    @torch.no_grad()
    def segment_steps(self, cv_image, clicks_xy):
        """
        Replay Chain-of-Prompts one click at a time, yielding a snapshot per click.

        This is the building block behind the Figure-2 O(T) reproduction: feed one
        click per cell type, in order, and visualize the cumulative state after each.

        Each yielded dict contains:
            'click_index'       : int, 0-based (one per cell type)
            'user_click'        : (x, y) the click decoded this step
            'propagated_points' : (M, 2) int32, propagated points inserted this step
            'label_map'         : (H, W) int32, cumulative instance map (a copy)
            'inst_to_click'     : {instance_id: click_index} cumulative
            'num_instances'     : int, cumulative instance count
        """
        self.encode(cv_image)
        label_map = np.zeros((self.ih, self.iw), dtype=np.int32)
        score_map = np.full((self.ih, self.iw), -1.0, dtype=np.float32)
        inst_areas = {} if self.nms_iou > 0 else None
        inst_counter = 0
        inst_to_click = {}
        seen = np.zeros((0, 2), dtype=np.int32)

        for ci, (ix, iy) in enumerate(clicks_xy):
            # decode the user click itself
            mask, score = sam_predict_best(self.model, int(ix), int(iy))
            ok, inst_counter = safe_mask_insert(mask, score, label_map, score_map,
                                                inst_counter, inst_areas, self.nms_iou)
            if ok:
                inst_to_click[inst_counter] = ci

            # HSG + FPR propagation, then decode each new point
            pts = self.propagate((int(ix), int(iy)))
            new_pts = _deduplicate(pts, seen, min_dist=5.0) if len(seen) else pts
            inserted_pts = []
            for (px, py) in new_pts:
                m, s = sam_predict_best(self.model, int(px), int(py))
                ok2, inst_counter = safe_mask_insert(m, s, label_map, score_map,
                                                     inst_counter, inst_areas, self.nms_iou)
                if ok2:
                    inst_to_click[inst_counter] = ci
                    inserted_pts.append((int(px), int(py)))
            if len(new_pts):
                seen = np.vstack([seen, new_pts]) if len(seen) else new_pts

            yield {
                'click_index': ci,
                'user_click': (int(ix), int(iy)),
                'propagated_points': (np.array(inserted_pts, dtype=np.int32)
                                      if inserted_pts else np.zeros((0, 2), dtype=np.int32)),
                'label_map': label_map.copy(),
                'inst_to_click': dict(inst_to_click),
                'num_instances': inst_counter,
            }

    @torch.no_grad()
    def segment(self, cv_image, clicks_xy):
        """
        Full pipeline: encode -> per click {decode click + propagate + decode points}.

        Returns:
            label_map (H, W) int32 instance map,
            overlay   (H, W, 3) BGR visualization,
            info      dict with per-click stats (#propagated, #instances, ...).
        """
        label_map = np.zeros((self.ih, self.iw) if hasattr(self, 'ih') else (1, 1), dtype=np.int32)
        per_click = []
        for step in self.segment_steps(cv_image, clicks_xy):
            label_map = step['label_map']
            per_click.append({'click': step['user_click'],
                              'propagated_points': int(len(step['propagated_points'])),
                              'total_instances': step['num_instances']})
        overlay = colorize_instance_map(label_map, cv_image)
        info = {'num_clicks': len(clicks_xy),
                'num_instances': int(label_map.max()),
                'per_click': per_click}
        return label_map, overlay, info
