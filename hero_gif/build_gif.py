# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

"""
Hero GIF, step 2 (CPU only): compose assets/hero.gif from the recorded runs.

Each scene shows one image twice, SAM 3 without CoP (left) and SAM 3 with CoP (right), with the
AJI against the ground truth, and a summary scene closes the GIF. Every mask, click, and number
comes from scenes.json and results/<key>/ (written by run_models.py), so no model is needed here.

    python hero_gif/build_gif.py                  # writes assets/hero.gif
    python hero_gif/build_gif.py --preview DIR    # writes every frame to DIR as a PNG instead

The committed GIF was rendered on Windows with Segoe UI. On other systems the script falls back
to the DejaVu Sans bundled with matplotlib: same layout, slightly different glyphs.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--results', default=os.path.join(HERE, 'results'), help='folder written by run_models.py')
ap.add_argument('--out', default=os.path.join(ROOT, 'assets', 'hero.gif'))
ap.add_argument('--preview', metavar='DIR', help='write the frames as PNGs to DIR and skip the GIF')
args = ap.parse_args()

W, H = 1280, 960
TEAL, TEAL_BG, MINT_LINE = (0, 102, 102), (232, 245, 244), (150, 205, 200)
INK, MUTED, LINE, WHITE, GREY_BAR = (16, 24, 31), (91, 105, 117), (220, 229, 233), (255, 255, 255), (178, 188, 196)
TAB_DIM = (191, 217, 217)                                          # inactive scene tab
TYPE_COL = [(232, 150, 28), (0, 128, 128), (52, 112, 220)]          # CoP: one color per click (click order, not cell type)
PI_COL, PI_FILL = (0, 110, 175), (0, 170, 235)                     # SAM 3 w/o CoP: one click per cell
V = json.load(open(os.path.join(HERE, 'scenes.json'), encoding='utf-8'))
TITLE, BADGES = 'Chain-of-Prompts (CoP)', V['badges']
STAGES = [('Click', 'one per cell type · one per image'), ('Chain', 'spreads to look-alike cells'), ('Segment', 'frozen SAM 3 masks')]
M, GAP, TOP = 24, 24, 210                                          # side margin · gap between panels · top of the panels
BLOCK_MIN, BOTTOM = 140, 954                                       # minimum height of the bar block · bottom edge

WIN_FONTS = 'C:/Windows/Fonts'
FALLBACK = {'segoeui.ttf': 'DejaVuSans.ttf', 'seguisb.ttf': 'DejaVuSans-Bold.ttf', 'segoeuib.ttf': 'DejaVuSans-Bold.ttf'}


def F(name, size):
    path = os.path.join(WIN_FONTS, name)
    if not os.path.exists(path):
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', FALLBACK[name])
    return ImageFont.truetype(path, size)


F_TITLE, F_BADGE, F_AWARD, F_SUB = F('segoeuib.ttf', 40), F('seguisb.ttf', 16), F('segoeuib.ttf', 17), F('segoeui.ttf', 16)
F_PILL_B, F_PILL, F_H, F_HS, F_CHIP = F('segoeuib.ttf', 15), F('segoeui.ttf', 15), F('segoeuib.ttf', 20), F('segoeui.ttf', 15), F('seguisb.ttf', 15)
F_TAB, F_MSG = F('segoeuib.ttf', 16), F('segoeuib.ttf', 18)

logo = Image.open(os.path.join(HERE, 'logos', 'aibl.png')).convert('RGB'); logo = logo.resize((int(logo.width * 50 / logo.height), 50), Image.LANCZOS)
snu = Image.open(os.path.join(HERE, 'logos', 'snu.png')).convert('RGBa')
snu = snu.resize((round(snu.width * 50 / snu.height), 50), Image.LANCZOS).convert('RGBA')


def overlay(mask_lab, col, size, alpha=1.0):
    m = mask_lab > 0
    big = np.zeros(mask_lab.shape + (4,), np.uint8)
    big[m] = col + (int(160 * alpha),)
    edge = np.zeros_like(m)
    edge[1:, :] |= mask_lab[1:, :] != mask_lab[:-1, :]; edge[:, 1:] |= mask_lab[:, 1:] != mask_lab[:, :-1]
    edge &= m | np.roll(m, 1, 0) | np.roll(m, 1, 1)
    big[edge] = (255, 255, 255, int(235 * alpha))
    return Image.fromarray(big).resize(size, Image.NEAREST)


class Scene:
    """One scene: recorded run, panel layout, cached overlays."""

    def __init__(self, v, idx):
        self.v, self.idx = v, idx
        run = json.load(open(os.path.join(args.results, v['key'], 'cop_cell.json'), encoding='utf-8'))
        self.R, self.EV, self.PI = run['result'], run['events'], run['per_instance']
        assert self.R['same_as_cop_segment']
        maps = np.load(os.path.join(args.results, v['key'], 'maps.npz'))
        self.LAB, self.TYPES, self.PLAB = maps['cop'], maps['click_of'], maps['per_cell']
        self.T, self.N = len(self.R['clicks']), len(self.PI['clicks'])
        seeds = [r['aji'] for r in run.get('seed_runs', [])]        # fills the {seed_*} fields of the notes in scenes.json
        self.stats = dict(n_seeds=len(seeds), seed_lo=min(seeds), seed_hi=max(seeds), seed_mean=sum(seeds) / len(seeds)) if seeds else {}
        self.note = v['note'].format(**self.stats)
        self.CURVE = {k: val for k, val in self.PI['curve']}
        self.CO_AJI, self.PI_AJI, self.SAME_AJI = self.R['aji'], self.PI['curve'][-1][1], self.CURVE[self.T]
        img0 = Image.open(os.path.join(ROOT, v['img'])).convert('RGB')
        asp = img0.width / img0.height
        pw = min((W - 2 * M - GAP) / 2, (BOTTOM - BLOCK_MIN - TOP) * asp)  # as large as the width or the height allows
        self.pw, self.ph = int(pw), int(round(int(pw) / asp))
        x0 = (W - (2 * self.pw + GAP)) // 2
        self.panel = [dict(x=x0, y=TOP), dict(x=x0 + self.pw + GAP, y=TOP)]
        self.block = (TOP + self.ph + 14, BOTTOM)                  # the bar block takes all the height that is left
        self.S = self.pw / img0.width
        self.img0, self.IMG = img0, img0.resize((self.pw, self.ph), Image.LANCZOS)
        self.K_OF = np.full(int(self.PLAB.max()) + 1, 10 ** 9, np.int64)   # left panel: label -> index of the click that made it
        for k, c in enumerate(self.PI['clicks'], 1):
            if c['ok']:
                self.K_OF[c['label']] = k
        self.cache = {}

    def ov_cop(self, t, a, size=None):
        size = size or (self.pw, self.ph)
        key = ('cop', t, a, size)
        if key not in self.cache:
            ids = [i for i in range(1, len(self.TYPES)) if self.TYPES[i] == t]
            self.cache[key] = overlay(np.where(np.isin(self.LAB, ids), self.LAB, 0), TYPE_COL[t], size, a)
        return self.cache[key]

    def ov_pi(self, k, size=None):
        size = size or (self.pw, self.ph)
        key = ('pi', k, size)
        if key not in self.cache:
            self.cache[key] = overlay(np.where(self.K_OF[self.PLAB] <= k, self.PLAB, 0), PI_FILL, size)
        return self.cache[key]

    def ov_first(self):
        """Left panel, masks of the shared clicks (the first T): solid fill and a thick ring so small cells stay visible."""
        if 'first' not in self.cache:
            m = np.isin(self.PLAB, [c['label'] for c in self.PI['clicks'][:self.T] if c['ok']])
            big = np.zeros(m.shape + (4,), np.uint8)
            big[m] = PI_FILL + (235,)
            k = max(3, round(3 / self.S))
            ring = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~m
            big[ring] = PI_COL + (255,)
            self.cache['first'] = Image.fromarray(big).resize((self.pw, self.ph), Image.NEAREST)
        return self.cache['first']

    def pxy(self, j, x, y):
        return self.panel[j]['x'] + x * self.S, self.panel[j]['y'] + y * self.S


SC = [Scene(v, i) for i, v in enumerate(V['scenes'])]


def rrect(dr, box, r, fill=None, outline=None, width=1):
    dr.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def header(im, dr, scene, active=None, done=(), stages=True):
    im.paste(snu, (34, 24), snu)
    im.paste(logo, (W - 34 - logo.width, 24))
    tw = dr.textlength(TITLE, font=F_TITLE)
    parts = [t_.rsplit(' ', 1) for t_ in BADGES]
    vw = max(dr.textlength(v, font=F_BADGE) for v, _ in parts) + 14
    aw = max(dr.textlength(a, font=F_AWARD) for _, a in parts) + 21
    x0 = (W - (tw + 14 + vw + aw)) / 2
    dr.text((x0, 20), TITLE, font=F_TITLE, fill=TEAL)
    n = len(parts); bh = 26; y = 50 - (n * bh + (n - 1) * 4) / 2; bx = x0 + tw + 14
    for v, a in parts:
        rrect(dr, (bx, y, bx + vw + aw, y + bh), 13, fill=TEAL)
        dr.text((bx + vw, y + bh / 2), v, font=F_BADGE, fill=WHITE, anchor='rm')
        dr.text((bx + vw + 7, y + bh / 2), a, font=F_AWARD, fill=WHITE, anchor='lm')
        y += bh + 4
    tabs = [s.v['tab'] for s in SC] + ['Summary']                   # scene tabs: fixed position and size, only the current one is solid
    tw_ = (W - 2 * M - 16 * (len(tabs) - 1)) / len(tabs); x = M
    cur = len(SC) if scene is None else scene
    for i, t in enumerate(tabs):
        on = i == cur
        rrect(dr, (x, 86, x + tw_, 112), 8, fill=TEAL if on else TAB_DIM)
        dr.text((x + tw_ / 2, 99), t, font=F_TAB, fill=WHITE if on else TEAL, anchor='mm')
        x += tw_ + 16
    if not stages:
        return
    parts = [(f'{k + 1} · {a}', b, k) for k, (a, b) in enumerate(STAGES)]
    widths = [dr.textlength(a, font=F_PILL_B) + dr.textlength('  ' + b, font=F_PILL) + 26 for a, b, _ in parts]
    x = (W - (sum(widths) + 34 * (len(parts) - 1))) / 2
    for (a, b, k), w in zip(parts, widths):
        fg, bg, bd, sub = ((WHITE, TEAL, None, (215, 238, 236)) if k == active else
                           (TEAL, TEAL_BG, MINT_LINE, TEAL) if k in done else (MUTED, WHITE, LINE, MUTED))
        rrect(dr, (x, 120, x + w, 146), 13, fill=bg, outline=bd, width=1 if bd else 0)
        dr.text((x + 13, 133), a, font=F_PILL_B, fill=fg, anchor='lm')
        dr.text((x + 13 + dr.textlength(a, font=F_PILL_B), 133), '  ' + b, font=F_PILL, fill=sub, anchor='lm')
        if k < len(parts) - 1:
            dr.text((x + w + 17, 132), '→', font=F_PILL_B, fill=MUTED, anchor='mm')
        x += w + 34


def star(dr, cx, cy, col, rr=14):
    pts = []
    for i in range(10):
        ang = np.pi / 2 + i * np.pi / 5; r_ = rr if i % 2 == 0 else rr * 0.43
        pts.append((cx + r_ * np.cos(ang), cy - r_ * np.sin(ang)))
    dr.polygon(pts, fill=col, outline=WHITE)


def panel_titles(dr, sc):
    heads = [('SAM 3 w/o CoP', '1 click per cell · O(N)', PI_COL), ('SAM 3 w/ CoP (Ours)', sc.v['right_sub'] + ' · O(T)', TEAL)]
    for j, (a, b, col) in enumerate(heads):
        x0, y0 = sc.panel[j]['x'], sc.panel[j]['y']
        dr.rectangle((x0 - 2, y0 - 2, x0 + sc.pw + 1, y0 + sc.ph + 1), outline=col, width=3)
        dr.text((x0, y0 - 10), a, font=F_H, fill=col, anchor='ls')
        dr.text((x0 + dr.textlength(a, font=F_H) + 10, y0 - 11), b, font=F_HS, fill=MUTED, anchor='ls')


def legend(dr, sc, st):
    """CoP click legend under the stage pills (colors follow the click order), led by the image description."""
    items = [f'Click {t + 1} · {lb}' for t, lb in enumerate(sc.v['labels'])]
    ws = [dr.textlength(s_, font=F_CHIP) + 36 for s_ in items]
    lead = sc.v['lead']; lw = dr.textlength(lead, font=F_SUB) + 16
    x = (W - (lw + sum(ws) + 8 * (len(ws) - 1))) / 2; y = 154
    dr.text((x, y + 12), lead, font=F_SUB, fill=MUTED, anchor='lm'); x += lw
    for t, (s_, w) in enumerate(zip(items, ws)):
        on = t < st['clicks'][1]
        rrect(dr, (x, y, x + w, y + 24), 12, fill=WHITE, outline=TYPE_COL[t] if on else LINE, width=2 if on else 1)
        dr.ellipse((x + 10, y + 6, x + 22, y + 18), fill=TYPE_COL[t] if on else LINE)
        dr.text((x + 27, y + 12), s_, font=F_CHIP, fill=INK if on else MUTED, anchor='lm')
        x += w + 8


def live_block(dr, sc, st):
    """Bar block under the panels: clicks per method (axis = number of cells N) and the live AJI, then the message and the note."""
    y0, y1 = sc.block
    rows_h = (y1 - y0) - 54
    rh = min(82, rows_h / 2)                                       # rows and text grow with the height that is left
    z = (rh - 36) / 46
    f_lab, f_txt, f_aji = F('segoeuib.ttf', round(15 + 9 * z)), F('segoeuib.ttf', round(15 + 6 * z)), F('segoeuib.ttf', round(22 + 14 * z))
    f_msg = F('segoeuib.ttf', round(18 + 4 * z))
    bh = round(rh * 0.62)
    yr = y0 + (rows_h - 2 * rh) / 2
    xa, xb = 300, 900
    rows = [('SAM 3 w/o CoP', PI_COL, 0), ('SAM 3 w/ CoP (Ours)', TEAL, 1)]
    for i, (name, col, j) in enumerate(rows):
        yc = yr + i * rh + rh / 2
        dr.text((M + 16, yc), name, font=f_lab, fill=col, anchor='lm')
        rrect(dr, (xa, yc - bh / 2, xb, yc + bh / 2), 6, fill=(244, 246, 247))     # axis (N cells)
        c = st['clicks'][j]
        if c:
            L = max((xb - xa) * c / sc.N, 8)
            rrect(dr, (xa, yc - bh / 2, xa + L, yc + bh / 2), 6, fill=col if j else (120, 170, 205))
        txt = f'{c} click{"s" if c != 1 else ""}' + (f'  ({sc.N / c:.0f}× fewer)' if j == 1 and st.get('final') else '')
        end = xa + max((xb - xa) * c / sc.N, 8) if c else xa
        dr.text((end + 12 if end < xb - 160 else xb + 12, yc), txt, font=f_txt, fill=col, anchor='lm')   # after the bar, or after the axis when the bar is full
        v = st['aji'][j]
        aj = 'AJI –' if v is None else f'AJI {v:.3f}' + (f'  ({100 * v / sc.PI_AJI:.0f}%)' if j == 1 and st.get('final') else '')
        dr.text((W - M - 16, yc), aj, font=f_aji, fill=col, anchor='rm')
    msg = st.get('msg')
    if msg:
        dr.text((W / 2, y1 - 40 - 8 * z), msg[0], font=f_msg, fill=msg[1], anchor='mm')
    dr.text((W / 2, y1 - 12), sc.note, font=F_SUB, fill=MUTED, anchor='mm')


def frame(sc, st, stage, done_st):
    im = Image.new('RGB', (W, H), WHITE); dr = ImageDraw.Draw(im)
    header(im, dr, sc.idx, stage, done_st)
    left = sc.IMG.copy().convert('RGBA')
    if st['k_pi']:
        left.alpha_composite(sc.ov_pi(st['k_pi']))
        left.alpha_composite(sc.ov_first())
    right = sc.IMG.copy().convert('RGBA')
    for t, a in st['m'].items():
        right.alpha_composite(sc.ov_cop(t, a))
    im.paste(left.convert('RGB'), (sc.panel[0]['x'], sc.panel[0]['y'])); im.paste(right.convert('RGB'), (sc.panel[1]['x'], sc.panel[1]['y']))
    panel_titles(dr, sc)
    legend(dr, sc, st)
    rr = 3 if sc.N > 150 else 4
    for c in sc.PI['clicks'][sc.T:st['k_pi']]:                      # the clicks a person keeps adding, one per cell
        cx, cy = sc.pxy(0, c['x'], c['y'])
        dr.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=WHITE, outline=PI_COL, width=2)
    for c in sc.PI['clicks'][:min(sc.T, st['k_pi'])]:                # the clicks shared with CoP: hollow circles so the mask stays visible
        cx, cy = sc.pxy(0, c['x'], c['y'])
        dr.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), outline=WHITE, width=5)
        dr.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), outline=PI_COL, width=3)
    for t, x, y in st['p']:
        cx, cy = sc.pxy(1, x, y)
        dr.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=TYPE_COL[t], outline=WHITE, width=2)
    for t, x, y in st['stars']:
        star(dr, *sc.pxy(1, x, y), TYPE_COL[t])
    live_block(dr, sc, st)
    return im


def summary_frame():
    """Last scene: the final result of every scene side by side (w/o and w/ CoP) with its numbers on the right."""
    im = Image.new('RGB', (W, H), WHITE); dr = ImageDraw.Draw(im)
    header(im, dr, None, stages=False)
    dr.text((W / 2, 130), V['summary_site'], font=F_SUB, fill=MUTED, anchor='mm')
    top, bot, title_h, gap_r = 146, 922, 30, 14
    stats_w = 380
    asp = [sc.img0.width / sc.img0.height for sc in SC]
    tw = min((W - 2 * M - 12 - 24 - stats_w) / 2, (bot - top - len(SC) * title_h - gap_r * (len(SC) - 1)) / sum(1 / a for a in asp))
    tw = int(tw)
    y = top
    for sc, a in zip(SC, asp):
        th = int(round(tw / a)); rh = title_h + th
        dr.text((M, y + 14), sc.v['tab'], font=F_H, fill=INK, anchor='lm')
        ty = y + title_h
        for j, (col, cap) in enumerate([(PI_COL, f'w/o CoP · {sc.N} clicks'), (TEAL, f'w/ CoP · {sc.T} click{"s" if sc.T > 1 else ""}')]):
            base = sc.img0.resize((tw, th), Image.LANCZOS).convert('RGBA')
            if j == 0:
                base.alpha_composite(sc.ov_pi(10 ** 9, (tw, th)))
            else:
                for t in range(sc.T):
                    base.alpha_composite(sc.ov_cop(t, 1.0, (tw, th)))
            x = M + j * (tw + 12)
            im.paste(base.convert('RGB'), (x, ty))
            dr.rectangle((x - 2, ty - 2, x + tw + 1, ty + th + 1), outline=col, width=3)
            cw = dr.textlength(cap, font=F_PILL_B) + 16                # label in the top-left corner of the thumbnail
            rrect(dr, (x + 6, ty + 6, x + 6 + cw, ty + 32), 6, fill=col)
            dr.text((x + 14, ty + 19), cap, font=F_PILL_B, fill=WHITE, anchor='lm')
        bx0 = M + 2 * tw + 12 + 24; bx1 = W - M
        ys = np.linspace(ty + 24, ty + th - 24, 6)
        cl = f"{sc.T} click{'s' if sc.T > 1 else ''}"
        dr.text((bx0, ys[0]), f'Same {cl} · AJI', font=F_PILL_B, fill=INK, anchor='lm')
        dr.text((bx0, ys[1]), f'w/o CoP {sc.SAME_AJI:.3f}  →  w/ CoP {sc.CO_AJI:.3f}', font=F('segoeuib.ttf', 20), fill=TEAL, anchor='lm')
        for r, (lab, col, n, aj) in enumerate([('w/o CoP', PI_COL, sc.N, sc.PI_AJI), ('w/ CoP', TEAL, sc.T, sc.CO_AJI)]):
            yc = ys[2 + r]
            dr.text((bx0, yc), lab, font=F_PILL_B, fill=col, anchor='lm')
            L = max(150 * n / sc.N, 6)
            rrect(dr, (bx0 + 74, yc - 11, bx0 + 74 + L, yc + 11), 5, fill=col if r else (120, 170, 205))
            dr.text((bx0 + 82 + L, yc), f'{n}', font=F_PILL_B, fill=col, anchor='lm')
            dr.text((bx1, yc), f'AJI {aj:.3f}', font=F('segoeuib.ttf', 20), fill=col, anchor='rm')
        dr.text((bx0, ys[4]), f'{sc.N / sc.T:.0f}× fewer clicks · {100 * sc.CO_AJI / sc.PI_AJI:.0f}% of the AJI', font=F_PILL_B, fill=TEAL, anchor='lm')
        if sc.PI['match_clicks']:
            dr.text((bx0, ys[5]), f"w/o CoP matches CoP only after {sc.PI['match_clicks']} clicks", font=F_PILL, fill=MUTED, anchor='lm')
        y += rh + gap_r
    dr.text((W / 2, 938), V['summary_note'].format(**next((s.stats for s in SC if s.stats), {})), font=F_SUB, fill=MUTED, anchor='mm')
    return im


def by_iter(sc, t, max_frames=7):
    """Points that click t propagated to and that ended up as masks, grouped by FPR iteration into at most max_frames steps."""
    present = set(np.unique(sc.LAB).tolist())
    g, j = {}, 0
    for e in sc.EV:
        if e['ok']:
            j += 1
            if e['type'] == t and j in present:
                g.setdefault(e['iter'], []).append((t, e['x'], e['y']))
    its = sorted(g)
    step = max(1, math.ceil(len(its) / max_frames))
    return [sum((g[i] for i in its[s:s + step]), []) for s in range(0, len(its), step)]


frames = []
add = lambda im, ms: frames.append((im, ms))
for sc in SC:
    T = sc.T
    st = dict(clicks=[0, 0], aji=[None, None], k_pi=0, m={}, p=[], stars=[])
    add(frame(sc, st, 0, set()), 1200)
    for t, (cx, cy) in enumerate(sc.R['clicks']):                  # both sides advance together for the shared clicks
        st['stars'] = st['stars'] + [(t, cx, cy)]; st['clicks'] = [t + 1, t + 1]; st['k_pi'] = t + 1
        add(frame(sc, st, 0, set()), 800)
        for g in by_iter(sc, t):
            st['p'] = st['p'] + g; add(frame(sc, st, 1, {0}), 330)
        for a in (0.5, 1.0):
            st['m'] = {**st['m'], t: a}; add(frame(sc, st, 2, {0, 1}), 230)
        st['p'] = [q for q in st['p'] if q[0] != t]
    st['aji'] = [sc.SAME_AJI, sc.CO_AJI]
    nm = sc.R['clicks_only']['n_inst']
    st['msg'] = (f"Same {T} click{'s' if T > 1 else ''}:  w/o CoP AJI {sc.SAME_AJI:.3f} (only {nm} clicked cell{'s' if nm != 1 else ''})   →   w/ CoP AJI {sc.CO_AJI:.3f}", TEAL)
    add(frame(sc, st, 2, {0, 1}), 3200)
    st['msg'] = ('w/o CoP: a person keeps clicking every cell to recover the masks (exact AJI at every step)', PI_COL)
    marks = sorted(q for q in sc.CURVE if q > T)
    pick = [marks[round(i)] for i in np.linspace(0, len(marks) - 1, min(18, len(marks)))]
    for k in dict.fromkeys(pick):                                  # the left side keeps clicking, one cell at a time
        st['clicks'][0] = k; st['k_pi'] = k
        st['aji'][0] = sc.CURVE[max(q for q in sc.CURVE if q <= k)]
        if sc.PI['match_clicks'] and k >= sc.PI['match_clicks']:
            st['msg'] = (f"w/o CoP reaches CoP's AJI only after {sc.PI['match_clicks']} clicks", PI_COL)
        add(frame(sc, st, 2, {0, 1}), 260)
    st['aji'][0] = sc.PI_AJI; st['final'] = True
    st['msg'] = (f"{sc.N / T:.0f}× fewer clicks with CoP · {100 * sc.CO_AJI / sc.PI_AJI:.0f}% of the every-cell AJI"
                 + (f" · w/o CoP needs {sc.PI['match_clicks']} clicks to match" if sc.PI['match_clicks'] else ''), TEAL)
    add(frame(sc, st, 2, {0, 1}), 4000)
add(summary_frame(), 7000)

# GIF palette: the flat UI colors are reserved so text and logos stay exact; the rest of each frame is quantized.
_BASE = [(26, 45, 86), (0, 179, 133), TEAL, TEAL_BG, MINT_LINE, INK, MUTED, LINE, WHITE, (0, 0, 0), *TYPE_COL, PI_COL, GREY_BAR, TAB_DIM,
         (215, 238, 236), (244, 246, 247), (120, 170, 205)]
RESERVED = list(dict.fromkeys(_BASE + [tuple(round(c[i] + (255 - c[i]) * s) for i in range(3))
                                       for c in [(26, 45, 86), (0, 179, 133), TEAL] for s in (0.25, 0.5, 0.75)]))


def to_palette(f):
    k = len(RESERVED)
    q = f.quantize(colors=256 - k, method=Image.MEDIANCUT)
    p = (q.getpalette() or [])[:3 * (256 - k)]
    p = [v for c in RESERVED for v in c] + p + [0] * (3 * (256 - k) - len(p))
    pi = Image.new('P', (1, 1)); pi.putpalette(p)
    out = f.quantize(palette=pi, dither=Image.FLOYDSTEINBERG)
    idx, rgb = np.array(out), np.array(f.convert('RGB'))
    for j, col in enumerate(RESERVED):
        idx[(rgb == col).all(-1)] = j
    fixed = Image.frombytes('P', out.size, idx.astype(np.uint8).tobytes()); fixed.putpalette(p)
    return fixed


total_ms = sum(ms for _, ms in frames)
if args.preview:
    os.makedirs(args.preview, exist_ok=True)
    for i, (f, _) in enumerate(frames):
        f.save(os.path.join(args.preview, f'frame_{i:03d}.png'))
    print(f'{len(frames)} frames ({total_ms / 1000:.1f} s) written to {args.preview}')
else:
    before = open(args.out, 'rb').read() if os.path.exists(args.out) else None
    pal = [to_palette(f) for f, _ in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=[ms for _, ms in frames], loop=0, optimize=True, disposal=1)
    after = open(args.out, 'rb').read()
    state = 'new file' if before is None else 'identical to the previous file' if before == after else 'differs from the previous file'
    print(f'{args.out}: {len(frames)} frames, {total_ms / 1000:.1f} s, {len(after) / 1e6:.1f} MB ({state})')
