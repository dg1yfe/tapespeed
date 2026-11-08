#!/usr/bin/env python3
"""tapespeed logo family generator.

Usage:  python3 build.py OUTDIR      (writes OUTDIR/{A,B,C}/*.svg)
Needs:  shapely >= 2 (pip install shapely)

Shapes are composed from primitives here; every stroke is converted to a
filled outline (outline.py) before writing, so the SVGs contain fills only.
"""
import math, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline import outline_svg
from shapely.geometry import box, Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
from shapely.geometry.polygon import orient

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
OUTLINE = os.environ.get("TAPESPEED_LOGO_STROKES") != "1"   # set =1 to keep raw strokes (debug)

# ---------------------------------------------------------------- palette
BLACK = "#0b0b0b"
GREEN = "#5ef04a"      # phosphor
GREEN_GLOW = "#2fae2a"
GREEN_L = "#1a7f1f"    # on light backgrounds (5.0:1 on #fcfcfb)
CREAM = "#f3ead8"
SURF = "#fcfcfb"
STEEL1, STEEL2, STEEL3, STEEL4 = "#e9e8e3", "#b9b8b2", "#7d7c77", "#3a3a37"
SCREEN = "#04160a"

THEMES = {
    "dark":  dict(bg=BLACK, word1=CREAM, word2=GREEN, line=GREEN, steel_hi=STEEL1, steel=STEEL2,
                  steel_lo=STEEL3, steel_dk=STEEL4, dark="#141413", tick=GREEN, halo=True),
    "light": dict(bg=SURF, word1=BLACK, word2=GREEN_L, line=GREEN_L, steel_hi="#d6d5cf", steel="#9d9c96",
                  steel_lo="#5f5e5a", steel_dk="#2e2e2c", dark="#1b1b1a", tick=GREEN_L, halo=False),
}


def f(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- geometry -> svg path
def gpath(g, flip=True):
    """shapely (Multi)Polygon -> svg path d. y is flipped (geometry is y-up)."""
    polys = [g] if isinstance(g, Polygon) else list(getattr(g, "geoms", []))
    d = []
    for p in polys:
        if p.is_empty:
            continue
        p = orient(p, -1.0 if flip else 1.0)   # after the y-flip: CCW outer, CW holes
        for ring in [p.exterior] + list(p.interiors):
            cs = list(ring.coords)[:-1]
            d.append("M" + "L".join(f"{f(x)},{f(-y if flip else y)}" for x, y in cs) + "Z")
    return "".join(d)


def rrect(x, y, w, h, r):
    r = min(r, w / 2 - 0.01, h / 2 - 0.01)
    return box(x + r, y + r, x + w - r, y + h - r).buffer(r, quad_segs=16)


def soften(g, r=3):
    return g.buffer(-r, quad_segs=8).buffer(r, quad_segs=8)


# ---------------------------------------------------------------- heavy italic wordmark
# y-up units: baseline 0, x-height 100, ascender 140, descender -40
V, H, W = 31, 23, 112


def bowl(x0, w=W):
    return rrect(x0, 0, w, 100, 44).difference(rrect(x0 + V, H, w - 2 * V, 100 - 2 * H, 16))


def L_t(x0):
    g = unary_union([box(x0 + 10, 24, x0 + 10 + V, 134), box(x0, 72, x0 + 74, 100),
                     rrect(x0 + 10, 0, 64, 52, 24).difference(box(x0 + 10 + V, H, x0 + 200, 200))])
    return g, 76


def L_a(x0):
    return unary_union([bowl(x0), box(x0 + W - V, 0, x0 + W, 100)]), W


def L_p(x0):
    return unary_union([bowl(x0), box(x0, -42, x0 + V, 100)]), W


def L_d(x0):
    return unary_union([bowl(x0), box(x0 + W - V, 0, x0 + W, 140)]), W


def L_e(x0):
    g = rrect(x0, 0, W, 100, 44)
    g = g.difference(rrect(x0 + V, 58, W - 2 * V, 100 - H - 58, 9))
    g = g.difference(rrect(x0 + V, H - 1, W + 20, 42 - H + 1, 9))
    return g, W


def L_s(x0, w=100):
    g = rrect(x0, 0, w, 100, 42)
    g = g.difference(rrect(x0 + V - 2, 58, w + 20, 100 - H - 58, 9))
    g = g.difference(rrect(x0 - 20, H, w - V + 2 + 20, 42 - H, 9))
    return g, w


LETTERS = dict(t=L_t, a=L_a, p=L_p, e=L_e, s=L_s, d=L_d)


def wordmark_geoms(slant=14, gap=9):
    x = 0
    parts = []
    for i, ch in enumerate("tapespeed"):
        g, w = LETTERS[ch](x)
        g = soften(g, 2.5)
        g = affinity.skew(g, xs=slant, origin=(0, 0))
        parts.append((i < 4, g))
        x += w + gap
    a = unary_union([g for k, g in parts if k])
    b = unary_union([g for k, g in parts if not k])
    return a, b


def speed_lines(slant=14):
    """tapered horizontal streaks, left of the word (y-up units, x<0)."""
    gs = []
    for y, ln in ((92, 120), (72, 170), (52, 210), (32, 160), (12, 110)):
        x1 = -18 - (100 - y) * math.tan(math.radians(slant)) * 0 + y * math.tan(math.radians(slant)) - 8
        gs.append(Polygon([(x1 - ln, y - 1.2), (x1, y - 5.5), (x1, y + 5.5), (x1 - ln, y + 1.2)]))
    return unary_union(gs)


WM_A, WM_B = wordmark_geoms()
WM_BOUNDS = unary_union([WM_A, WM_B]).bounds  # (minx, miny, maxx, maxy) y-up


def wordmark_svg(x, base_y, xh, c1, c2, lines=None):
    """place wordmark: left edge x, baseline base_y, x-height xh (svg units)."""
    s = xh / 100
    minx = WM_BOUNDS[0]
    out = [f'<g transform="translate({f(x - minx * s)} {f(base_y)}) scale({s:.4f})">']
    if lines:
        out.append(f'<path d="{gpath(speed_lines())}" fill="{lines}"/>')
    out.append(f'<path d="{gpath(WM_A)}" fill="{c1}"/><path d="{gpath(WM_B)}" fill="{c2}"/></g>')
    width = (WM_BOUNDS[2] - minx) * s
    return "".join(out), width


# ---------------------------------------------------------------- small caps (monoline, buffered)
def _arc(cx, cy, rx, ry, a0, a1, n=24):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def caps(ch, sw=17):
    r = sw / 2
    L = lambda pts: LineString(pts).buffer(r, quad_segs=6)
    if ch == "S":
        pts = _arc(30, 75, 28, 25, 20, 270) + _arc(30, 25, 30, 25, 90, -160)[1:]
        return L(pts), 62
    if ch == "P":
        return unary_union([L([(0, 0), (0, 100)]), L([(0, 100), (30, 100)] + _arc(30, 75, 27, 25, 90, -90) + [(0, 50)])]), 60
    if ch == "E":
        return unary_union([L([(52, 100), (0, 100), (0, 0), (52, 0)]), L([(0, 52), (42, 52)])]), 55
    if ch == "D":
        return L([(0, 0), (0, 100), (22, 100)] + _arc(22, 50, 42, 50, 90, -90) + [(0, 0)]).union(
            Polygon([(0, 0), (0, 100), (22, 100), (22, 0)]).buffer(0).difference(box(r, r, 100, 100 - r))), 66
    if ch == "O":
        return L(_arc(34, 50, 34, 50, 0, 360, 48)), 70
    if ch == "K":
        return unary_union([L([(0, 0), (0, 100)]), L([(56, 100), (6, 44)]), L([(22, 60), (60, 0)])]), 62
    if ch == "v":  # check mark
        return L([(0, 48), (26, 18), (72, 88)]), 76
    if ch == " ":
        return Polygon(), 30
    raise KeyError(ch)


def caps_text(text, track=22):
    x, gs = 0, []
    for ch in text:
        g, w = caps(ch)
        gs.append(affinity.translate(g, x, 0))
        x += w + track
    return unary_union([g for g in gs if not g.is_empty]), x - track


def caps_arc(text, cx, cy, radius, height, center_deg=-90, track=22):
    """letters upright around an arc (svg coords, y down); returns svg path d."""
    s = height / 100
    widths = [caps(ch)[1] for ch in text]
    total = sum(widths) + track * (len(text) - 1)
    ang_total = math.degrees(total * s / radius)
    a = center_deg - ang_total / 2
    ds = []
    for ch, w in zip(text, widths):
        g, _ = caps(ch)
        mid = a + math.degrees((w * s / 2) / radius)
        g = affinity.translate(g, -w / 2, 0)
        g = affinity.scale(g, s, -s, origin=(0, 0))      # to svg (y down), baseline at 0
        g = affinity.rotate(g, mid + 90, origin=(0, 0))
        px, py = cx + radius * math.cos(math.radians(mid)), cy + radius * math.sin(math.radians(mid))
        g = affinity.translate(g, px, py)
        ds.append(gpath(g, flip=False))
        a += math.degrees((w + track) * s / radius)
    return "".join(ds)


# ---------------------------------------------------------------- spectrum
def spectrum_pts(x0, x1, ybase, ytop, n=220, noise=0.06, seed=7, peaks=((0.27, 1.0, 0.007), (0.62, 0.32, 0.006))):
    rnd = random.Random(seed)
    pts = []
    for i in range(n + 1):
        u = i / n
        v = noise * (0.55 + 0.45 * rnd.random()) * (1 if i % 2 else 0.55)
        for pu, ph, pw in peaks:
            v += ph / (1 + ((u - pu) / pw) ** 2)
            v += ph * 0.12 / (1 + ((u - pu) / (pw * 6)) ** 2)
        v = min(v, 1.0)
        pts.append((x0 + (x1 - x0) * u, ybase - (ybase - ytop) * v))
    return pts


def poly_d(pts, close=False):
    return "M" + "L".join(f"{f(x)},{f(y)}" for x, y in pts) + ("Z" if close else "")


# ---------------------------------------------------------------- reel
def reel_svg(cx, cy, r, fill, cut, hub, hub_ring, idp, cut_is_hole=False, round_holes=False):
    """three trapezoid windows; cut_is_hole -> knock-out via mask."""
    wins = []
    for k in range(3):
        a0 = -90 + k * 120 + 30
        if round_holes:
            rm = r * 0.52
            wins.append(gpath(Point(cx + rm * math.cos(math.radians(a0 - 30)), cy + rm * math.sin(math.radians(a0 - 30))).buffer(r * 0.27, quad_segs=12), flip=False))
            continue
        r1, r2, half = r * 0.36, r * 0.80, 27
        pts = [(cx + r1 * math.cos(math.radians(a0 + s * half * 0.7)), cy + r1 * math.sin(math.radians(a0 + s * half * 0.7))) for s in (-1, 1)]
        pts += [(cx + r2 * math.cos(math.radians(a)), cy + r2 * math.sin(math.radians(a))) for a in
                [a0 + half - i * (2 * half) / 12 for i in range(13)]]
        g = Polygon(pts).buffer(r * 0.05, quad_segs=6).buffer(-r * 0.0, quad_segs=6)
        wins.append(gpath(g, flip=False))
    wd = "".join(wins)
    if cut_is_hole:
        m = (f'<mask id="{idp}" maskUnits="userSpaceOnUse" x="{f(cx-r-2)}" y="{f(cy-r-2)}" width="{f(2*r+4)}" height="{f(2*r+4)}">'
             f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="#fff"/><path d="{wd}" fill="#000"/>'
             f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r*(0.13 if round_holes else 0.12))}" fill="#000"/></mask>')
        return m, f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}" mask="url(#{idp})"/>'
    s = [f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}"/>',
         f'<path d="{wd}" fill="{cut}"/>',
         f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r*0.2)}" fill="{hub}"/>',
         f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r*0.1)}" fill="{hub_ring}"/>']
    return "", "".join(s)


# ====================================================================
# MARK (deck + scope + knob + loop). mode: dark | light | mono ; style: A | B
# layout in a 2000-wide space; loop_y = bottom run of the cable
# ====================================================================
def mark(mode, style, idp, loop_y=455, loop_x=230, with_label=True):
    th = THEMES["light" if mode == "light" else "dark"]
    mono = BLACK if mode == "mono" else None
    defs, o = [], []
    A = style == "A"
    line = mono or th["line"]

    if A and not mono:
        defs.append(f'<linearGradient id="{idp}st" x1="0" y1="0" x2="0" y2="1">'
                    f'<stop offset="0" stop-color="{th["steel_hi"]}"/><stop offset="1" stop-color="{th["steel"]}"/></linearGradient>')
        defs.append(f'<linearGradient id="{idp}rl" x1="0" y1="0" x2="1" y2="1">'
                    f'<stop offset="0" stop-color="#f6f5f0"/><stop offset="1" stop-color="#a8a7a1"/></linearGradient>')
        defs.append(f'<linearGradient id="{idp}bz" x1="0" y1="0" x2="0" y2="1">'
                    f'<stop offset="0" stop-color="#cfcec8"/><stop offset="1" stop-color="#6c6b66"/></linearGradient>')
        defs.append(f'<radialGradient id="{idp}sc" cx="0.5" cy="0.55" r="0.65">'
                    f'<stop offset="0" stop-color="#0e3d17"/><stop offset="1" stop-color="{SCREEN}"/></radialGradient>')
        defs.append(f'<linearGradient id="{idp}sp" x1="0" y1="0" x2="0" y2="1">'
                    f'<stop offset="0" stop-color="{GREEN}" stop-opacity="0.95"/><stop offset="1" stop-color="{GREEN}" stop-opacity="0.35"/></linearGradient>')
        defs.append(f'<radialGradient id="{idp}kn" cx="0.35" cy="0.3" r="0.8">'
                    f'<stop offset="0" stop-color="#6a6a66"/><stop offset="1" stop-color="#101010"/></radialGradient>')
        defs.append(f'<filter id="{idp}gl" x="-20%" y="-20%" width="140%" height="140%">'
                    f'<feGaussianBlur stdDeviation="7"/><feComponentTransfer><feFuncA type="linear" slope="0.9"/></feComponentTransfer>'
                    f'<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    glow = f' filter="url(#{idp}gl)"' if (A and not mono and mode == "dark") else ""

    if not A:
        if mono:
            o.append(f'<rect x="215" y="25" width="1735" height="445" rx="64" fill="none" stroke="{mono}" stroke-width="14"/>')
        elif mode == "dark":
            o.append(f'<rect x="215" y="25" width="1735" height="445" rx="64" fill="#121312" stroke="{STEEL3}" stroke-width="6"/>')
        else:
            o.append(f'<rect x="215" y="25" width="1735" height="445" rx="64" fill="none" stroke="{GREEN_L}" stroke-width="10"/>')
    # ---- cable (drawn first, under everything)
    cw = 15 if A else 17
    knob_c = (1795, 330)
    p_deck_in = (330, 250)
    cable2 = (f"M{knob_c[0]},{knob_c[1]+112}V{loop_y-40}Q{knob_c[0]},{loop_y} {knob_c[0]-40},{loop_y}"
              f"H{loop_x+40}Q{loop_x},{loop_y} {loop_x},{loop_y-40}V{p_deck_in[1]+40}"
              f"Q{loop_x},{p_deck_in[1]} {loop_x+40},{p_deck_in[1]}H{p_deck_in[0]-34}")
    cable1 = f"M1712,165H1800Q1890,165 1890,262"
    arrow = f"M{p_deck_in[0]-52},{p_deck_in[1]-26}L{p_deck_in[0]-18},{p_deck_in[1]}L{p_deck_in[0]-52},{p_deck_in[1]+26}Z"
    if not A and not mono and th["halo"]:
        for d in (cable1, cable2):
            o.append(f'<path d="{d}" fill="none" stroke="{GREEN}" stroke-opacity="0.28" stroke-width="{cw*2.6}" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(f'<g{glow}><path d="{cable1}" fill="none" stroke="{line}" stroke-width="{cw}" stroke-linecap="round" stroke-linejoin="round"/>'
             f'<path d="{cable2}" fill="none" stroke="{line}" stroke-width="{cw}" stroke-linecap="round" stroke-linejoin="round"/>'
             f'<path d="{arrow}" fill="{line}" stroke="{line}" stroke-width="8" stroke-linejoin="round"/></g>')

    # ---- deck
    if mono:
        o.append(f'<rect x="340" y="70" width="570" height="300" rx="22" fill="none" stroke="{mono}" stroke-width="16"/>')
    else:
        body = f"url(#{idp}st)" if A else th["steel"]
        o.append(f'<rect x="340" y="70" width="570" height="300" rx="22" fill="{body}" stroke="{th["steel_lo"]}" stroke-width="4"/>')
    # lower transport panel
    if mono:
        o.append(f'<rect x="545" y="290" width="160" height="50" rx="8" fill="{mono}"/>')
        for x in (500, 750):
            o.append(f'<circle cx="{x}" cy="318" r="20" fill="{mono}"/>')
    else:
        if A:
            o.append(f'<rect x="352" y="282" width="546" height="78" rx="12" fill="{th["steel_lo"]}" fill-opacity="0.35"/>')
        o.append(f'<rect x="545" y="292" width="160" height="48" rx="8" fill="{th["steel_dk"]}"/>')
        if A:
            o.append(f'<rect x="560" y="302" width="130" height="14" rx="4" fill="{th["steel_lo"]}"/>')
            for x in (380, 425, 825, 870):
                o.append(f'<circle cx="{x}" cy="318" r="17" fill="{th["dark"]}" stroke="{th["steel_lo"]}" stroke-width="4"/>')
        for x in (500, 750):
            o.append(f'<circle cx="{x}" cy="318" r="21" fill="{f"url(#{idp}rl)" if A else th["steel_hi"]}" stroke="{th["steel_lo"]}" stroke-width="3"/>')
        # tape path
        o.append(f'<path d="M386,215L500,297M864,215L750,297" stroke="{th["steel_dk"]}" stroke-width="5"/>')
    rr, rcy = (138, 165) if A else (124, 176)
    for i, cx in enumerate((480, 770)):
        if mono:
            m, s = reel_svg(cx, rcy, rr, mono, None, None, None, f"{idp}r{i}", cut_is_hole=True, round_holes=not A)
            defs.append(m); o.append(s)
        else:
            fill = f"url(#{idp}rl)" if A else th["steel_hi"]
            m, s = reel_svg(cx, rcy, rr, fill, th["dark"], th["dark"], th["steel"], f"{idp}r{i}", round_holes=not A)
            o.append(s)
            if A:
                o.append(f'<circle cx="{cx}" cy="165" r="138" fill="none" stroke="{th["steel_lo"]}" stroke-width="3"/>')
    # ---- scope
    sx0, sy0, sx1, sy1 = 950, 40, 1700, 370
    ix0, iy0, ix1, iy1 = sx0 + 34, sy0 + 34, sx1 - 34, sy1 - 34
    if mono:
        o.append(f'<rect x="{sx0}" y="{sy0}" width="{sx1-sx0}" height="{sy1-sy0}" rx="40" fill="none" stroke="{mono}" stroke-width="16"/>')
        o.append(f'<rect x="{ix0}" y="{iy0}" width="{ix1-ix0}" height="{iy1-iy0}" rx="20" fill="none" stroke="{mono}" stroke-width="7"/>')
    else:
        bz = f"url(#{idp}bz)" if A else th["steel_lo"]
        o.append(f'<rect x="{sx0}" y="{sy0}" width="{sx1-sx0}" height="{sy1-sy0}" rx="40" fill="{bz}"/>')
        scr = f"url(#{idp}sc)" if A else SCREEN
        o.append(f'<rect x="{ix0}" y="{iy0}" width="{ix1-ix0}" height="{iy1-iy0}" rx="20" fill="{scr}"/>')
    # plug on the scope's right side
    o.append(f'<rect x="1694" y="148" width="26" height="34" rx="5" fill="{mono or (GREEN if mode=="dark" else GREEN_L)}"/>')
    # grid (A only)
    if A and not mono:
        gl = []
        for k in range(1, 12):
            x = ix0 + (ix1 - ix0) * k / 12
            gl.append(f"M{f(x)},{iy0+6}V{iy1-6}")
        for k in range(1, 6):
            y = iy0 + (iy1 - iy0) * k / 6
            gl.append(f"M{ix0+6},{f(y)}H{ix1-6}")
        o.append(f'<path d="{"".join(gl)}" stroke="{GREEN}" stroke-opacity="0.22" stroke-width="2.5"/>')
    # spectrum
    base, top = iy1 - 16, iy0 + 22
    if A:
        pts = spectrum_pts(ix0 + 18, ix1 - 18, base, top)
        trace_col = mono or GREEN
        if mono:
            o.append(f'<path d="{poly_d(pts)}" fill="none" stroke="{mono}" stroke-width="6" stroke-linejoin="round"/>')
        else:
            o.append(f'<path d="{poly_d([(pts[0][0], base)] + pts + [(pts[-1][0], base)], True)}" fill="url(#{idp}sp)"/>')
            o.append(f'<path d="{poly_d(pts)}" fill="none" stroke="{GREEN}" stroke-width="4" stroke-linejoin="round"{glow}/>')
    else:
        pts = spectrum_pts(ix0 + 24, ix1 - 24, base, top, n=60, noise=0.035, peaks=((0.27, 1.0, 0.014), (0.62, 0.34, 0.012)))
        o.append(f'<path d="{poly_d(pts)}" fill="none" stroke="{mono or GREEN}" stroke-width="13" stroke-linejoin="round" stroke-linecap="round"/>')
    # "✓ SPEED OK" label (A only, colour)
    if A and with_label and not mono:
        g, w = caps_text("v SPEED OK")
        s = 30 / 100
        lx = ix1 - 30 - w * s
        o.append(f'<rect x="{f(lx-14)}" y="{iy0+16}" width="{f(w*s+28)}" height="54" rx="6" fill="{GREEN}" fill-opacity="0.12"/>')
        o.append(f'<g transform="translate({f(lx)} {iy0+58}) scale({s})"><path d="{gpath(g)}" fill="{GREEN}"/></g>')
    # ---- knob
    kx, ky = knob_c
    if mono:
        o.append(f'<circle cx="{kx}" cy="{ky}" r="112" fill="none" stroke="{mono}" stroke-width="14"/>')
        o.append(f'<circle cx="{kx}" cy="{ky}" r="54" fill="none" stroke="{mono}" stroke-width="12"/>')
        o.append(f'<path d="M{kx},{ky}L{f(kx+40*math.cos(math.radians(-50)))},{f(ky+40*math.sin(math.radians(-50)))}" stroke="{mono}" stroke-width="12" stroke-linecap="round"/>')
    else:
        o.append(f'<circle cx="{kx}" cy="{ky}" r="112" fill="{th["dark"] if mode=="dark" else "#1b1b1a"}" stroke="{th["steel_lo"]}" stroke-width="5"/>')
        kfill = f"url(#{idp}kn)" if A else "#2b2b29"
        o.append(f'<circle cx="{kx}" cy="{ky}" r="58" fill="{kfill}" stroke="{th["steel"]}" stroke-width="6"/>')
        o.append(f'<path d="M{kx},{ky}L{f(kx+44*math.cos(math.radians(-50)))},{f(ky+44*math.sin(math.radians(-50)))}" stroke="{STEEL1}" stroke-width="9" stroke-linecap="round"/>')
    # ticks
    tc = mono or (GREEN if mode == "dark" or mode == "light" else GREEN)
    if mode == "light":
        tc = GREEN  # ticks sit on the dark dial
    tk = []
    for k in range(7 if A else 5):
        a = math.radians(-210 + k * (240 / ((7 if A else 5) - 1)))
        if A and with_label and -150 < math.degrees(a) < -30:
            continue
        r1, r2 = 72, 96
        tk.append(f"M{f(kx+r1*math.cos(a))},{f(ky+r1*math.sin(a))}L{f(kx+r2*math.cos(a))},{f(ky+r2*math.sin(a))}")
    if mono:
        tk = []
    o.append(f'<path d="{"".join(tk)}" stroke="{tc}" stroke-width="8" stroke-linecap="round"/>')
    if A and with_label and not mono:
        o.append(f'<path d="{caps_arc("SPEED", kx, ky, 80, 22)}" fill="{GREEN}"/>')
    return "<defs>" + "".join(defs) + "</defs>" + "".join(o)


MARK_VB = (200, 20, 1760, 470)   # x, y, w, h for the mark alone (loop_y=455)


# ====================================================================
# C — minimal mark: reel inside the feedback loop; the loop's top is the peak
# ====================================================================
def c_mark(mode, idp, size=64, icon=False):
    th = THEMES["light" if mode == "light" else "dark"]
    mono = BLACK if mode == "mono" else None
    line = mono or th["line"]
    reel_c = mono or (CREAM if mode == "dark" else STEEL4)
    cut = th["bg"]
    defs, o = [], []
    cx, cy, R = 32, 36, 23        # loop radius (stroke centre)
    sw = 5.2
    # loop: circle arc from ~ -60° (just right of the peak) clockwise to ~ -125° (left of peak)
    # with a peak spike at top centre, and an arrowhead at the end pointing into the gap
    a_start, a_end = -62, 242   # degrees, clockwise in svg
    pts = []
    n = 60
    for i in range(n + 1):
        a = math.radians(a_start + (a_end - a_start) * i / n)
        pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
    d = "M" + "L".join(f"{f(x)},{f(y)}" for x, y in pts)
    # the peak on top: from the end of the arc (left of top) up the spike and down onto the start point
    xs, ys = pts[-1]
    x0, y0 = pts[0]
    peak = f"M{f(xs)},{f(ys)}L{f(cx-5.6)},{f(ys)}L{f(cx)},{f(cy-R-12)}L{f(cx+5.6)},{f(y0)}L{f(x0)},{f(y0)}"
    # arrowhead on the loop pointing clockwise, placed at ~ 120° (lower left)
    o.append(f'<path d="{d}" fill="none" stroke="{line}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(f'<path d="{peak}" fill="none" stroke="{line}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')
    # arrowhead at ~ 40° (lower right), pointing along the clockwise tangent
    aa = math.radians(35)
    ax, ay = cx + R * math.cos(aa), cy + R * math.sin(aa)
    tx, ty = -math.sin(aa), math.cos(aa)      # clockwise tangent in svg coords
    nx, ny = math.cos(aa), math.sin(aa)
    L_, Wd = 7.5, 6.5
    tip = (ax + tx * L_ * 0.6, ay + ty * L_ * 0.6)
    b1 = (ax - tx * L_ * 0.5 + nx * Wd, ay - ty * L_ * 0.5 + ny * Wd)
    b2 = (ax - tx * L_ * 0.5 - nx * Wd, ay - ty * L_ * 0.5 - ny * Wd)
    o.append(f'<path d="M{f(b1[0])},{f(b1[1])}L{f(tip[0])},{f(tip[1])}L{f(b2[0])},{f(b2[1])}Z" fill="{line}" stroke="{line}" stroke-width="2.4" stroke-linejoin="round"/>')
    # reel in the middle (knock-out windows)
    m, s = reel_svg(cx, cy, 14.5, reel_c, None, None, None, f"{idp}r", cut_is_hole=True, round_holes=True)
    defs.append(m); o.append(s)
    # fit the loop (peak tip to arrow) into the 64 grid with ~2 units margin
    return "<defs>" + "".join(defs) + "</defs>" + '<g transform="translate(32 32) scale(0.95) translate(-33 -30)">' + "".join(o) + "</g>"


# ====================================================================
# icons
# ====================================================================
def icon_A(mode, idp):
    """CRT with glowing peak + SPEED knob (detailed)."""
    th = THEMES["light" if mode == "light" else "dark"]
    mono = BLACK if mode == "mono" else None
    o, defs = [], []
    if mono:
        o.append(f'<mask id="{idp}k" maskUnits="userSpaceOnUse" x="0" y="0" width="64" height="64"><rect width="64" height="64" fill="#fff"/><circle cx="51" cy="51" r="14.5" fill="#000"/></mask>')
        o.append(f'<g mask="url(#{idp}k)"><rect x="3.5" y="7.5" width="57" height="45" rx="9" fill="none" stroke="#0b0b0b" stroke-width="5"/>'
                 '<path d="M10,44H20L24.5,15L29,44H38L40.5,33L43,44H54" fill="none" stroke="#0b0b0b" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/></g>')
        o.append('<circle cx="51" cy="51" r="12" fill="#fcfcfb" fill-opacity="0"/><circle cx="51" cy="51" r="8.5" fill="none" stroke="#0b0b0b" stroke-width="4"/>')
        o.append('<path d="M51,51L55,46" stroke="#0b0b0b" stroke-width="3.4" stroke-linecap="round"/>')
        return "".join(o)
    defs.append(f'<linearGradient id="{idp}bz" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d9d8d2"/><stop offset="1" stop-color="#6c6b66"/></linearGradient>')
    defs.append(f'<radialGradient id="{idp}sc" cx="0.5" cy="0.6" r="0.7"><stop offset="0" stop-color="#0f4a1b"/><stop offset="1" stop-color="{SCREEN}"/></radialGradient>')
    defs.append(f'<filter id="{idp}gl" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="1.2"/>'
                f'<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    o.append(f'<rect x="2" y="6" width="60" height="48" rx="10" fill="url(#{idp}bz)"/>')
    o.append(f'<rect x="7" y="11" width="50" height="38" rx="6" fill="url(#{idp}sc)"/>')
    o.append(f'<path d="M19.5,12V48M32,12V48M44.5,12V48M8,30H56" stroke="{GREEN}" stroke-opacity="0.22" stroke-width="1"/>')
    o.append(f'<path d="M10,44H20L24.5,15L29,44H38L40.5,33L43,44H54" fill="none" stroke="{GREEN}" stroke-width="3.6" '
             f'stroke-linejoin="round" stroke-linecap="round" filter="url(#{idp}gl)"/>')
    o.append(f'<circle cx="51" cy="51" r="11.5" fill="#141413" stroke="#7d7c77" stroke-width="1.5"/>')
    o.append(f'<circle cx="51" cy="51" r="6.5" fill="#3a3a37"/>')
    o.append(f'<path d="M51,51L55.5,45.5" stroke="{STEEL1}" stroke-width="2.6" stroke-linecap="round"/>')
    o.append(f'<path d="M42.5,56.5L41,58M59.5,56.5L61,58" stroke="{GREEN}" stroke-width="2" stroke-linecap="round"/>')
    return "<defs>" + "".join(defs) + "</defs>" + "".join(o)


def icon_B(mode, idp):
    """flat: reel + scope peak + green loop with knob."""
    th = THEMES["light" if mode == "light" else "dark"]
    mono = BLACK if mode == "mono" else None
    line = mono or th["line"]
    o, defs = [], []
    # scope (top-right)
    if mono:
        o.append(f'<rect x="27" y="4" width="33" height="27" rx="5" fill="none" stroke="{mono}" stroke-width="4"/>')
        tr = mono
    else:
        o.append(f'<rect x="27" y="4" width="33" height="27" rx="6" fill="{th["steel_lo"]}"/>')
        o.append(f'<rect x="30.5" y="7.5" width="26" height="20" rx="3.5" fill="{SCREEN}"/>')
        tr = GREEN
    o.append(f'<path d="M33,23.5H39.5L43.5,10.5L47.5,23.5H54" fill="none" stroke="{tr}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>')
    # reel (left)
    reel_c = mono or (th["steel_hi"] if mode == "dark" else STEEL4)
    m, s = reel_svg(17, 21, 14, reel_c, None, None, None, f"{idp}r", cut_is_hole=True, round_holes=True)
    defs.append(m); o.append(s)
    # loop: scope bottom -> down -> knob -> left -> up into reel, arrow
    if not mono and th["halo"]:
        o.append(f'<path d="M43.5,31V50M43.5,50H17V40" fill="none" stroke="{GREEN}" stroke-opacity="0.28" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(f'<path d="M43.5,31V44M38,52H22Q17,52 17,47V40" fill="none" stroke="{line}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(f'<path d="M11.5,43L17,36.5L22.5,43Z" fill="{line}" stroke="{line}" stroke-width="2" stroke-linejoin="round"/>')
    # knob
    if mono:
        o.append(f'<circle cx="45" cy="51" r="8" fill="none" stroke="{mono}" stroke-width="4"/><path d="M45,51L48.5,47.5" stroke="{mono}" stroke-width="3.4" stroke-linecap="round"/>')
    else:
        o.append(f'<circle cx="45" cy="51" r="9.5" fill="#2b2b29" stroke="{line}" stroke-width="3"/>')
        o.append(f'<path d="M45,51L49,47" stroke="{STEEL1}" stroke-width="2.6" stroke-linecap="round"/>')
    return "<defs>" + "".join(defs) + "</defs>" + "".join(o)


# ====================================================================
# assembly
# ====================================================================
def doc(content, vb, title, bg=None, scale=1):
    x, y, w, h = vb
    b = f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="{bg}"/>' if bg else ""
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{f(x)} {f(y)} {f(w)} {f(h)}" '
         f'width="{f(w*scale)}" height="{f(h*scale)}" role="img" aria-label="{title}">\n'
         f'<title>{title}</title>\n{b}{content}\n</svg>\n')
    return outline_svg(s, nd=2) if OUTLINE else s


def write(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(s)


def word_cols(mode):
    if mode == "mono":
        return BLACK, BLACK, BLACK
    th = THEMES[mode]
    return th["word1"], th["word2"], th["line"]


def lockup(mark_svg, mvb, mode, gap=None, lines=True):
    """mark scaled to height Hm, wordmark right of it, x-height ~ 0.5 Hm."""
    x, y, w, h = mvb
    Hm = 120.0
    s = Hm / h
    mw = w * s
    pad = 10
    c1, c2, cl = word_cols(mode)
    xh = 52
    gap = gap if gap is not None else 26
    base = pad + Hm / 2 + xh * 0.62
    res = (WM_BOUNDS[0] - speed_lines().bounds[0]) * xh / 100 if lines else 0
    ws, ww = wordmark_svg(pad + mw + gap + res, base, xh, c1, c2, lines=cl if lines else None)
    body = (f'<g transform="translate({f(pad)} {f(pad)}) scale({s:.5f}) translate({f(-x)} {f(-y)})">{mark_svg}</g>' + ws)
    return body, (0, 0, pad + mw + gap + res + ww + pad, Hm + 2 * pad)


def emblem(mode, idp):
    """the emblem's composition: mark above, wordmark below, loop around everything."""
    c1, c2, cl = word_cols(mode)
    m = mark(mode, "A", idp, loop_y=672, loop_x=110)
    ws, ww = wordmark_svg(470, 592, 112, c1, c2, lines=cl)
    return m + ws, (60, 10, 1900, 700)


def main():
    for mode in ("dark", "light", "mono"):
        sfx = {"dark": "", "light": "-light", "mono": "-mono"}[mode]
        # ---- A
        write(f"{OUT}/A/mark{sfx}.svg", doc(mark(mode, "A", f"a{mode}m"), MARK_VB, "tapespeed set 2 A: mark", scale=0.25))
        c, vb = emblem(mode, f"a{mode}e")
        write(f"{OUT}/A/emblem{sfx}.svg", doc(c, vb, "tapespeed set 2 A: emblem", scale=0.5))
        write(f"{OUT}/A/icon{sfx}.svg", doc(icon_A(mode, f"a{mode}i"), (0, 0, 64, 64), "tapespeed set 2 A: icon", scale=2))
        c, vb = lockup(mark(mode, "A", f"a{mode}l", with_label=False), MARK_VB, mode)
        write(f"{OUT}/A/lockup{sfx}.svg", doc(c, vb, "tapespeed"))
        # ---- B
        write(f"{OUT}/B/mark{sfx}.svg", doc(mark(mode, "B", f"b{mode}m", loop_y=425, loop_x=275), MARK_VB, "tapespeed set 2 B: badge", scale=0.25))
        write(f"{OUT}/B/icon{sfx}.svg", doc(icon_B(mode, f"b{mode}i"), (0, 0, 64, 64), "tapespeed set 2 B: icon", scale=2))
        c, vb = lockup(mark(mode, "B", f"b{mode}l", loop_y=425, loop_x=275), MARK_VB, mode)
        write(f"{OUT}/B/lockup{sfx}.svg", doc(c, vb, "tapespeed"))
        # ---- C
        write(f"{OUT}/C/mark{sfx}.svg", doc(c_mark(mode, f"c{mode}m"), (0, 0, 64, 64), "tapespeed set 2 C: mark", scale=4))
        write(f"{OUT}/C/icon{sfx}.svg", doc(c_mark(mode, f"c{mode}i", icon=True), (0, 0, 64, 64), "tapespeed set 2 C: icon", scale=2))
        c, vb = lockup(c_mark(mode, f"c{mode}l"), (0, 0, 64, 64), mode, gap=14)
        write(f"{OUT}/C/lockup{sfx}.svg", doc(c, vb, "tapespeed"))


if __name__ == "__main__":
    main()
