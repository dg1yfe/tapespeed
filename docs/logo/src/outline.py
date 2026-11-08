"""Convert SVG strokes to filled outlines (shapely). Handles the subset the
logo generator emits: <path> (M L H V Q C Z, absolute), <circle>, <rect>."""
import re
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, LinearRing, Polygon, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
STROKE_ATTRS = ("stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-opacity", "stroke-miterlimit")


def _num(s):
    return float(s)


def parse_path(d, qseg=16):
    """-> list of (points, closed)."""
    toks = re.findall(r"[MLHVQCZmlhvqcz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    subs, pts, i, cmd = [], [], 0, None
    cur = (0.0, 0.0)
    start = cur

    def take(n):
        nonlocal i
        v = [float(t) for t in toks[i:i + n]]
        i += n
        return v
    while i < len(toks):
        t = toks[i]
        if re.match(r"[A-Za-z]", t):
            cmd = t
            i += 1
            if cmd in "Zz":
                if pts:
                    subs.append((pts, True))
                pts = []
                cur = start
                continue
        if cmd not in "MLHVQC":
            raise ValueError(f"unsupported path command {cmd!r}")
        if cmd == "M":
            if pts:
                subs.append((pts, False))
            cur = tuple(take(2)); start = cur; pts = [cur]; cmd = "L"
        elif cmd == "L":
            cur = tuple(take(2)); pts.append(cur)
        elif cmd == "H":
            cur = (take(1)[0], cur[1]); pts.append(cur)
        elif cmd == "V":
            cur = (cur[0], take(1)[0]); pts.append(cur)
        elif cmd == "Q":
            x1, y1, x, y = take(4)
            x0, y0 = cur
            for k in range(1, qseg + 1):
                u = k / qseg
                pts.append(((1-u)**2*x0 + 2*(1-u)*u*x1 + u*u*x, (1-u)**2*y0 + 2*(1-u)*u*y1 + u*u*y))
            cur = (x, y)
        elif cmd == "C":
            x1, y1, x2, y2, x, y = take(6)
            x0, y0 = cur
            for k in range(1, qseg + 1):
                u = k / qseg
                a, b, c, e = (1-u)**3, 3*(1-u)**2*u, 3*(1-u)*u*u, u**3
                pts.append((a*x0 + b*x1 + c*x2 + e*x, a*y0 + b*y1 + c*y2 + e*y))
            cur = (x, y)
    if pts:
        subs.append((pts, False))
    return subs


def _dedupe(pts):
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def _rrect(x, y, w, h, r):
    if w <= 0 or h <= 0:
        return Polygon()
    r = max(0.0, min(r, w / 2 - 1e-6, h / 2 - 1e-6))
    if r <= 0:
        return box(x, y, x + w, y + h)
    return box(x + r, y + r, x + w - r, y + h - r).buffer(r, quad_segs=16)


def geom_to_d(g, nd=2, tol=0.0):
    def fmt(v):
        t = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return "0" if t in ("", "-0") else t
    if tol > 0:
        g = g.simplify(tol, preserve_topology=True)
    polys = [g] if isinstance(g, Polygon) else list(getattr(g, "geoms", []))
    out = []
    for p in polys:
        if p.is_empty or not isinstance(p, Polygon):
            continue
        p = orient(p, 1.0)
        for ring in [p.exterior] + list(p.interiors):
            cs = list(ring.coords)[:-1]
            out.append("M" + "L".join(f"{fmt(x)},{fmt(y)}" for x, y in cs) + "Z")
    return "".join(out)


def stroke_geom(el):
    tag = el.tag.split("}")[-1]
    w = float(el.get("stroke-width", "1"))
    cap = {"round": 1, "square": 3}.get(el.get("stroke-linecap", "butt"), 2)
    join = {"round": 1, "bevel": 3}.get(el.get("stroke-linejoin", "miter"), 2)
    ml = float(el.get("stroke-miterlimit", "4"))
    if tag == "circle":
        cx, cy, r = (float(el.get(k)) for k in ("cx", "cy", "r"))
        outer = Point(cx, cy).buffer(r + w / 2, quad_segs=32)
        inner = Point(cx, cy).buffer(r - w / 2, quad_segs=32) if r > w / 2 else Polygon()
        return outer.difference(inner), Point(cx, cy).buffer(r, quad_segs=32)
    if tag == "rect":
        x, y = float(el.get("x", 0)), float(el.get("y", 0))
        W, H = float(el.get("width")), float(el.get("height"))
        rx = float(el.get("rx", el.get("ry", 0)))
        outer = _rrect(x - w / 2, y - w / 2, W + w, H + w, rx + w / 2 if rx else 0)
        inner = _rrect(x + w / 2, y + w / 2, W - w, H - w, max(rx - w / 2, 0) if rx else 0)
        return outer.difference(inner), _rrect(x, y, W, H, rx)
    if tag == "path":
        parts, fills = [], []
        for pts, closed in parse_path(el.get("d")):
            pts = _dedupe(pts)
            if closed and len(pts) >= 3:
                ring = LinearRing(pts)
                parts.append(ring.buffer(w / 2, cap_style=cap, join_style=join, mitre_limit=ml, quad_segs=12))
                fills.append(Polygon(pts).buffer(0))
            elif len(pts) >= 2:
                parts.append(LineString(pts).buffer(w / 2, cap_style=cap, join_style=join, mitre_limit=ml, quad_segs=12))
                if len(pts) >= 3:
                    fills.append(Polygon(pts).buffer(0))
            elif len(pts) == 1 and cap == 1:
                parts.append(Point(pts[0]).buffer(w / 2, quad_segs=12))
        return unary_union(parts), unary_union(fills) if fills else Polygon()
    raise ValueError(f"stroke on <{tag}> not supported")


def outline_tree(root, nd=2, tol=0.0):
    n = 0
    for parent in list(root.iter()):
        for idx, el in enumerate(list(parent)):
            s = el.get("stroke")
            if not s or s == "none" or float(el.get("stroke-width", "1")) <= 0:
                for a in STROKE_ATTRS:
                    if a in el.attrib and s in (None, "none"):
                        del el.attrib[a]
                continue
            sg, fg = stroke_geom(el)
            fill = el.get("fill")
            tag = el.tag.split("}")[-1]
            if fill is None and tag == "path" and fg.area > 1e-6:
                fill = "#000"   # SVG default fill
            keep = {k: v for k, v in el.attrib.items() if k not in STROKE_ATTRS and k not in ("fill", "fill-opacity", "d", "cx", "cy", "r", "x", "y", "width", "height", "rx", "ry")}
            so = el.get("stroke-opacity")
            pos = list(parent).index(el)
            parent.remove(el)
            new = []
            same = fill == s and el.get("fill-opacity") == so
            if fill and fill != "none" and not same:
                fel = ET.Element(el.tag, {k: v for k, v in el.attrib.items() if k not in STROKE_ATTRS})
                new.append(fel)
            g = unary_union([sg, fg]) if same else sg
            if not g.is_empty:
                attrs = dict(keep)
                attrs["d"] = geom_to_d(g, nd, tol)
                attrs["fill"] = s
                if so is not None:
                    attrs["fill-opacity"] = so
                new.append(ET.Element(f"{{{NS}}}path", attrs))
            for k, e in enumerate(new):
                parent.insert(pos + k, e)
            n += 1
    return n


def outline_svg(text, nd=2):
    root = ET.fromstring(text)
    vb = [float(v) for v in root.get("viewBox").split()]
    tol = max(vb[2], vb[3]) / 8000.0          # ~0.06 px at 512 px output
    outline_tree(root, nd, tol)
    return ET.tostring(root, encoding="unicode") + "\n"
