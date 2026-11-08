"""Contact sheet: sheet.py TMPDIR  (TMPDIR holds copies of A/ B/ C/). Writes TMPDIR/sheet.svg;
render it with rsvg-convert (images must sit next to the sheet)."""
import os, sys
T = sys.argv[1]
DK, LT = "#0b0b0b", "#fcfcfb"
o = []
def img(p, x, y, w, h): o.append(f'<image href="{p}" x="{x}" y="{y}" width="{w}" height="{h}"/>')
def tile(x, y, w, h, c): o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{c}" stroke="#c9c8c3"/>')
def txt(x, y, s, size=14, c="#52514e", w="400"):
    o.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" font-weight="{w}" fill="{c}">{s}</text>')
for v in "ABC":
    for b in ("mark-mono", "icon-mono", "lockup-mono", "emblem-mono"):
        p = f"{T}/{v}/{b}.svg"
        if os.path.exists(p):
            open(f"{T}/{v}/{b}-w.svg", "w").write(open(p).read().replace("#0b0b0b", "#fcfcfb"))
G = 24; X = G
# row 0: A emblem on light and on dark
txt(X, 34, "A · emblem on light", 18, "#0b0b0b", "700")
tile(X, 46, 860, 300, LT); img("A/emblem-light.svg", X + 10, 56, 840, 280)
txt(X + 860 + G, 34, "A · emblem: main logo (dark / light / mono)", 18, "#0b0b0b", "700")
x2 = X + 860 + G
tile(x2, 46, 860, 300, DK); img("A/emblem.svg", x2 + 10, 56, 840, 280)
y = 362
w3 = (860 * 2 + G - 2 * 12) / 3
for k, (f_, bg) in enumerate((("A/emblem-light.svg", LT), ("A/emblem-mono.svg", LT), ("A/emblem-mono-w.svg", DK))):
    tile(X + k * (w3 + 12), y, w3, 200, bg); img(f_, X + k * (w3 + 12) + 8, y + 8, w3 - 16, 184)
y += 230
NAMES = {"A": "A · main logo (header, splash)", "B": "B · badge (mid sizes, lockup)", "C": "C · icon (favicon, app icon)"}
CW = (860 * 2 + G - 2 * G) / 3
for i, v in enumerate("ABC"):
    x = X + i * (CW + G); yy = y
    txt(x, yy, NAMES[v], 17, "#0b0b0b", "700"); yy += 12
    mh = 210 if v != "C" else 210
    tile(x, yy, CW, mh, DK); img(f"{v}/mark.svg", x + 12, yy + 10, CW - 24, mh - 20); yy += mh + 8
    tile(x, yy, CW, mh, LT); img(f"{v}/mark-light.svg", x + 12, yy + 10, CW - 24, mh - 20); yy += mh + 8
    hw = (CW - 8) / 2
    tile(x, yy, hw, 110, LT); img(f"{v}/mark-mono.svg", x + 8, yy + 8, hw - 16, 94)
    tile(x + hw + 8, yy, hw, 110, DK); img(f"{v}/mark-mono-w.svg", x + hw + 16, yy + 8, hw - 16, 94); yy += 118
    for k, (bg, a, b, c) in enumerate(((DK, f"{v}/icon.svg", f"{v}/icon-mono-w.svg", f"{v}/icon.svg"), (LT, f"{v}/icon-light.svg", f"{v}/icon-mono.svg", f"{v}/icon-light.svg"))):
        tx = x + k * (hw + 8)
        tile(tx, yy, hw, 96, bg)
        img(a, tx + 12, yy + 16, 64, 64); img(a, tx + 90, yy + 32, 32, 32); img(b, tx + 134, yy + 32, 32, 32); img(c, tx + 180, yy + 40, 16, 16)
    yy += 104
    for f_, bg in ((f"{v}/lockup.svg", DK), (f"{v}/lockup-light.svg", LT), (f"{v}/lockup-mono.svg", LT)):
        tile(x, yy, CW, 72, bg); img(f_, x + 10, yy + 8, CW - 20, 56); yy += 80
H = yy + 34; W = X + 860 * 2 + G + G
txt(G, H - 12, "per variant: mark on dark · on light · mono black / white · icon 64, 32, mono 32, 16 px (actual size) on dark / light · lockup dark / light / mono", 13)
open(f"{T}/sheet.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}">'
    f'<rect width="100%" height="100%" fill="#efeeea"/>' + "".join(o) + "</svg>")
print(W, H)
