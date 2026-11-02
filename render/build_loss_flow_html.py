"""Build vis_CBD/vis_loss_flow/loss_flow.html: train_CBD (version 5, NGBC) on a white
background, serif (Times-like) font, two inline-SVG panels:
  (a) one-step loss flow with real inputs / W / cage / outputs of one sample
      (assets from save_loss_flow_assets.py)
  (b) the shared LinearEncoder architecture of phi / g / psi + per-module table

Active losses are the ckpts_CBD9 setting only (in_type 1, out_type 1, relu,
align_latent, no optional losses): recon-def, recon-neu, exp-z.

Run from repo root (after render/save_loss_flow_assets.py):
    python render/build_loss_flow_html.py
"""
import base64
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_loss_flow"
ASSETS = OUT_ROOT / "assets"

VAL = json.load(open(ASSETS / "values.json"))
N, M, L = VAL["N"], VAL["M"], VAL["L"]

FONT = "'Times New Roman', Times, 'Liberation Serif', 'Nimbus Roman', serif"
C_NET, C_NET_BG = "#2b5d9b", "#edf3fa"
C_LOSS, C_LOSS_BG = "#b0382b", "#fcf0ee"
C_TEN = "#555b66"
C_TXT = "#111111"
C_SUB = "#555555"
C_CS, C_CD = "#2b6cb0", "#d1462f"  # cage_s / cage_d (match cage.png)


# ---------------------------------------------------------------- svg helpers
def img(name, x, y, w, h):
    b64 = base64.b64encode((ASSETS / f"{name}.png").read_bytes()).decode("ascii")
    return (f'<image href="data:image/png;base64,{b64}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid meet"/>')


def est_width(s, size):
    """Rough rendered width of a label (Times-like ~0.47 em/char, subscripts x0.72)."""
    w = 0.0
    for part, is_sub in [(p, i % 2 == 1) for i, p in enumerate(re.split(r"</?sub>", s))]:
        plain = re.sub(r"<[^>]+>", "", part)
        plain = re.sub(r"&#770;|&#8203;", "", plain)
        plain = re.sub(r"&[^;]+;", "x", plain)
        w += len(plain) * 0.47 * size * (0.72 if is_sub else 1.0)
    return w


def text(x, y, s, size=16, color=C_TXT, anchor="middle", weight="normal", italic=False):
    style = ' font-style="italic"' if italic else ""
    # mixed-tspan labels: anchor at start with an estimated width, since some SVG
    # renderers mis-place tspans under text-anchor middle/end.
    if "<" in s and anchor != "start":
        w = est_width(s, size)
        x = x - (w / 2 if anchor == "middle" else w)
        anchor = "start"
    sub_size, dy = round(size * 0.72, 1), round(size * 0.28, 1)
    s = re.sub(r"<sub>(.*?)</sub>",
               lambda m_: (f'<tspan font-size="{sub_size}" font-style="normal" dy="{dy}">{m_.group(1)}</tspan>'
                           f'<tspan dy="-{dy}">&#8203;</tspan>'), s)
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{style}>{s}</text>')


def var(base, sub=None, hat=False):
    """italic variable with optional upright subscript, e.g. var('v', 'def', hat=True)."""
    b = f"{base}&#770;" if hat else base
    out = f'<tspan font-style="italic">{b}</tspan>'
    if sub:
        out += f"<sub>{sub}</sub>"
    return out


def box(x, y, w, h, stroke, fill="#ffffff", rx=6, dash=False, sw=1.4):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def path(points, color=C_TEN, head=True, dash=False, width=1.4):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
    dd = ' stroke-dasharray="5 4"' if dash else ""
    mk = ""
    if head:
        mk = ' marker-end="url(#ah-loss)"' if color == C_LOSS else ' marker-end="url(#ah)"'
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dd}{mk}/>'


def net(x, y, w, h, sym, name):
    return "".join([
        box(x, y, w, h, C_NET, C_NET_BG),
        text(x + w / 2, y + h / 2 + 2, sym, 24, C_NET, italic=True),
        text(x + w / 2, y + h / 2 + 22, name, 12, C_SUB),
    ])


def tensor(x, cy, w, label, shape, stroke=C_TEN, h=44):
    return "".join([
        box(x, cy - h / 2, w, h, stroke),
        text(x + w / 2, cy - 2, label, 15),
        text(x + w / 2, cy + 15, shape, 12, C_SUB),
    ])


def latent_strip(x, cy, colors, cell=11, h=22):
    out = []
    for i, c in enumerate(colors):
        out.append(f'<rect x="{x + i * cell}" y="{cy - h / 2}" width="{cell}" height="{h}" '
                   f'fill="{c}" stroke="#ffffff" stroke-width="1"/>')
    out.append(f'<rect x="{x}" y="{cy - h / 2}" width="{cell * len(colors)}" height="{h}" '
               f'fill="none" stroke="{C_TEN}" stroke-width="1.2"/>')
    return "".join(out)


def blend(cx, cy):
    return (f'<circle cx="{cx}" cy="{cy}" r="15" fill="#fff" stroke="{C_TXT}" stroke-width="1.4"/>'
            + text(cx, cy + 6, "&#215;", 20))


DEFS = f"""<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{C_TEN}"/></marker>
  <marker id="ah-loss" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{C_LOSS}"/></marker>
</defs>"""


def svg_wrap(w, h, parts):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'{DEFS}<rect width="100%" height="100%" fill="#ffffff"/>{"".join(parts)}</svg>')


# ================================================================ (a) loss flow
WA, HA = 1560, 640
A, B, C = 170, 380, 545  # rows: expression / neutral / weight

p = [text(14, 30, "(a)", 20, anchor="start", weight="bold")]

# inputs
p += [
    img("expression", 10, A - 80, 140, 150),
    text(80, A + 88, "expression", 15),
    text(80, A + 108, f"({var('v', 'def')}, {var('n', 'def')})", 15),
    img("neutral", 10, B - 80, 140, 150),
    text(80, B + 88, "neutral", 15),
    text(80, B + 108, f"({var('v', 'src')}, {var('n', 'src')})", 15),
]
XF, WF = 190, 180
p += [
    tensor(XF, A, WF, f"[&#916;{var('v')}, {var('n', 'def')}, {var('v', 'src')}, {var('n', 'src')}]", f"{N} &#215; 12"),
    tensor(XF, B, WF, f"[0, {var('n', 'def')}, {var('v', 'src')}, {var('n', 'src')}]", f"{N} &#215; 12"),
    tensor(XF, C, WF, f"[{var('v', 'src')}, {var('n', 'src')}]", f"{N} &#215; 6"),
    path([(150, A - 8), (XF, A - 8)]),
    path([(150, B - 30), (XF - 2, A + 12)]),
    path([(150, A + 30), (XF - 2, B - 12)]),
    path([(150, B + 8), (XF, B + 8)]),
    path([(80, B + 116), (80, C), (XF, C)]),
]

# phi (shared by both branches) -> z, z_s
XP, WP = 410, 92
p += [
    net(XP, A - 32, WP, B - A + 64, "&#966;", "exp_z_model"),
    path([(XF + WF, A), (XP, A)]),
    path([(XF + WF, B), (XP, B)]),
]
XZ, CELL, NCELL = 545, 11, 12
z_cols = ["#2b5d9b", "#7fa6d1", "#c5d7ec", "#4f7fb8", "#e3ecf6", "#2b5d9b",
          "#9dbbe0", "#3d6ea9", "#d4e2f1", "#6a95c8", "#b1cae6", "#2b5d9b"]
zs_cols = ["#e6e6e6"] * NCELL
WZ = CELL * NCELL
p += [
    latent_strip(XZ, A, z_cols, CELL), text(XZ + WZ / 2, A + 30, f"{var('z')}  (1 &#215; {L})", 15),
    latent_strip(XZ, B, zs_cols, CELL), text(XZ + WZ / 2, B + 30, f"{var('z', 's')}  (1 &#215; {L})", 15),
    path([(XP + WP, A), (XZ, A)]),
    path([(XP + WP, B), (XZ, B)]),
]

# g (key_d_model, shared) -> c_d, c_s (+ cage scatter)
XG, WG = 735, 92
XC, WC = 870, 112
p += [
    net(XG, A - 32, WG, B - A + 64, "g", "key_d_model"),
    path([(XZ + WZ, A), (XG, A)]),
    path([(XZ + WZ, B), (XG, B)]),
    tensor(XC, A, WC, var('c', 'd'), f"{M} &#215; 3", C_CD),
    tensor(XC, B, WC, var('c', 's'), f"{M} &#215; 3", C_CS),
    path([(XG + WG, A), (XC, A)]),
    path([(XG + WG, B), (XC, B)]),
    img("cage", XC - 8, A + 30, WC + 16, B - A - 60),
]

# psi -> W
p += [
    net(XP, C - 30, WP, 60, "&#968;", "key_weight_model"),
    path([(XF + WF, C), (XP, C)]),
    img("weight_col", XZ - 12, C - 55, 80, 92),
    img("weight_argmax", XZ + 72, C - 55, 80, 92),
    text(XZ + 70, C + 58, f"{var('W')}  ({N} &#215; {M}),  &#931;<sub>m</sub> {var('W')} = 1", 15),
    text(XZ + 28, C + 40, f"{var('W')}[:, m]", 12, C_SUB),
    text(XZ + 112, C + 40, f"argmax<sub>m</sub> {var('W')}", 12, C_SUB),
    path([(XP + WP, C), (XZ - 14, C)]),
]

# blend + outputs
XB = 1030
XO, WO = 1060, 120
p += [
    blend(XB, A), blend(XB, B),
    path([(XC + WC, A), (XB - 15, A)]),
    path([(XC + WC, B), (XB - 15, B)]),
    path([(XZ + 156, C), (XB, C), (XB, B + 15)], head=False),
    path([(XB, C), (XB, B + 15)]),
    path([(XB - 26, C), (XB - 26, A + 34), (XB - 10, A + 11)]),
    text(XB + 8, C - 8, var('W'), 15, C_SUB, anchor="start"),
    img("pred_def", XO + 6, A - 75, WO, 140),
    img("pred_neu", XO + 6, B - 75, WO, 140),
    text(XO + 6 + WO / 2, A + 84, f"{var('v', 'def', hat=True)} = {var('W')}{var('c', 'd')}", 16),
    text(XO + 6 + WO / 2, B + 84, f"{var('v', 'src', hat=True)} = {var('W')}{var('c', 's')}", 16),
    path([(XB + 15, A), (XO + 10, A)]),
    path([(XB + 15, B), (XO + 10, B)]),
]

# losses
XL, WL, HL = 1210, 320, 70
m_ = var('m')
p += [
    box(XL, A - HL / 2, WL, HL, C_LOSS, C_LOSS_BG),
    text(XL + 12, A - 10, var('L', 'def'), 19, C_LOSS, anchor="start"),
    text(XL + WL - 12, A - 10, "&#955; = 1", 14, C_SUB, anchor="end"),
    text(XL + 12, A + 20, f"&#8214;{m_}&#8857;({var('v', 'def', True)} &#8722; {var('v', 'def')})&#8214;&#178;"
         f" + &#8214;(1&#8722;{m_})&#8857;({var('v', 'def', True)} &#8722; {var('v', 'src')})&#8214;&#178;", 13.5, anchor="start"),
    box(XL, B - HL / 2, WL, HL, C_LOSS, C_LOSS_BG),
    text(XL + 12, B - 10, var('L', 'neu'), 19, C_LOSS, anchor="start"),
    text(XL + WL - 12, B - 10, "&#955; = 1", 14, C_SUB, anchor="end"),
    text(XL + 12, B + 20, f"&#8214;{m_}&#8857;({var('v', 'src', True)} &#8722; {var('v', 'src')})&#8214;&#178;"
         f" + &#8214;(1&#8722;{m_})&#8857;({var('v', 'src', True)} &#8722; {var('v', 'src')})&#8214;&#178;", 13.5, anchor="start"),
    text(XL + 12, B + 58, f"{m_} = hat({var('v', 'src')}) : face mask", 13, C_SUB, anchor="start"),
    path([(XO + 6 + WO, A), (XL, A)], C_LOSS),
    path([(XO + 6 + WO, B), (XL, B)], C_LOSS),
]

# L_z above z
ZX, ZY, ZW, ZH = XZ - 85, 22, 310, 66
p += [
    box(ZX, ZY, ZW, ZH, C_LOSS, C_LOSS_BG),
    text(ZX + 12, ZY + 24, var('L', 'z'), 19, C_LOSS, anchor="start"),
    text(ZX + ZW - 12, ZY + 24, "&#955; = 0.5", 14, C_SUB, anchor="end"),
    text(ZX + 12, ZY + 50, f"L<sub>FACS</sub>({var('z')}[:53]) + &#8214;{var('z')}[53:]&#8214;&#178;"
         f"   (ICT: &#8214;{var('z')} &#8722; {var('e')}&#8214;&#178;)", 13.5, anchor="start"),
    path([(XZ + WZ / 2, A - 11), (XZ + WZ / 2, ZY + ZH)], C_LOSS),
]

# total
TY = C - 40
BUS = XL + WL + 14
p += [
    box(XL, TY, WL, 80, C_LOSS, "#ffffff"),
    text(XL + WL / 2, TY + 32, f"{var('L')} = {var('L', 'def')} + {var('L', 'neu')} + 0.5 {var('L', 'z')}", 18),
    text(XL + WL / 2, TY + 60, f"&#8711; &#8594; &#966;, g, &#968;", 15, C_LOSS),
    path([(XL + WL, A), (BUS, A)], C_LOSS, head=False),
    path([(XL + WL, B), (BUS, B)], C_LOSS, head=False),
    path([(ZX + ZW, ZY + ZH / 2), (BUS, ZY + ZH / 2), (BUS, C)], C_LOSS, head=False),
    path([(BUS, C), (XL + WL + 2, C)], C_LOSS),
]
svg_a = svg_wrap(WA, HA, p)


# ================================================================ (b) LinearEncoder
WB, HB = 1560, 370
Y1, Y2 = 130, 290  # main path / style path
q = [text(14, 30, "(b)", 20, anchor="start", weight="bold"),
     text(60, 30, "LinearEncoder  (shared by &#966;, g, &#968;)", 17, anchor="start")]

q += [
    tensor(20, Y1, 100, var('x'), f"{var('N')} &#215; {var('C')}"),
    box(160, Y1 - 26, 120, 52, C_NET, C_NET_BG),
    text(220, Y1 - 2, "Linear", 15), text(220, Y1 + 16, f"{var('C')} &#8594; 128", 12, C_SUB),
    path([(120, Y1), (160, Y1)]),
]
# modulation block x4
MX, MW = 330, 330
q += [
    box(MX, Y1 - 58, MW, 116, C_NET, "#ffffff", dash=True),
    text(MX + MW - 10, Y1 - 40, "&#215; 4", 17, C_NET, anchor="end", weight="bold"),
    box(MX + 20, Y1 - 24, 110, 48, C_NET, C_NET_BG),
    text(MX + 75, Y1 - 2, "MLP", 15), text(MX + 75, Y1 + 15, "128 &#8594; 128", 12, C_SUB),
    text(MX + 225, Y1 + 6, f"{var('h')} &#8857; &#963;<sub>k</sub>({var('s')}) + &#956;<sub>k</sub>({var('s')})", 16),
    path([(MX + 130, Y1), (MX + 160, Y1)]),
    path([(280, Y1), (MX + 20, Y1)]),
    text(20, Y2 + 62, "MLP = Linear &#8594; LayerNorm &#8594; LeakyReLU(0.2) &#8594; Linear  (no residual);  MLP&#8325;: 5 Linear layers", 13, C_SUB, anchor="start"),
]
# output linear
OX = 710
q += [
    box(OX, Y1 - 26, 130, 52, C_NET, C_NET_BG),
    text(OX + 65, Y1 - 2, "Linear", 15), text(OX + 65, Y1 + 15, "128 &#8594; out", 12, C_SUB),
    path([(MX + MW, Y1), (OX, Y1)]),
]
# heads
HX, HW = 890, 320
heads = [
    (Y1 - 75, f"&#966; :  mean<sub>N</sub> &#8594; {var('z')}  (1 &#215; {L})"),
    (Y1, f"g :  reshape &#8594; {var('c')}  ({M} &#215; 3)"),
    (Y1 + 75, f"&#968; :  col-norm &#8594; ReLU &#8594; row-norm &#8594; {var('W')}"),
]
for hy, s in heads:
    q += [box(HX, hy - 22, HW, 44, C_TEN), text(HX + 12, hy + 6, s, 15, anchor="start"),
          path([(OX + 130, Y1), (HX, hy)])]

# style branch
q += [
    path([(70, Y1 + 22), (70, Y2), (160, Y2)]),
    box(160, Y2 - 26, 120, 52, C_NET, C_NET_BG),
    text(220, Y2 - 2, "MLP&#8325;", 15), text(220, Y2 + 15, f"{var('C')} &#8594; 128", 12, C_SUB),
    box(320, Y2 - 22, 110, 44, C_TEN),
    text(375, Y2 + 6, f"mean<sub>N</sub>", 15),
    path([(280, Y2), (320, Y2)]),
    f'<circle cx="475" cy="{Y2}" r="13" fill="#fff" stroke="{C_TXT}" stroke-width="1.4"/>',
    text(475, Y2 + 6, "+", 18),
    path([(430, Y2), (462, Y2)]),
    path([(300, Y1), (300, Y1 + 80), (475, Y1 + 80), (475, Y2 - 13)]),
    text(390, Y1 + 96, f"mean<sub>N</sub>({var('h')})", 13, C_SUB),
    path([(488, Y2), (530, Y2)]),
    text(545, Y2 + 6, f"{var('s')}  (1 &#215; 128)", 16, anchor="start"),
    path([(MX + 225, Y2 - 14), (MX + 225, Y1 + 58)], C_NET),
    text(MX + 235, Y2 - 30, f"&#963;<sub>k</sub>, &#956;<sub>k</sub> : MLP", 13, C_NET, anchor="start"),
]

# module table
TX, TY0 = 1250, 70
cols = [0, 70, 140, 225]
rows = [
    ("", f"in {var('C')}", "out", "params"),
    ("&#966;", "12", f"{L}", "486 K"),
    ("g", f"{L}", f"{M}&#183;3", "524 K"),
    ("&#968;", "6", f"{M}", "476 K"),
]
for r, row in enumerate(rows):
    y = TY0 + 34 * r
    for c, s in enumerate(row):
        q.append(text(TX + cols[c], y, s, 16 if c == 0 else 15,
                      C_NET if c == 0 else (C_SUB if r == 0 else C_TXT), anchor="start", italic=(c == 0)))
q += [
    f'<line x1="{TX - 6}" y1="{TY0 + 10}" x2="{TX + 290}" y2="{TY0 + 10}" stroke="{C_TEN}" stroke-width="1"/>',
    text(TX, TY0 + 34 * 4 + 6, f"{var('N')} = {N} vertices,  {M} cage points", 13, C_SUB, anchor="start"),
]
svg_b = svg_wrap(WB, HB, q)


# ---------------------------------------------------------------- page
html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NGBC loss flow</title>
<style>
  html, body {{ background:#ffffff; margin:0; font-family:{FONT}; }}
  .wrap {{ max-width:{WA}px; margin:0 auto; padding:16px; }}
  svg {{ width:100%; height:auto; display:block; margin-bottom:12px; }}
</style>
</head><body>
<div class="wrap">
{svg_a}
{svg_b}
</div>
</body></html>
"""

OUT_ROOT.mkdir(parents=True, exist_ok=True)
out_path = OUT_ROOT / "loss_flow.html"
out_path.write_text(html)
print(f"wrote {out_path}")
