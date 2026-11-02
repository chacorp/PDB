"""Build vis_CBD/vis_loss_flow/loss_flow.html: train_CBD (version 5, NGBC) on a white
background, serif (Times-like) font, two inline-SVG panels:
  (a) one-step loss flow with real inputs / W / cage / outputs of one sample
      (assets from save_loss_flow_assets.py)
  (b) the LinearEncoder architecture + per-module table
phi = exp_z_model -> key_d_model is drawn as one model; psi = key_weight_model.

Losses drawn: recon-def, recon-neu (ckpts_CBD9 setting: in_type 1, out_type 1, relu).
The latent loss (exp-z) is intentionally left out of the diagram.

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
        text(x + w / 2, y + h / 2 + 9, sym, 26, C_NET, italic=True),
        text(x + w / 2, y + h / 2 + 22, name, 12, C_SUB) if name else "",
    ])


def tensor(x, cy, w, label, shape, stroke=C_TEN, h=44):
    return "".join([
        box(x, cy - h / 2, w, h, stroke),
        text(x + w / 2, cy - 2, label, 15),
        text(x + w / 2, cy + 15, shape, 12, C_SUB),
    ])



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
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}" '
            f'style="max-width:{w}px">'
            f'{DEFS}<rect width="100%" height="100%" fill="#ffffff"/>{"".join(parts)}</svg>')


# ================================================================ (a) loss flow
WA, HA = 1040, 500
A, B, C = 95, 290, 432  # rows: expression / neutral / weight

p = [text(4, 18, "(a)", 20, anchor="start", weight="bold")]

# inputs
p += [
    img("expression", 10, A - 80, 140, 150),
    text(80, A + 90, f"expression {var('v', 'def')}", 16),
    img("neutral", 10, B - 80, 140, 150),
    text(80, B + 90, f"neutral {var('v', 'src')}", 16),
]

# phi (exp_z_model + key_d_model) -> c_d, c_s ; psi -> W
XP, WP = 225, 100
p += [
    net(XP, A - 30, WP, B - A + 60, "&#966;", ""),
    path([(150, A), (XP, A)]),
    path([(150, B - 20), (XP, A + 22)]),
    path([(150, B), (XP, B)]),
    net(XP, C - 28, WP, 56, "&#968;", ""),
    path([(80, B + 100), (80, C), (XP, C)]),
]

# c_d / c_s + cage points between them
XC, WC, HC = 375, 80, 34
p += [
    box(XC, A - HC / 2, WC, HC, C_CD), text(XC + WC / 2, A + 6, var('c', 'd'), 17),
    box(XC, B - HC / 2, WC, HC, C_CS), text(XC + WC / 2, B + 6, var('c', 's'), 17),
    path([(XP + WP, A), (XC, A)]),
    path([(XP + WP, B), (XC, B)]),
    img("cage", XC - 25, A + HC / 2 + 6, WC + 50, B - A - HC - 12),
]

# W (argmax over cage points)
XW, WW = 368, 94
p += [
    img("weight_argmax", XW, C - 50, WW, 100),
    text(XW + WW / 2, C + 64, var('W'), 17),
    path([(XP + WP, C), (XW - 2, C)]),
]

# blend + outputs
XB = 512
XO, WO = 540, 120
p += [
    blend(XB, A), blend(XB, B),
    path([(XC + WC, A), (XB - 15, A)]),
    path([(XC + WC, B), (XB - 15, B)]),
    path([(XW + WW + 2, C), (XB, C), (XB, B + 15)]),
    path([(XB - 26, C), (XB - 26, A + 34), (XB - 10, A + 11)]),
    img("pred_def", XO + 6, A - 75, WO, 140),
    img("pred_neu", XO + 6, B - 75, WO, 140),
    text(XO + 6 + WO / 2, A + 88, f"{var('v', 'def', hat=True)} = {var('W')}{var('c', 'd')}", 17),
    text(XO + 6 + WO / 2, B + 88, f"{var('v', 'src', hat=True)} = {var('W')}{var('c', 's')}", 17),
    path([(XB + 15, A), (XO + 10, A)]),
    path([(XB + 15, B), (XO + 10, B)]),
]

# losses
XL, WL, HL = 700, 318, 64
m_ = var('m')
p += [
    box(XL, A - HL / 2, WL, HL, C_LOSS, C_LOSS_BG),
    text(XL + 12, A - 8, var('L', 'def'), 19, C_LOSS, anchor="start"),
    text(XL + 12, A + 20, f"&#8214;{m_}&#8857;({var('v', 'def', True)} &#8722; {var('v', 'def')})&#8214;&#178;"
         f" + &#8214;(1&#8722;{m_})&#8857;({var('v', 'def', True)} &#8722; {var('v', 'src')})&#8214;&#178;", 13.5, anchor="start"),
    box(XL, B - HL / 2, WL, HL, C_LOSS, C_LOSS_BG),
    text(XL + 12, B - 8, var('L', 'neu'), 19, C_LOSS, anchor="start"),
    text(XL + 12, B + 20, f"&#8214;{m_}&#8857;({var('v', 'src', True)} &#8722; {var('v', 'src')})&#8214;&#178;"
         f" + &#8214;(1&#8722;{m_})&#8857;({var('v', 'src', True)} &#8722; {var('v', 'src')})&#8214;&#178;", 13.5, anchor="start"),
    path([(XO + 6 + WO, A), (XL, A)], C_LOSS),
    path([(XO + 6 + WO, B), (XL, B)], C_LOSS),
]

# total
TH = 56
BUS = XL + WL + 12
p += [
    box(XL, C - TH / 2, WL, TH, C_LOSS, "#ffffff"),
    text(XL + WL / 2, C + 7, f"{var('L')} = {var('L', 'def')} + {var('L', 'neu')}", 19),
    path([(XL + WL, A), (BUS, A)], C_LOSS, head=False),
    path([(XL + WL, B), (BUS, B)], C_LOSS, head=False),
    path([(BUS, A), (BUS, C)], C_LOSS, head=False),
    path([(BUS, C), (XL + WL + 2, C)], C_LOSS),
]
svg_a = svg_wrap(WA, HA, p)


# ================================================================ (b) LinearEncoder
WB, HB = 1560, 300
Y1, Y2 = 110, 240  # main path / style path
q = [text(4, 22, "(b)", 20, anchor="start", weight="bold"),
     text(50, 22, "LinearEncoder  (building block of &#966; and &#968;)", 17, anchor="start")]

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
    text(20, Y2 + 50, "MLP = Linear &#8594; LayerNorm &#8594; LeakyReLU(0.2) &#8594; Linear  (no residual);  MLP&#8325;: 5 Linear layers", 13, C_SUB, anchor="start"),
]
# output linear
OX = 710
q += [
    box(OX, Y1 - 26, 130, 52, C_NET, C_NET_BG),
    text(OX + 65, Y1 - 2, "Linear", 15), text(OX + 65, Y1 + 15, "128 &#8594; out", 12, C_SUB),
    path([(MX + MW, Y1), (OX, Y1)]),
]
# heads
HX, HW = 890, 340
heads = [
    (Y1 - 45, f"&#966; :  mean<sub>N</sub> &#8594; LinearEncoder &#8594; {var('c')}  ({M} &#215; 3)"),
    (Y1 + 45, f"&#968; :  col-norm &#8594; ReLU &#8594; row-norm &#8594; {var('W')}"),
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
TX, TY0 = 1270, 70
cols = [0, 70, 140, 225]
rows = [
    ("", f"in {var('C')}", "out", "params"),
    ("&#966;", "12", f"{M}&#183;3", "1,010 K"),
    ("&#968;", "6", f"{M}", "476 K"),
]
for r, row in enumerate(rows):
    y = TY0 + 34 * r
    for c, s in enumerate(row):
        q.append(text(TX + cols[c], y, s, 16 if c == 0 else 15,
                      C_NET if c == 0 else (C_SUB if r == 0 else C_TXT), anchor="start", italic=(c == 0)))
q += [
    f'<line x1="{TX - 6}" y1="{TY0 + 10}" x2="{TX + 280}" y2="{TY0 + 10}" stroke="{C_TEN}" stroke-width="1"/>',
    text(TX, TY0 + 34 * 3 + 6, f"{var('N')} = {N} vertices,  {M} cage points", 13, C_SUB, anchor="start"),
]
svg_b = svg_wrap(WB, HB, q)


# ---------------------------------------------------------------- page
html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NGBC loss flow</title>
<style>
  html, body {{ background:#ffffff; margin:0; font-family:{FONT}; }}
  .wrap {{ max-width:{max(WA, WB)}px; margin:0 auto; padding:6px; }}
  svg {{ width:100%; height:auto; display:block; margin-bottom:4px; }}
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
