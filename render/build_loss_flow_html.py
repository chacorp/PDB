"""Build vis_CBD/vis_loss_flow/loss_flow.html: train_CBD (version 5, NGBC) one-step
loss flow as an inline SVG on a white background, with the inputs / intermediate
tensors / outputs of one real sample embedded (assets from save_loss_flow_assets.py).

Active losses are the ckpts_CBD9 setting only (in_type 1, out_type 1, relu,
align_latent, no optional losses): recon-def, recon-neu, exp-z.

Run from repo root (after render/save_loss_flow_assets.py):
    python render/build_loss_flow_html.py
"""
import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_loss_flow"
ASSETS = OUT_ROOT / "assets"

VAL = json.load(open(ASSETS / "values.json"))
N, M, L = VAL["N"], VAL["M"], VAL["L"]

W_SVG, H_SVG = 1790, 900
A, B, C = 230, 490, 770  # row centers: expression branch / neutral branch / weight branch

C_NET, C_NET_BG = "#2b6cb0", "#eef4fb"
C_LOSS, C_LOSS_BG = "#c0392b", "#fdf1ef"
C_TEN = "#6b7280"
C_TXT = "#111827"
C_SUB = "#4b5563"


def img(name, x, y, w, h):
    b64 = base64.b64encode((ASSETS / f"{name}.png").read_bytes()).decode("ascii")
    return (f'<image href="data:image/png;base64,{b64}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid meet"/>')


def text(x, y, s, size=13, color=C_TXT, anchor="middle", weight="normal", italic=False):
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{style}>{s}</text>')


def box(x, y, w, h, stroke, fill="#ffffff", rx=8, dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{d}/>'


def net(x, cy, w, title, sub):
    h = 60
    return "".join([
        box(x, cy - h / 2, w, h, C_NET, C_NET_BG),
        text(x + w / 2, cy - 4, title, 15, C_NET, weight="bold"),
        text(x + w / 2, cy + 16, sub, 11, C_SUB),
    ])


def tensor(x, cy, w, title, shape, h=52):
    return "".join([
        box(x, cy - h / 2, w, h, C_TEN),
        text(x + w / 2, cy - 4, title, 13),
        text(x + w / 2, cy + 14, shape, 11, C_SUB),
    ])


def arrow(points, color=C_TEN, dash=False, width=1.6):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
    dd = ' stroke-dasharray="5 4"' if dash else ""
    marker = "url(#ah-loss)" if color == C_LOSS else "url(#ah)"
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dd} marker-end="{marker}"/>'


def line(points, color=C_TEN, width=1.6):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def blend(cx, cy):
    return (f'<circle cx="{cx}" cy="{cy}" r="17" fill="#fff" stroke="{C_TXT}" stroke-width="1.6"/>'
            + text(cx, cy + 5, "&#215;", 18))


def loss_box(x, cy, key, formula_lines, thumbs):
    h, w = 200, 300
    y = cy - h / 2
    lam = VAL["lambda"][key]
    out = [box(x, y, w, h, C_LOSS, C_LOSS_BG),
           text(x + 14, y + 24, key, 16, C_LOSS, anchor="start", weight="bold"),
           text(x + w - 14, y + 24, f"&#955; = {lam}", 12, C_SUB, anchor="end")]
    for i, s in enumerate(formula_lines):
        out.append(text(x + 14, y + 48 + 17 * i, s, 12, C_TXT, anchor="start"))
    tx = x + 14
    for name, cap in thumbs:
        out.append(img(name, tx, y + 92, 74, 80))
        out.append(text(tx + 37, y + 186, cap, 10.5, C_SUB))
        tx += 92
    out.append(text(x + w - 14, y + h - 14, f"{VAL['loss'][key]:.3e}", 13, C_LOSS, anchor="end", weight="bold"))
    return "".join(out)


parts = []
# ---------------------------------------------------------------- inputs
parts += [
    img("expression", 15, A - 95, 160, 175), text(95, A + 98, "expression  v_def, n_def", 12),
    img("neutral", 15, B - 95, 160, 175), text(95, B + 98, "neutral  v_src, n_src", 12),
    text(95, A - 112, "Input", 15, weight="bold"),
    text(95, B + 116, f"N = {N}", 10.5, C_SUB),
    text(95, B + 130, "train: random subset (p = 0.7)", 10.5, C_SUB),
]
X_IN, W_IN = 245, 205
parts += [
    tensor(X_IN, A, W_IN, "[&#916;v, n_def, v_src, n_src]", f"N &#215; 12"),
    tensor(X_IN, B, W_IN, "[0, n_def, v_src, n_src]", f"N &#215; 12"),
    tensor(X_IN, C, W_IN, "[v_src, n_src]", f"N &#215; 6"),
    arrow([(175, A - 10), (X_IN, A - 10)]),
    arrow([(175, B - 30), (X_IN - 2, A + 14)]),
    arrow([(175, A + 30), (X_IN - 2, B - 14)]),
    arrow([(175, B + 10), (X_IN, B + 10)]),
    arrow([(95, B + 140), (95, C), (X_IN, C)]),
]

# ---------------------------------------------------------------- encoders
X_PHI, W_PHI = 495, 120
parts += [
    net(X_PHI, A, W_PHI, "&#966;", "exp_z_model"),
    net(X_PHI, B, W_PHI, "&#966;", "exp_z_model"),
    net(X_PHI, C, W_PHI, "&#968;", "key_weight_model"),
    text(X_PHI + W_PHI / 2, (A + B) / 2 + 5, "shared", 11, C_NET, italic=True),
    line([(X_PHI + W_PHI / 2, A + 30), (X_PHI + W_PHI / 2, (A + B) / 2 - 10)], C_NET, 1.2),
    line([(X_PHI + W_PHI / 2, (A + B) / 2 + 12), (X_PHI + W_PHI / 2, B - 30)], C_NET, 1.2),
    arrow([(X_IN + W_IN, A), (X_PHI, A)]),
    arrow([(X_IN + W_IN, B), (X_PHI, B)]),
    arrow([(X_IN + W_IN, C), (X_PHI, C)]),
    text(X_PHI + W_PHI / 2, C + 48, "col-norm &#8594; ReLU &#8594; row-norm", 10.5, C_SUB),
]

# ---------------------------------------------------------------- latents / weights
X_Z, W_Z = 660, 210
parts += [
    img("exp_z", X_Z, A - 62, W_Z, 78),
    text(X_Z + W_Z / 2, A + 34, f"exp_z  (1 &#215; {L})", 12),
    arrow([(X_PHI + W_PHI, A), (X_Z, A)]),
    tensor(X_Z, B, W_Z, "exp_z_s", f"1 &#215; {L}"),
    arrow([(X_PHI + W_PHI, B), (X_Z, B)]),
    img("weight_col", X_Z - 4, C - 82, 104, 120),
    img("weight_argmax", X_Z + 110, C - 82, 104, 120),
    text(X_Z + 48, C + 52, "W[:, m]", 11, C_SUB),
    text(X_Z + 162, C + 52, "argmax&#8344; W", 11, C_SUB),
    text(X_Z + W_Z / 2, C + 74, f"W  (N &#215; {M}),  &#931;&#8344; W = 1", 12),
    arrow([(X_PHI + W_PHI, C), (X_Z - 6, C)]),
]

# ---------------------------------------------------------------- decoder / cage
X_KD, W_KD = 915, 120
X_CG, W_CG = 1080, 120
parts += [
    net(X_KD, A, W_KD, "key_d", "key_d_model"),
    net(X_KD, B, W_KD, "key_d", "key_d_model"),
    text(X_KD + W_KD / 2, (A + B) / 2 + 5, "shared", 11, C_NET, italic=True),
    line([(X_KD + W_KD / 2, A + 30), (X_KD + W_KD / 2, (A + B) / 2 - 10)], C_NET, 1.2),
    line([(X_KD + W_KD / 2, (A + B) / 2 + 12), (X_KD + W_KD / 2, B - 30)], C_NET, 1.2),
    arrow([(X_Z + W_Z, A), (X_KD, A)]),
    arrow([(X_Z + W_Z, B), (X_KD, B)]),
    tensor(X_CG, A, W_CG, "cage_d", f"{M} &#215; 3"),
    tensor(X_CG, B, W_CG, "cage_s", f"{M} &#215; 3"),
    arrow([(X_KD + W_KD, A), (X_CG, A)]),
    arrow([(X_KD + W_KD, B), (X_CG, B)]),
    img("cage", X_CG - 10, A + 36, W_CG + 20, B - A - 72),
]

# ---------------------------------------------------------------- blend + outputs
X_BL = 1245
X_OUT, W_OUT = 1275, 120
parts += [
    blend(X_BL, A), blend(X_BL, B),
    arrow([(X_CG + W_CG, A), (X_BL - 17, A)]),
    arrow([(X_CG + W_CG, B), (X_BL - 17, B)]),
    # W -> both blends
    line([(X_Z + W_Z + 4, C), (X_BL, C)], C_TEN),
    arrow([(X_BL, C), (X_BL, B + 17)]),
    line([(X_BL - 26, C), (X_BL - 26, A + 40)], C_TEN),
    arrow([(X_BL - 26, A + 40), (X_BL - 11, A + 13)]),
    text(X_BL + 6, C - 10, "W", 12, C_SUB, anchor="start"),
    img("pred_def", X_OUT + 18, A - 95, W_OUT, 150),
    img("pred_neu", X_OUT + 18, B - 95, W_OUT, 150),
    text(X_OUT + 18 + W_OUT / 2, A + 72, "pred_def = W&#183;cage_d", 12),
    text(X_OUT + 18 + W_OUT / 2, B + 72, "pred_neu = W&#183;cage_s", 12),
    text(X_OUT + 18 + W_OUT / 2, A - 112, "Output", 15, weight="bold"),
    arrow([(X_BL + 17, A), (X_OUT + 22, A)]),
    arrow([(X_BL + 17, B), (X_OUT + 22, B)]),
]

# ---------------------------------------------------------------- losses
X_L = 1470
parts += [
    loss_box(X_L, A, "recon-def",
             ["MSE(v_def&#183;m, pred_def&#183;m)",
              "+ MSE(v_src&#183;(1&#8722;m), pred_def&#183;(1&#8722;m))"],
             [("expression", "target v_def"), ("t_mask", "m = hat(v_src)")]),
    loss_box(X_L, B, "recon-neu",
             ["MSE(v_src&#183;m, pred_neu&#183;m)",
              "+ MSE(v_src&#183;(1&#8722;m), pred_neu&#183;(1&#8722;m))"],
             [("neutral", "target v_src"), ("t_mask", "m = hat(v_src)")]),
    arrow([(X_OUT + 18 + W_OUT, A), (X_L, A)], C_LOSS),
    arrow([(X_OUT + 18 + W_OUT, B), (X_L, B)], C_LOSS),
]

# exp-z above the exp_z latent
EZ_X, EZ_Y, EZ_W, EZ_H = X_Z - 20, 22, 250, 76
parts += [
    box(EZ_X, EZ_Y, EZ_W, EZ_H, C_LOSS, C_LOSS_BG),
    text(EZ_X + 12, EZ_Y + 22, "exp-z", 16, C_LOSS, anchor="start", weight="bold"),
    text(EZ_X + EZ_W - 12, EZ_Y + 22, f"&#955; = {VAL['lambda']['exp-z']}", 12, C_SUB, anchor="end"),
    text(EZ_X + 12, EZ_Y + 44, "ICT: MSE(exp_coeff, exp_z)", 11.5, C_TXT, anchor="start"),
    text(EZ_X + 12, EZ_Y + 62, "else: L_FACS(z[:53]) + &#8214;z[53:]&#8214;&#178;", 11.5, C_TXT, anchor="start"),
    text(EZ_X + EZ_W - 12, EZ_Y + 44, f"{VAL['loss']['exp-z']:.2e}", 12, C_LOSS, anchor="end", weight="bold"),
    arrow([(X_Z + W_Z / 2, A - 62), (X_Z + W_Z / 2, EZ_Y + EZ_H)], C_LOSS),
]

# total (bus on the right edge)
TOT_Y = C - 55
BUS = X_L + 300 + 14
parts += [
    box(X_L, TOT_Y, 300, 120, C_LOSS, "#ffffff"),
    text(X_L + 150, TOT_Y + 28, "total", 16, C_LOSS, weight="bold"),
    text(X_L + 150, TOT_Y + 54, "1.0&#183;recon-def + 1.0&#183;recon-neu", 12),
    text(X_L + 150, TOT_Y + 72, "+ 0.5&#183;exp-z", 12),
    text(X_L + 150, TOT_Y + 102, f"{VAL['total']:.3e}", 14, C_LOSS, weight="bold"),
    line([(X_L + 300, A), (BUS, A)], C_LOSS),
    line([(X_L + 300, B), (BUS, B)], C_LOSS),
    line([(EZ_X + EZ_W, EZ_Y + EZ_H / 2), (BUS, EZ_Y + EZ_H / 2), (BUS, C)], C_LOSS),
    arrow([(BUS, C), (X_L + 302, C)], C_LOSS),
    text(X_L + 150, TOT_Y + 145, "&#8711; &#8594; &#968;, &#966;, key_d", 12, C_LOSS),
]

# ---------------------------------------------------------------- page
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_SVG} {H_SVG}" font-family="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif">
<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{C_TEN}"/></marker>
  <marker id="ah-loss" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{C_LOSS}"/></marker>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
{''.join(parts)}
</svg>"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NGBC loss flow</title>
<style>
  html, body {{ background:#ffffff; margin:0; }}
  .wrap {{ max-width:{W_SVG}px; margin:0 auto; padding:16px; }}
  svg {{ width:100%; height:auto; display:block; }}
</style>
</head><body>
<div class="wrap">
{svg}
</div>
</body></html>
"""

OUT_ROOT.mkdir(parents=True, exist_ok=True)
out_path = OUT_ROOT / "loss_flow.html"
out_path.write_text(html)
print(f"wrote {out_path}")
