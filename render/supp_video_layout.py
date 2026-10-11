"""
Supplementary video, stage 3: compose a 1920x1080 frame (black background) from the
per-output cells of render/supp_video_render.py, same arrangement as the HTML figures:
  left  : Source Neutral / Source Expression / Target Neutral (one column)
  right : rows = Self / Cross / Cyclic retargeting, columns = models
  top   : video title + scenario, animation frame number (dataloader_CBD index)
  bottom: column captions

    python render/supp_video_layout.py            # pre-vis frames (one per video)
"""
from pathlib import Path

import re
import html

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
CELLS_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "cells"
W, H = 1920, 1080
BG, FG, LINE = (0, 0, 0), (255, 255, 255), (90, 90, 90)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # has glyphs Serif lacks (e.g. U+2112)
# labels / captions exactly as in the HTML reports (HTML snippets, drawn by _rich)
REF_ROWS = [("source_neutral", "Source Neutral Mesh"), ("source_expression", "Source Expression Mesh"),
            ("target_neutral", "Target Neutral Mesh")]
ROWS = [("self", "Self retargeting"), ("cross", "Cross retargeting"), ("cyclic", "Cyclic retargeting")]

VIDEOS = {
    # build_design_study_figure_gallery.COLUMNS
    "design_study": dict(title="Design study", columns=[
        ("ours", "Ours<sup>&dagger;</sup>"), ("no_activation", "no activation"), ("softplus", "w/softplus"),
        ("elu", "w/ELU"), ("soft_pou", "w/softPOU"), ("wo_mask", "w/oMask"), ("eq15_delta", "Equation 15"),
        ("eq16_matrix", "Equation 16")]),
    # vis_comparison.html
    "comparison": dict(title="Comparison", columns=[("NC", "NC"), ("NFR", "NFR"), ("NFS", "NFS"), ("PDB", "PDB")]),
    # build_ablation_gallery_mask.FIG_COLUMNS
    "loss_ablation": dict(title="Loss ablation", columns=[
        ("baseline", "Ours<sup>&dagger;</sup>"),
        ("w_consist", "w/ <i>&#8466;</i><sub>consist</sub>"),
        ("w_cyclic", "w/ <i>&#8466;</i><sub>cyclic</sub>"),
        ("w_both", "w/ <i>&#8466;</i><sub>consist</sub> + <i>&#8466;</i><sub>cyclic</sub>")]),
}
SCENARIO_TITLE = {"mf_to_coma": "mf -> coma", "mf_to_biwi": "mf -> biwi", "mf_to_ict": "mf -> ict",
                  "ict_to_mf_m00": "ict m00 -> mf", "ict_to_mf_m02": "ict m02 -> mf"}


def _font(size, path=FONT):
    return ImageFont.truetype(path, size)


def _has_glyphs(font, text):
    notdef = font.getmask("\ue000")
    return all(bytes(font.getmask(ch)) != bytes(notdef) or ch.isspace() for ch in text)


def _font_for(text, size):
    f = _font(size)
    return f if _has_glyphs(f, text) else _font(size, FONT_FALLBACK)


def _fit(path, size):
    canvas = Image.new("RGB", (size, size), BG)
    if path is None or not Path(path).exists():
        return canvas
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def _segments(html_text):
    """minimal HTML snippet -> [(text, style)] with style in normal / sup / sub / i"""
    text = html.unescape(re.sub(r"<(/?)(sup|sub|i)>", lambda m: f"\x00{m.group(1)}{m.group(2)}\x00", html_text))
    out, style = [], "normal"
    for tok in text.split("\x00"):
        if tok in ("sup", "sub", "i"):
            style = tok
        elif tok in ("/sup", "/sub", "/i"):
            style = "normal"
        elif tok:
            out.append((tok, style))
    return out


def _rich_image(html_text, size):
    """render an HTML caption (sup / sub / italic) to an RGBA image, glyphs on a common baseline"""
    base = _font(size)
    H_ = size * 3
    baseline = size * 2
    canvas = Image.new("RGBA", (size * 2 + int(base.getlength(html.unescape(re.sub(r"<[^>]+>", "", html_text)))) * 2,
                                H_), BG + (0,))
    x = 4
    for text, style in _segments(html_text):
        f = _font_for(text, int(size * 0.7) if style in ("sup", "sub") else size)
        ascent = f.getmetrics()[0]
        shift = {"sup": -int(size * 0.4), "sub": int(size * 0.25)}.get(style, 0)
        y = baseline - ascent + shift
        adv = f.getlength(text)
        if style == "i":  # no italic face installed: draw upright, then shear around the baseline
            g = Image.new("RGBA", (int(adv) + size, H_), BG + (0,))
            ImageDraw.Draw(g).text((size // 2, y), text, fill=FG, font=f)
            k = 0.2
            g = g.transform(g.size, Image.AFFINE, (1, k, -k * baseline, 0, 1, 0),
                            resample=Image.BICUBIC, fillcolor=BG + (0,))
            canvas.alpha_composite(g, (int(x) - size // 2, 0))
        else:
            ImageDraw.Draw(canvas).text((x, y), text, fill=FG, font=f)
        x += adv + (2 if style == "i" else 0)
    return canvas.crop(canvas.getbbox() or (0, 0, 1, 1))


def _rich(img, html_text, center_x, top_y, size):
    im = _rich_image(html_text, size)
    img.paste(im, (int(center_x - im.width / 2), int(top_y)), im)


def _vtext(img, text, xy_center, font, max_len=None):
    """text rotated 90 degrees (reads bottom to top), centered at xy_center; the font
    is shrunk so the text fits within max_len pixels"""
    while max_len and font.getlength(text) > max_len and font.size > 10:
        font = _font(font.size - 1)
    w, h = font.getbbox(text)[2:]
    tmp = Image.new("RGBA", (w + 4, h + 6), BG + (0,))
    ImageDraw.Draw(tmp).text((2, 0), text, fill=FG, font=font)
    tmp = tmp.rotate(90, expand=True)
    img.paste(tmp, (xy_center[0] - tmp.width // 2, xy_center[1] - tmp.height // 2), tmp)


def compose_frame(ref_cells, result_cells, columns, title, frame_text):
    """ref_cells {ref key: png}, result_cells {(row, column key): png}, columns [(key, caption)]"""
    top, bottom, label_w, gap = 70, 50, 34, 24
    n = len(columns)
    cell = min((H - top - bottom) // 3, (W - 2 * label_w - gap - 20) // (n + 1))
    grid_w = 2 * label_w + gap + (n + 1) * cell
    x0 = (W - grid_w) // 2
    top += (H - top - bottom - 3 * cell) // 2  # center vertically when width-limited
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((x0, top - 52), title, fill=FG, font=_font(30))
    fw = d.textlength(frame_text, font=_font(30))
    d.text((x0 + grid_w - fw, top - 52), frame_text, fill=FG, font=_font(30))
    lab = _font(20)
    for r, (key, label) in enumerate(REF_ROWS):
        y = top + r * cell
        _vtext(img, label, (x0 + label_w // 2, y + cell // 2), lab, cell - 8)
        img.paste(_fit(ref_cells.get(key), cell), (x0 + label_w, y))
    xs = x0 + label_w + cell + gap // 2
    d.line([(xs, top), (xs, top + 3 * cell)], fill=LINE, width=2)
    gx = x0 + label_w + cell + gap
    for r, (rkey, rlabel) in enumerate(ROWS):
        y = top + r * cell
        _vtext(img, rlabel, (gx + label_w // 2, y + cell // 2), lab, cell - 8)
        for c, (ckey, _) in enumerate(columns):
            img.paste(_fit(result_cells.get((rkey, ckey)), cell), (gx + label_w + c * cell, y))
    for c, (_, caption) in enumerate(columns):
        cx = gx + label_w + c * cell + cell // 2
        _rich(img, caption, cx, top + 3 * cell + 10, 20)
    return img


def video_frame(video, scenario, frame_idx):
    cfg = VIDEOS[video]
    d = CELLS_ROOT / video / scenario
    fn = f"frame_{frame_idx:06d}.png"
    ref = {"source_neutral": d / "ref" / "source_neutral.png", "target_neutral": d / "ref" / "target_neutral.png",
           "source_expression": d / "source_expression" / fn}
    res = {(r, c): d / c / r / fn for r, _ in ROWS for c, _ in cfg["columns"]}
    return compose_frame(ref, res, cfg["columns"], f"{cfg['title']}  |  {SCENARIO_TITLE[scenario]}",
                         f"frame {frame_idx}")


def previs(frame_idx=4300):
    out_dir = REPO_ROOT / "vis_CBD" / "supp_video" / "previs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for video, scenario in [("design_study", "mf_to_coma"), ("comparison", "mf_to_ict"),
                            ("loss_ablation", "mf_to_biwi")]:
        p = out_dir / f"{video}_{scenario}_frame_{frame_idx:06d}.png"
        video_frame(video, scenario, frame_idx).save(p)
        print("wrote", p)


if __name__ == "__main__":
    previs()
