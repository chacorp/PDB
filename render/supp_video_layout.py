"""
Supplementary video, stage 3: compose a 1920x1080 frame (black background) from the
per-output cells of render/supp_video_render.py, same arrangement as the HTML figures:
  left  : Source Neutral / Source Expression / Target Neutral (one column)
  right : rows = Self / Cross / Cyclic retargeting, columns = models
  top   : video title + scenario, animation frame number (dataloader_CBD index)
  bottom: column captions

    python render/supp_video_layout.py            # pre-vis frames (one per video)
    python render/supp_video_layout.py comparison # all frames + mp4 for a video
stylized / ttt: cells of render/userstudy_render.py (vis_CBD/userstudy, scenario
"{source}_to_{target}"), rows Cross / Cyclic only.
"""
from pathlib import Path

import re
import html

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
CELLS_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "cells"
USERSTUDY_ROOT = REPO_ROOT / "vis_CBD" / "userstudy"  # cells/ and videos/ref/ of userstudy_render.py
W, H = 1920, 1080
BG, FG, LINE = (0, 0, 0), (255, 255, 255), (90, 90, 90)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # has glyphs Serif lacks (e.g. U+2112)
# labels / captions exactly as in the HTML reports (HTML snippets, drawn by _rich)
REF_ROWS = [("source_neutral", "Source Neutral Mesh"), ("source_expression", "Source Expression Mesh"),
            ("target_neutral", "Target Neutral Mesh")]
ROWS = [("self", "Self retargeting"), ("cross", "Cross retargeting"), ("cyclic", "Cyclic retargeting")]
CELL_ASPECT = 0.7  # displayed cell width / height (left / right margins of each render trimmed)

VIDEOS = {
    # build_design_study_figure_gallery.COLUMNS with capitalization / spacing fixed (video only)
    "design_study": dict(title="Design study", columns=[
        ("ours", "Ours<sup>&dagger;</sup>"), ("no_activation", "No activation"), ("softplus", "w/ softplus"),
        ("elu", "w/ ELU"), ("soft_pou", "w/ soft POU"), ("wo_mask", "w/o Mask"), ("eq15_delta", "Equation 15"),
        ("eq16_matrix", "Equation 16")]),
    # vis_comparison.html
    "comparison": dict(title="Comparison", columns=[("NC", "NC"), ("NFR", "NFR"), ("NFS", "NFS"), ("PDB", "PDB")]),
    # build_ablation_gallery_mask.FIG_COLUMNS
    "loss_ablation": dict(title="Loss ablation", columns=[
        ("baseline", "Ours<sup>&dagger;</sup>"),
        ("w_consist", "w/ <i>&#8466;</i><sub>consist</sub>"),
        ("w_cyclic", "w/ <i>&#8466;</i><sub>cyclic</sub>"),
        ("w_both", "w/ <i>&#8466;</i><sub>consist</sub> + <i>&#8466;</i><sub>cyclic</sub>")]),
    # user study stylized targets (same PDB as vis_comparison_stylize_retarget.html)
    "stylized": dict(title="Retargeting on stylized meshes", columns=[("NC", "NC"), ("NFR", "NFR"), ("NFS", "NFS"), ("PDB", "PDB")],
                     rows=ROWS[1:], cells="userstudy"),
    # the same targets, PDB before / after test-time training on the target
    "ttt": dict(title="Test-time training", columns=[("PDB", "PDB"), ("PDB-TTT", "PDB-TTT")],
                rows=ROWS[1:], cells="userstudy"),
}
SCENARIO_TITLE = {"mf_to_coma": "mf -> coma", "coma_to_biwi": "coma -> biwi", "mf_to_biwi": "mf -> biwi", "mf_to_ict": "mf -> ict",
                  "ict_to_mf_m00": "ict m00 -> mf", "ict_to_mf_m02": "ict m02 -> mf"}


def _font(size, path=FONT):
    return ImageFont.truetype(path, size)


def _has_glyphs(font, text):
    notdef = font.getmask("\ue000")
    return all(bytes(font.getmask(ch)) != bytes(notdef) or ch.isspace() for ch in text)


def _font_for(text, size):
    f = _font(size)
    return f if _has_glyphs(f, text) else _font(size, FONT_FALLBACK)


def _fit(path, w, h):
    """square cell image -> center-cropped to the w:h aspect (trims the empty left/right
    margins, like the HTML figure's .clip) and resized to w x h"""
    if path is None or not Path(path).exists():
        return Image.new("RGB", (w, h), BG)
    im = Image.open(path).convert("RGB")
    cw = min(im.width, round(im.height * w / h))
    x0 = (im.width - cw) // 2
    return im.crop((x0, 0, x0 + cw, im.height)).resize((w, h), Image.LANCZOS)


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


def compose_frame(ref_cells, result_cells, columns, header, frame_text, rows=ROWS):
    """ref_cells {ref key: png}, result_cells {(row, column key): png}, columns [(key, caption)],
    rows [(row key, label)]: fewer than three are centered vertically next to the reference column"""
    top, bottom, label_w, gap = 44, 38, 30, 60  # gap: reference column | method outputs
    row_gap = 24                                 # vertical gap between the three rows
    n = len(columns)
    # cell height: as tall as the screen allows, unless the columns run out of width
    ch = min((H - top - bottom - 2 * row_gap) // 3, int((W - 2 * label_w - gap - 8) / ((n + 1) * CELL_ASPECT)))
    cw = int(ch * CELL_ASPECT)
    grid_w = 2 * label_w + gap + (n + 1) * cw
    grid_h = 3 * ch + 2 * row_gap
    x0 = (W - grid_w) // 2
    top += (H - top - bottom - grid_h) // 2  # center vertically when width-limited
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    hf = _font(26)
    if header:
        d.text((x0, top - 38), header, fill=FG, font=hf)
    d.text((x0 + grid_w - d.textlength(frame_text, font=hf), top - 38), frame_text, fill=FG, font=hf)
    lab = _font(20)
    for r, (key, label) in enumerate(REF_ROWS):
        y = top + r * (ch + row_gap)
        _vtext(img, label, (x0 + label_w // 2, y + ch // 2), lab, ch - 8)
        img.paste(_fit(ref_cells.get(key), cw, ch), (x0 + label_w, y))
    xs = x0 + label_w + cw + gap // 2
    d.line([(xs, top), (xs, top + grid_h)], fill=LINE, width=2)
    gx = x0 + label_w + cw + gap
    rows_h = len(rows) * ch + (len(rows) - 1) * row_gap
    ry = top + (grid_h - rows_h) // 2
    for r, (rkey, rlabel) in enumerate(rows):
        y = ry + r * (ch + row_gap)
        _vtext(img, rlabel, (gx + label_w // 2, y + ch // 2), lab, ch - 8)
        for c, (ckey, _) in enumerate(columns):
            img.paste(_fit(result_cells.get((rkey, ckey)), cw, ch), (gx + label_w + c * cw, y))
    for c, (_, caption) in enumerate(columns):
        _rich(img, caption, gx + label_w + c * cw + cw // 2, ry + rows_h + 8, 20)
    return img


def video_frame(video, scenario, frame_idx):
    cfg = VIDEOS[video]
    rows = cfg.get("rows", ROWS)
    fn = f"frame_{frame_idx:06d}.png"
    if cfg.get("cells") == "userstudy":
        src, tgt = scenario.split("_to_")
        nrm, cells = USERSTUDY_ROOT / "videos" / "ref", USERSTUDY_ROOT / "cells" / src
        ref = {"source_neutral": nrm / f"{src}_neutral.png", "target_neutral": nrm / f"{tgt}_neutral.png",
               "source_expression": cells / "ref" / "source_expression" / fn}
        d = cells / tgt
    else:
        d = CELLS_ROOT / video / scenario
        ref = {"source_neutral": d / "ref" / "source_neutral.png", "target_neutral": d / "ref" / "target_neutral.png",
               "source_expression": d / "source_expression" / fn}
    res = {(r, c): d / c / r / fn for r, _ in rows for c, _ in cfg["columns"]}
    # no video title / scenario text on screen; only the animation frame number
    return compose_frame(ref, res, cfg["columns"], "", f"frame {frame_idx}", rows)


def previs(frame_idx=4300):
    out_dir = REPO_ROOT / "vis_CBD" / "supp_video" / "previs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for video, scenario in [("design_study", "mf_to_coma"), ("comparison", "mf_to_ict"),
                            ("loss_ablation", "mf_to_biwi")]:
        p = out_dir / f"{video}_{scenario}_frame_{frame_idx:06d}.png"
        video_frame(video, scenario, frame_idx).save(p)
        print("wrote", p)


from supp_video_frames import VIDEO_FRAMES  # scenario order and frames per video


def make_video(video, fps=30):
    """compose every frame (scenarios in order, frames in dataloader order) and encode the mp4"""
    import subprocess
    out_dir = REPO_ROOT / "vis_CBD" / "supp_video" / "frames" / video
    out_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for scenario, frames in VIDEO_FRAMES[video].items():
        for f in frames:
            p = out_dir / f"{k:06d}.png"
            if not p.exists():
                video_frame(video, scenario, f).save(p)
            k += 1
        print(f"{video}/{scenario}: {len(frames)} frames", flush=True)
    mp4 = REPO_ROOT / "vis_CBD" / "supp_video" / f"{video}.mp4"
    # system ffmpeg has libx264 (the conda one does not)
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
    subprocess.run([ffmpeg, "-y", "-framerate", str(fps), "-i", str(out_dir / "%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", mp4, f"({k} frames)", flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:]:
        for v in sys.argv[1:]:
            make_video(v)
    else:
        previs()
