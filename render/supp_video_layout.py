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

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
CELLS_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "cells"
W, H = 1920, 1080
BG, FG, LINE = (0, 0, 0), (255, 255, 255), (90, 90, 90)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
REF_ROWS = [("source_neutral", "Source Neutral"), ("source_expression", "Source Expression"),
            ("target_neutral", "Target Neutral")]
ROWS = [("self", "Self"), ("cross", "Cross"), ("cyclic", "Cyclic")]

VIDEOS = {
    "design_study": dict(title="Design study", columns=[
        ("ours", "Ours (relu)"), ("no_activation", "no activation"), ("softplus", "softplus"), ("elu", "elu"),
        ("soft_pou", "soft POU"), ("wo_mask", "w/o mask"), ("eq15_delta", "delta"), ("eq16_matrix", "matrix")]),
    "comparison": dict(title="Comparison", columns=[("NC", "NC"), ("NFR", "NFR"), ("NFS", "NFS"), ("PDB", "PDB")]),
    "loss_ablation": dict(title="Loss ablation", columns=[
        ("baseline", "Ours"), ("w_consist", "w/ consist"), ("w_cyclic", "w/ cyclic"),
        ("w_both", "w/ consist + cyclic")]),
}
SCENARIO_TITLE = {"mf_to_coma": "mf -> coma", "mf_to_biwi": "mf -> biwi", "mf_to_ict": "mf -> ict",
                  "ict_to_mf_m00": "ict m00 -> mf", "ict_to_mf_m02": "ict m02 -> mf"}


def _font(size):
    return ImageFont.truetype(FONT, size)


def _fit(path, size):
    canvas = Image.new("RGB", (size, size), BG)
    if path is None or not Path(path).exists():
        return canvas
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def _vtext(img, text, xy_center, font):
    """text rotated 90 degrees (reads bottom to top), centered at xy_center"""
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
        _vtext(img, label, (x0 + label_w // 2, y + cell // 2), lab)
        img.paste(_fit(ref_cells.get(key), cell), (x0 + label_w, y))
    xs = x0 + label_w + cell + gap // 2
    d.line([(xs, top), (xs, top + 3 * cell)], fill=LINE, width=2)
    gx = x0 + label_w + cell + gap
    for r, (rkey, rlabel) in enumerate(ROWS):
        y = top + r * cell
        _vtext(img, rlabel, (gx + label_w // 2, y + cell // 2), lab)
        for c, (ckey, _) in enumerate(columns):
            img.paste(_fit(result_cells.get((rkey, ckey)), cell), (gx + label_w + c * cell, y))
    for c, (_, caption) in enumerate(columns):
        cx = gx + label_w + c * cell + cell // 2
        tw = d.textlength(caption, font=lab)
        d.text((cx - tw / 2, top + 3 * cell + 12), caption, fill=FG, font=lab)
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
