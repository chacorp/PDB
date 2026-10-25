"""
Final supplementary video: ffmpeg title cards (black background, white Times New Roman,
centered, 0.5 s fade in / out) + the rendered parts, joined with the concat filter.

  paper title card -> weight_vis_combined.mp4 (fades out to black) -> "Method" card ->
  pdb_method_final.mp4 -> "Design Study" card -> design_study.mp4 -> "Loss Ablation" card ->
  loss_ablation.mp4 -> "Comparison" card -> comparison.mp4 -> "Retargeting on Stylized Meshes" card ->
  stylized.mp4 -> "Test-Time Training" card -> ttt.mp4 -> "Thank you" card
  (experiment videos fade in / out)

    python render/supp_final_video.py
"""
import subprocess
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parent.parent
FFMPEG, FFPROBE = "/usr/bin/ffmpeg", "/usr/bin/ffprobe"   # conda ffmpeg lacks libx264
FONT = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
OUT = ROOT / "vis_CBD" / "method_video" / "supplementary_final.mp4"
W, H, FPS = 1920, 1080, 30
CARD_SEC, CARD_FADE, VIDEO_FADE = 3.0, 0.5, 1.0
SUPP = ROOT / "vis_CBD" / "supp_video"
# (kind, title text or video path, fade in, fade out)
PARTS = [
    ("card", "PDB: Point-Based Deformation Blending for Facial Animation Retargeting"),
    ("video", ROOT / "video_mi" / "weight_vis_combined.mp4", False, True),
    ("card", "Method"),
    ("video", ROOT / "vis_CBD/method_video/manim/videos/method_video/1080p30/pdb_method_final.mp4", False, False),
    ("card", "Design Study"),
    ("video", SUPP / "design_study.mp4", True, True),
    ("card", "Loss Ablation"),
    ("video", SUPP / "loss_ablation.mp4", True, True),
    ("card", "Comparison"),
    ("video", SUPP / "comparison.mp4", True, True),
    ("card", "Retargeting on Stylized Meshes"),
    ("video", SUPP / "stylized.mp4", True, True),
    ("card", "Test-Time Training"),
    ("video", SUPP / "ttt.mp4", True, True),
    ("card", "Thank you"),
]


def duration(path):
    out = subprocess.check_output([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", str(path)])
    return float(out)


def font_size(text, max_w=0.85 * W, max_size=80):
    """largest size <= max_size whose text width fits max_w"""
    size = max_size
    while ImageFont.truetype(FONT, size).getlength(text) > max_w:
        size -= 2
    return size


def main():
    inputs, chains, labels = [], [], []
    for i, part in enumerate(PARTS):
        if part[0] == "card":
            text = part[1].replace(":", r"\:")
            inputs += ["-f", "lavfi", "-t", str(CARD_SEC), "-i", f"color=c=black:s={W}x{H}:r={FPS}"]
            chains.append(
                f"[{i}:v]drawtext=fontfile={FONT}:text='{text}':fontcolor=white:"
                f"fontsize={font_size(part[1])}:x=(w-text_w)/2:y=(h-text_h)/2,"
                f"fade=t=in:st=0:d={CARD_FADE},fade=t=out:st={CARD_SEC - CARD_FADE}:d={CARD_FADE},"
                f"format=yuv420p,setsar=1,setpts=PTS-STARTPTS[p{i}]")
        else:
            _, path, fin, fout = part
            inputs += ["-i", str(path)]
            f = f"[{i}:v]fps={FPS},scale={W}:{H},setsar=1,setpts=PTS-STARTPTS"
            if fin:
                f += f",fade=t=in:st=0:d={VIDEO_FADE}"
            if fout:
                f += f",fade=t=out:st={duration(path) - VIDEO_FADE:.3f}:d={VIDEO_FADE}"
            chains.append(f + f",format=yuv420p[p{i}]")
        labels.append(f"[p{i}]")
    graph = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(PARTS)}:v=1:a=0[v]"
    cmd = [FFMPEG, "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), str(OUT)]
    subprocess.run(cmd, check=True)
    print("wrote", OUT, f"{duration(OUT):.2f} s")


if __name__ == "__main__":
    main()
