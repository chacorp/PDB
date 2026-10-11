"""Build vis_CBD/vis_comparison/vis_comparison-2.html: the vis_comparison.html
qualitative figure (no metrics section), with the "PDB" column images taken from
the continued PDB run (ckpts_CBD8/2026-10-08-22-31-39-NGBCv5, cells rendered into
figure_continue/ by COMPARISON_SET=continue render_comparison_figure.py). The
column caption and page text keep the name "PDB". Each diff heatmap is normalized
on its own, so cells from the two renders are directly interchangeable.
"""
from pathlib import Path

import build_comparison_figure_html as fig_mod
from comparison_frames import MF_FRAMES, ICT_FRAMES_M00, ICT_FRAMES_M02

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "figure"
CONT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "figure_continue"
OUT_HTML = REPO_ROOT / "vis_CBD" / "vis_comparison" / "vis_comparison-2.html"
COLUMNS = ["NC", "NFS", "NFR", "PDB"]
COL_SRC = {"PDB": (CONT_ROOT, "PDB-cont")}

SCENARIOS = [
    ("mf_to_ict", MF_FRAMES),
    ("ict_to_mf_m00", ICT_FRAMES_M00),
    ("ict_to_mf_m02", ICT_FRAMES_M02),
    ("mf_to_coma", MF_FRAMES),
    ("mf_to_biwi", MF_FRAMES),
    ("ict_to_coma_m00", ICT_FRAMES_M00),
    ("ict_to_coma_m02", ICT_FRAMES_M02),
    ("ict_to_biwi_m00", ICT_FRAMES_M00),
    ("ict_to_biwi_m02", ICT_FRAMES_M02),
]

CSS = fig_mod.CSS.replace("background:#111; color:#eee;", "background:#fff; color:#111;") \
                 .replace("nav { position:sticky; top:0; background:#111;", "nav { position:sticky; top:0; background:#fff;")


def main():
    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>comparison: NC / NFS / NFR / PDB (mf / ict / coma / biwi)</title>
<style>{CSS}</style></head><body>
<h1>comparison baselines (NC / NFS / NFR / PDB): self / cross / cyclic retargeting</h1>
"""]
    parts.append('<section class="figure-card figure-section"><h2 id="figure">Qualitative figure</h2>'
                 '<p class="legend">Two tables per (scenario, frame). Left: Source Neutral Mesh / Source Expression '
                 'Mesh (real captured GT deformation) / Target Neutral Mesh. Right: rows = self / cross / cyclic '
                 'retargeting, columns = NC / NFS / NFR / PDB. Diff-heatmap colors (YlOrRd) show where each output '
                 'deforms the mesh: every cell is normalized on its own, against that row&#39;s diff_base (source '
                 'template for self/cyclic, target template for cross).</p>')
    parts.append('<nav>' + "".join(f'<a href="#{k}">{k}</a>' for k, _ in SCENARIOS) + '</nav>')
    for key, frames in SCENARIOS:
        parts.append(f'<h2 id="{key}">{fig_mod.TITLES[key]}</h2>')
        for fr in frames:
            parts.append(f"<h3>frame {fr}</h3>")
            parts.append(fig_mod.build_frame(key, fr, root=FIG_ROOT, columns=COLUMNS, col_src=COL_SRC))
    parts.append("</section></body></html>")
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
