"""Uniform column norm vs area-weighted normalization in key_weight_model
-> vis_CBD/vis_comparison/vis_area_norm.html

Uniform (trained model):  id_in = mean_i f_i          col_norm = sqrt(sum_i x_ik^2)
Area-weighted          :  id_in = sum_i a_i f_i       col_norm = sqrt(sum_i a_i x_ik^2)
(a_i = barycentric vertex area / total area; see eval_CBD_robust.area_key_weight)

Per dataset (ICT ID0, COMA 00138) and condition (clean, dec50, dec25, subdiv1):
    area     : relative vertex weight a_i * N (uniform = 1 everywhere), log2 scale
    dens_u/a : column-norm contribution per unit surface area for one cage point
               uniform: x_i^2 / sum x^2 / a_i      (= x^2 x vertex density)
               area   : x_i^2 / sum a x^2          (density independent)
    kw_<m>_<k>: key_weight column k (two columns) for method m
    kwerr_<m>: sum_k |kw - P kw_clean| per vertex
    self_<m> : self retargeting output, colored by |self - GT|
plus histograms of per-column norm ratio (pert / clean, divided by its mean) and
self MSE over the first N frames.

Run from repo root:  python render/vis_area_norm.py
"""
import sys
import json
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "render"))
import cache_robust_figure as CF  # noqa: E402  (chdir, matplotrender import hack)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import eval_CBD_robust as R  # noqa: E402

OUT_DIR = REPO_ROOT / "vis_CBD" / "vis_comparison" / "area_norm_figure"
OUT_HTML = REPO_ROOT / "vis_CBD" / "vis_comparison" / "vis_area_norm.html"
CONDS = ["clean", "dec50", "dec25", "subdiv1"]
METHODS = [("uni", "균일 col_norm (학습된 방식)"), ("area", "면적가중 정규화")]
DATASETS = [("ict", "ICT (ID000)"), ("coma", "COMA (FaceTalk_170809_00138_TA)")]
N_MSE_FRAMES = 128
C_UNI, C_AREA = "#2a78d6", "#eb6834"


# ---------------------------------------------------------------------------
# rendering with an arbitrary colormap
# ---------------------------------------------------------------------------
def render(V, F, out_path, cnorm, cmap="YlOrRd"):
    """cnorm in [0,1] per vertex, blended over gray shading (same look as CF.save_cell)."""
    Vd, ratio = CF.fit_view(V)
    vn = CF.calc_face_norm(np.asarray(V) * CF.M_SCALE, F, mode="v")
    shade = np.clip(vn @ CF.LIGHT_DIR, 0, 1)[..., None].repeat(3, -1) * 0.6 + 0.3
    col = plt.get_cmap(cmap)(np.clip(cnorm, 0, 1))[..., :3]
    m = 0.85 if cmap != "YlOrRd" else np.clip(cnorm, 0, 1)[:, None]
    C = shade * (1 - m) + col * m
    CF.plot_mesh_gouraud([Vd], [F], Cs=[C], is_color=True, mesh_scale=1.0, mesh_trans=np.zeros(3),
                         backface_culling=True, rot_list=[[0, 0, 0]], size=CF.SIZE, mode="shade",
                         bg_black=False, logdir=str(out_path.parent), name=out_path.stem, save=True, show=False)
    plt.close("all")


def colorbar(path, cmap, lo, hi, label, ticks=None, ticklabels=None):
    fig, ax = plt.subplots(figsize=(2.6, 0.42), dpi=150)
    fig.subplots_adjust(bottom=0.55, top=0.95, left=0.06, right=0.94)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cmap), cax=ax,
                      orientation="horizontal")
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
@torch.no_grad()
def raw_out(model, C):
    """key_weight_model output right before col_norm (uniform id_in or area id_in)."""
    m = model.key_weight_model
    x = torch.cat([C["neu"], C["neu_norm"]], -1)
    return m.forward_func(x)[0][0]


@torch.no_grad()
def raw_out_area(model, C, w):
    m = model.key_weight_model
    x = torch.cat([C["neu"], C["neu_norm"]], -1)
    out = m.layer_in(x)
    id_in = (m.adain_in(x) * w).sum(-2, keepdim=True) + (out * w).sum(-2, keepdim=True)
    return m.forward_func(x, id_in=id_in)[0][0]


def run_dataset(key):
    sys_argv = sys.argv
    sys.argv = ["x", "--dataset", key, "--num_workers", "4"] + (["--ict_ids", "0"] if key == "ict" else [])
    opts = R.get_opts()
    sys.argv = sys_argv
    ev = R.RobustEval(opts)
    ev.idents, seqs, hole_ref = R.build_sequences(opts)
    V_ref = torch.from_numpy(ev.idents[hole_ref][0]).float()[None]
    ev.hole_candidates = np.nonzero(R.plateau_hat_points(V_ref, **R.SELF_MASK)[0, :, 0].numpy() > 0.999)[0]
    seq = [s for s in seqs if s["name"] == CF.REP_SEQ[key]][0]
    rob_meta = json.loads((CF.OUT_ROOT / key / "meta.json").read_text())
    frame = rob_meta["frames"][0]
    out = OUT_DIR / key
    out.mkdir(parents=True, exist_ok=True)

    ctx, kw, w, raw = {}, {}, {}, {}
    for c in CONDS:
        C = ev.ctx(seq["src"], c)
        ctx[c] = C
        w[c] = R.vertex_area_weights(C["neu"], C["faces"])
        kw[("uni", c)] = C["kw"]
        kw[("area", c)] = R.area_key_weight(ev.model, C["neu"], C["neu_norm"], C["faces"])
        raw[("uni", c)] = raw_out(ev.model, C)
        raw[("area", c)] = raw_out_area(ev.model, C, w[c])
    C0 = ctx["clean"]

    # ----- per-column norm ratio + kw error -------------------------------------------------
    stats = {"frame": frame, "seq": seq["name"], "n_vert": {c: int(ctx[c]["n_vert"]) for c in CONDS},
             "ratio": {}, "ratio_cv": {}, "ratio_mean": {}, "kw_relerr": {}, "self_mse": {}}
    for mth, _ in METHODS:
        wt = (lambda c: torch.ones_like(w[c])) if mth == "uni" else (lambda c: w[c])
        n0 = (raw[(mth, "clean")].square() * wt("clean")[0]).sum(0).sqrt()
        for c in CONDS[1:]:
            n = (raw[(mth, c)].square() * wt(c)[0]).sum(0).sqrt()
            r = (n / n0).cpu().numpy()
            stats["ratio"][f"{mth}_{c}"] = (r / r.mean()).tolist()
            stats["ratio_cv"][f"{mth}_{c}"] = float(r.std() / r.mean())
            stats["ratio_mean"][f"{mth}_{c}"] = float(r.mean())
        for c in CONDS:
            sub = torch.sparse.mm(ctx[c]["P"], kw[(mth, "clean")][0])
            stats["kw_relerr"][f"{mth}_{c}"] = float((kw[(mth, c)][0] - sub).norm() / sub.norm())

    # ----- self MSE over first N frames ------------------------------------------------------
    frames_all = seq["frames"][:N_MSE_FRAMES]
    for mth, _ in METHODS:
        for c in CONDS:
            Cm = dict(ctx[c], kw=kw[(mth, c)])
            s = 0.0
            for st in range(0, len(frames_all), 16):
                gt = ev.apply_P(Cm["P"], frames_all[st:st + 16].to(ev.device))
                o, _ = ev.encode(Cm, gt)
                s += ((o - gt) ** 2).mean(dim=(1, 2)).sum().item()
            stats["self_mse"][f"{mth}_{c}"] = s / len(frames_all)

    # ----- choose 2 cage points ---------------------------------------------------------------
    face = torch.zeros(C0["n_vert"], dtype=torch.bool, device=ev.device)
    face[torch.from_numpy(ev.hole_candidates).to(ev.device)] = True
    k_face = int(kw[("uni", "clean")][0][face].sum(0).argmax())
    r25 = np.array(stats["ratio"]["uni_dec25"])
    k_dev = int(np.abs(r25 - 1).argmax())
    if k_dev == k_face:
        k_dev = int(np.argsort(-np.abs(r25 - 1))[1])
    stats["cols"] = [k_face, k_dev]

    # ----- render -----------------------------------------------------------------------------
    gt0 = seq["frames"][frame:frame + 1].to(ev.device)
    kw_vmax = {(m, k): float(np.percentile(kw[(m, "clean")][0, :, k].cpu().numpy(), 99.5)) + 1e-12
               for m, _ in METHODS for k in stats["cols"]}
    self_err, kwerr = {}, {}
    for mth, _ in METHODS:
        for c in CONDS:
            Cm = dict(ctx[c], kw=kw[(mth, c)])
            gt = ev.apply_P(Cm["P"], gt0)
            o, _ = ev.encode(Cm, gt)
            self_err[(mth, c)] = (o[0], (o - gt)[0].norm(dim=-1).cpu().numpy())
            sub = torch.sparse.mm(ctx[c]["P"], kw[(mth, "clean")][0])
            kwerr[(mth, c)] = (kw[(mth, c)][0] - sub).abs().sum(-1).cpu().numpy()
    se_vmax = 3 * float(np.percentile(self_err[("uni", "clean")][1], 99))
    ke_vmax = float(max(np.percentile(kwerr[("uni", "dec25")], 99), 1e-6))
    stats["vmax"] = dict(self_err=se_vmax, kwerr=ke_vmax, kw={f"{m}_{k}": v for (m, k), v in kw_vmax.items()})

    dens = {}
    for c in CONDS:
        a = w[c][0, :, 0]
        for mth, _ in METHODS:
            x2 = raw[(mth, c)][:, stats["cols"][0]].square()
            if mth == "uni":
                d = x2 / x2.sum() / a           # uniform: share per unit area
            else:
                d = x2 / (a * x2).sum()         # area: share per unit area
            dens[(mth, c)] = d.cpu().numpy()
    ld = np.log10(np.concatenate([dens[("uni", "clean")], dens[("area", "clean")]]) + 1e-12)
    dens_lo, dens_hi = float(np.percentile(ld, 5)), float(np.percentile(ld, 99.5))
    stats["vmax"]["dens_log10"] = [dens_lo, dens_hi]

    for c in CONDS:
        Vn = ctx[c]["neu"][0].cpu().numpy()
        Fc = ctx[c]["faces"].cpu().numpy()
        aN = (w[c][0, :, 0] * ctx[c]["n_vert"]).cpu().numpy()
        render(Vn, Fc, out / f"area_{c}.png", (np.log2(aN) + 3) / 6, cmap="RdBu_r")
        for mth, _ in METHODS:
            render(Vn, Fc, out / f"dens_{mth}_{c}.png",
                   (np.log10(dens[(mth, c)] + 1e-12) - dens_lo) / (dens_hi - dens_lo), cmap="viridis")
            for k in stats["cols"]:
                render(Vn, Fc, out / f"kw_{mth}_{k}_{c}.png", kw[(mth, c)][0, :, k].cpu().numpy() / kw_vmax[(mth, k)])
            render(Vn, Fc, out / f"kwerr_{mth}_{c}.png", kwerr[(mth, c)] / ke_vmax)
            o, e = self_err[(mth, c)]
            render(o.cpu().numpy(), Fc, out / f"self_{mth}_{c}.png", e / se_vmax)
        print(f"[{key}] {c} rendered")
    colorbar(out / "cb_dens.png", "viridis", dens_lo, dens_hi, "log10 (norm share / unit area)")
    colorbar(out / "cb_area.png", "RdBu_r", -3, 3, "a_i · N  (1 = same as uniform)",
             ticks=[-3, -2, -1, 0, 1, 2, 3], ticklabels=["1/8", "1/4", "1/2", "1", "2", "4", "8"])

    # ----- histogram chart --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), dpi=150, sharey=True)
    bins = np.linspace(0, 2.5, 51)
    for ax, c in zip(axes, CONDS[1:]):
        for mth, lab, col in [("uni", "uniform col_norm", C_UNI), ("area", "area-weighted", C_AREA)]:
            r = np.array(stats["ratio"][f"{mth}_{c}"])
            ax.hist(np.clip(r, 0, 2.5), bins=bins, histtype="step", lw=2, color=col,
                    label=f"{lab} (CV {stats['ratio_cv'][f'{mth}_{c}']:.3f})")
        ax.axvline(1, color="#999", lw=1, ls="--")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("column-norm ratio (pert / clean) / mean", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("# cage points", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hist_ratio.png")
    plt.close(fig)

    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    del ev
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def img(path, cls="clip"):
    if not path.exists():
        return '<div class="missing">missing</div>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    tag = f'<div class="{cls}"><img src="data:image/png;base64,{b64}" loading="lazy"></div>'
    txt = path.with_suffix(".rescale.txt")
    if txt.exists():
        tag += f'<div class="rescale">&times;{txt.read_text().strip()}</div>'
    return tag


def fmt(v):
    return f"{v:.3g}"


def section(key, title):
    d = OUT_DIR / key
    st = json.loads((d / "stats.json").read_text())
    k = st["cols"][0]
    p = [f'<section id="{key}"><h2>{title} · cage point {k}</h2>', '<table class="grid"><tbody>']
    for m, label in METHODS:
        p.append(f'<tr><th class="rowhead">{label}</th>' +
                 "".join(f'<td class="imgcell">{img(d / f"kw_{m}_{k}_{c}.png")}</td>' for c in CONDS) + "</tr>")
    p.append('</tbody><tfoot><tr><th class="rowhead-col"></th>' +
             "".join(f"<th>{c} ({st['n_vert'][c]:,} 정점)</th>" for c in CONDS) + "</tr></tfoot></table></section>")
    return "\n".join(p)


def build_html():
    from build_vis_robustness_html import CSS
    secs = "\n".join(section(k, t) for k, t in DATASETS)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Area vs Column Norm</title><style>{CSS}</style></head><body>
<h1>key_weight: 균일 column norm vs 면적가중 정규화</h1>
<p class="legend">
<b>균일 column norm</b>(학습된 방식): 각 cage point 열을 √(Σ<sub>i</sub> x<sub>i</sub>²)로 나눔 — 정점을 개수로 셈.<br>
<b>면적가중 정규화</b>: √(Σ<sub>i</sub> a<sub>i</sub> x<sub>i</sub>²)로 나눔 (a<sub>i</sub> = 정점 면적 / 전체 면적) — 정점을 표면적으로 셈. AdaIN 입력의 정점 평균도 면적가중 평균으로 바꿈.<br>
색 = 해당 cage point의 key_weight, 범위는 행마다 clean의 99.5 백분위로 고정.
</p>
{secs}
</body></html>"""
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if "--html_only" not in sys.argv:
        for k, _ in DATASETS:
            run_dataset(k)
    build_html()
