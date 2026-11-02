"""
Assets for the train_CBD (version 5, NGBC) loss-flow diagram
(render/build_loss_flow_html.py).

One forward pass of the ckpts_CBD9 num_cage_v=64 best checkpoint on one mf_ROM
test frame (absolute idx 8880, same sample as the design_study figure), full mesh
(no train-time subsampling). Saves to vis_CBD/vis_loss_flow/assets/:
  - neutral.png / expression.png       inputs (expression: diff heatmap vs neutral)
  - t_mask.png                         plateau hat mask m on the neutral (side view: m < 1
                                       only on the neck / back of the head)
  - weight_col.png                     one column of W (cage point most tied to the
                                       max-displacement vertex)
  - weight_argmax.png                  argmax_m W (dominant cage point per vertex)
  - pred_def.png / pred_neu.png        outputs (same diff scale as expression.png)
  - cage.png                           cage_s / cage_d points only (front xy projection)
  - exp_z.png                          exp_z (128D) bar plot
  - values.json                        per-sample loss terms + tensor shapes

Run from repo root:
    python render/save_loss_flow_assets.py
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import render_design_study_figure as base  # noqa: E402
from eval_CBD import Trainer  # noqa: E402
from utils.loss_utils import non_ict_loss  # noqa: E402

CKPT = "./ckpts_CBD9/2026-10-30-00-43-11-NGBCv5"  # num_cage_v=64
FRAME_IDX = 8880
OUT_DIR = REPO_ROOT / "vis_CBD" / "vis_loss_flow" / "assets"


def colorize(V, F_, values, cmap):
    """Shaded base color blended with cmap(values), values in [0, 1]."""
    V_scaled = np.asarray(V) * base.M_SCALE + base.M_TRNS
    vn = base.calc_face_norm(V_scaled, F_, mode='v')
    shading = np.clip(vn @ base.LIGHT_DIR, 0, 1)
    vc = shading[..., None].repeat(3, axis=-1) * 0.6 + 0.3
    Sc = plt.get_cmap(cmap)(values)[..., :3]
    return vc * 0.35 + Sc * 0.65


def categorical_colors(V, F_, labels):
    V_scaled = np.asarray(V) * base.M_SCALE + base.M_TRNS
    vn = base.calc_face_norm(V_scaled, F_, mode='v')
    shading = np.clip(vn @ base.LIGHT_DIR, 0, 1)[..., None]
    rng = np.random.RandomState(0)
    palette = plt.get_cmap("tab20")(np.arange(20))[:, :3]
    perm = rng.permutation(labels.max() + 1) % 20
    Sc = palette[perm[labels]]
    return Sc * (0.55 + 0.45 * shading)


def save_side_mesh(V, F_, color, out_path, yaw=70):
    base.plot_mesh_gouraud(
        [V], [F_], Cs=[color], mesh_scale=base.M_SCALE, mesh_trans=base.M_TRNS,
        is_color=True, backface_culling=True, rot_list=[[0, yaw, 0]],
        size=base.SIZE, mode="shade", bg_black=False,
        logdir=str(out_path.parent), name=out_path.stem, save=True, show=False,
    )


def save_cage_plot(cage_s, cage_d, out_path):
    """cage_s / cage_d points only (front xy projection), cropped tight to the points."""
    fig, ax = plt.subplots(figsize=(2.4, 3.0), dpi=300)
    for s, d in zip(cage_s, cage_d):
        ax.plot([s[0], d[0]], [s[1], d[1]], c="#999999", lw=0.8, zorder=2)
    ax.scatter(cage_s[:, 0], cage_s[:, 1], s=18, c="#2b6cb0", linewidths=0, zorder=3)
    ax.scatter(cage_d[:, 0], cage_d[:, 1], s=18, c="#d1462f", linewidths=0, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)


def save_exp_z_plot(z, out_path):
    fig, ax = plt.subplots(figsize=(4.0, 1.4), dpi=200)
    idx = np.arange(len(z))
    ax.bar(idx[:53], z[:53], width=1.0, color="#2b6cb0", label="z[:53]")
    ax.bar(idx[53:], z[53:], width=1.0, color="#9aa5b1", label="z[53:]")
    ax.axhline(0, c="#444", lw=0.5)
    ax.set_xlim(-1, len(z))
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opts = base.ds_mod.build_opts(CKPT, {"align_latent": True, "last_activation": "relu"})
    trainer = Trainer(opts)
    model = trainer.model.eval()
    device = opts.device

    ds = base.ds_mod.dataloader_CBD.EvalDataset(data_name="mf_ROM", toggle=False)
    vertices, template, vertices_normal, template_normal, faces, _id_name, mesh_data = ds[FRAME_IDX]
    faces = np.asarray(faces)

    to_b = lambda t: torch.as_tensor(t).float()[None].to(device)  # noqa: E731
    with torch.no_grad():
        pred_def, _, _, exp_z, pred_neu, t_mask, W, cage_s, cage_d = model(
            to_b(template), to_b(vertices), to_b(template_normal), to_b(vertices_normal),
            torch.as_tensor(mesh_data), epoch=0,
        )
        v_src, v_def = to_b(template), to_b(vertices)
        m, inv_m = t_mask, 1.0 - t_mask
        recon_def = F.mse_loss(v_def * m, pred_def * m) + F.mse_loss(v_src * inv_m, pred_def * inv_m)
        recon_neu = F.mse_loss(v_src * m, pred_neu * m) + F.mse_loss(v_src * inv_m, pred_neu * inv_m)
        exp_z_loss = non_ict_loss(exp_z[..., :53]) + F.mse_loss(torch.zeros_like(exp_z[..., 53:]), exp_z[..., 53:])

    lam = {"recon-def": 1.0, "recon-neu": 1.0, "exp-z": 0.5}
    terms = {"recon-def": recon_def.item(), "recon-neu": recon_neu.item(), "exp-z": exp_z_loss.item()}
    values = {
        "ckpt": CKPT, "frame": FRAME_IDX,
        "N": int(template.shape[0]), "M": int(cage_s.shape[1]), "L": int(exp_z.shape[-1]),
        "loss": terms, "lambda": lam,
        "total": sum(terms[k] * lam[k] for k in terms),
    }
    json.dump(values, open(OUT_DIR / "values.json", "w"), indent=2)
    print(json.dumps(values, indent=2))

    V_src = np.asarray(template)
    V_def = np.asarray(vertices)
    P_def = pred_def[0].cpu().numpy()
    P_neu = pred_neu[0].cpu().numpy()
    mask = t_mask[0, :, 0].cpu().numpy() if t_mask.dim() == 3 else t_mask[0].cpu().numpy()
    W_np = W[0].cpu().numpy()

    base.save_plain_mesh(V_src, faces, OUT_DIR / "neutral.png")

    # expression / pred_def / pred_neu share one diff color scale vs. the neutral
    c_gt, c_def, c_neu = base.row_joint_colors([V_def, P_def, P_neu], faces, V_src)
    base.save_diff_mesh(V_def, faces, c_gt, OUT_DIR / "expression.png")
    base.save_diff_mesh(P_def, faces, c_def, OUT_DIR / "pred_def.png")
    base.save_diff_mesh(P_neu, faces, c_neu, OUT_DIR / "pred_neu.png")

    save_side_mesh(V_src, faces, colorize(V_src, faces, np.clip(mask, 0, 1) * 0.85, "Blues"),
                   OUT_DIR / "t_mask.png")

    v_star = int(np.linalg.norm(V_def - V_src, axis=-1).argmax())
    m_star = int(W_np[v_star].argmax())
    col = W_np[:, m_star] / (W_np[:, m_star].max() + 1e-12)
    base.save_diff_mesh(V_src, faces, colorize(V_src, faces, col, "YlOrRd"), OUT_DIR / "weight_col.png")
    base.save_diff_mesh(V_src, faces, categorical_colors(V_src, faces, W_np.argmax(-1)),
                        OUT_DIR / "weight_argmax.png")

    save_cage_plot(cage_s[0].cpu().numpy(), cage_d[0].cpu().numpy(), OUT_DIR / "cage.png")
    save_exp_z_plot(exp_z[0, 0].cpu().numpy(), OUT_DIR / "exp_z.png")
    print("ALL DONE:", OUT_DIR, "m* =", m_star, "v* =", v_star)


if __name__ == "__main__":
    main()
