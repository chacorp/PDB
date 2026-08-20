"""
eval_wcu.py -- Weight Column Usage

Measures how many cage weight columns (out of num_cage_v) are actually used
by the trained key_weight_model for a set of checkpoints, on the neutral
meshes of the 1 multiface test id + 10 ict test ids.

"Used" = at least one mesh vertex has weight > 0 for that cage column.
This is an exact criterion because last_activation=relu zeroes columns
exactly (see models/encoder.py LinearEncoder.forward: normalize -> relu).

Edit the CKPTS dict below to select which checkpoints/epochs to compare.

Usage:
    python eval_wcu.py
"""
import argparse

import numpy as np
import torch
import igl
import yaml

from eval_CBD import Trainer
from vis_key_weight import get_identity_meshes

CKPTS = {
    # label: (ckpt_dir, target_epoch, num_cage_v)
    # -- use_data2, num_cage_v=512, cage-consist+cyclic both on, 1000-epoch complete;
    #    lambda_cage_consist/lambda_cyclic weight comparison --
    "0.5/0.5 (default)": ("ckpts_CBD7/2026-08-06-00-00-28-NGBCv5", 1000, 512),
    "0.5/0.5 (explicit)": ("ckpts_CBD7/2026-08-07-02-48-53-NGBCv5", 1000, 512),
    "0.1/0.2 (a)": ("ckpts_CBD7/2026-08-11-21-15-27-NGBCv5", 1000, 512),
    "0.1/0.2 (b)": ("ckpts_CBD7/2026-08-11-23-05-25-NGBCv5", 1000, 512),
}


def build_trainer(ckpt_dir, target_epoch):
    config = f"{ckpt_dir}/train_opts.yml"
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    opts = argparse.Namespace(**opts_yaml)

    opts.ckpt = ckpt_dir
    opts.continue_ckpt = True
    opts.start_epoch = target_epoch
    opts.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    opts.tb = False

    return Trainer(opts)


def count_used_columns(trainer, verts, faces):
    normals = igl.per_vertex_normals(verts.astype(np.float64), faces).astype(np.float32)
    v_th = torch.from_numpy(verts)[None].float().to(trainer.device)
    n_th = torch.from_numpy(normals)[None].float().to(trainer.device)
    with torch.no_grad():
        key_weight = trainer.model.predict_coordinate(v_th, n_th)  # [1, N, M]
    used = (key_weight[0] > 0).any(dim=0)  # [M]
    return int(used.sum().item()), key_weight.shape[-1]


def main():
    identities = []
    for id_name, verts, faces in get_identity_meshes("multiface", "test"):
        identities.append(("multiface", id_name, verts, faces))
    for id_name, verts, faces in get_identity_meshes("ict", "test"):
        identities.append(("ict", id_name, verts, faces))

    results = {}
    for label, (ckpt_dir, target_epoch, num_cage_v) in CKPTS.items():
        print(f"Loading {label} (epoch {target_epoch}): {ckpt_dir}")
        trainer = build_trainer(ckpt_dir, target_epoch)
        row = []
        for dataset, id_name, verts, faces in identities:
            used, total = count_used_columns(trainer, verts, faces)
            assert total == num_cage_v
            row.append((dataset, id_name, used))
            print(f"  [{dataset}/{id_name}] used={used}/{total}")
        results[label] = row
        del trainer
        torch.cuda.empty_cache()

    labels = list(CKPTS.keys())
    header = "| dataset | id | " + " | ".join(f"used ({l})" for l in labels) + " |"
    sep = "|" + "---|" * (2 + len(labels))
    lines = [header, sep]
    used_lists = {l: [] for l in labels}
    for i, (dataset, id_name, _, _) in enumerate(identities):
        row_vals = [results[l][i][2] for l in labels]
        for l, v in zip(labels, row_vals):
            used_lists[l].append(v)
        lines.append(f"| {dataset} | {id_name} | " + " | ".join(str(v) for v in row_vals) + " |")
    avg_vals = [f"{np.mean(used_lists[l]):.2f}" for l in labels]
    lines.append(f"| **average** | ({len(identities)} ids) | " + " | ".join(avg_vals) + " |")

    print("\n".join(lines))
    with open("_tmp/eval_wcu_result.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
