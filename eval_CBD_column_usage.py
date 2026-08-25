"""Control-point weight-column utilization diagnostics for NGBC/PDF.

This script separates two properties that should not be conflated:

1. local support: how many control points a single target vertex blends; and
2. global utilization: how many control-point columns are used across a face.

It reuses the checkpoint and dataset loading code from ``eval_CBD_cif.py``.
Neutral meshes may have different vertex counts and topology.  Column masses
are computed per identity with surface-area weighting and are averaged only
afterwards, so a high-resolution identity does not dominate the result.

The optional functional analysis uses the 53 shared ICT-FaceKit blendshapes.
It measures whether a weight column is coupled to an expression-dependent
control-point motion, and can additionally compute column-pruning curves.

Examples
--------
Evaluate one checkpoint at its best checkpoint file::

    python eval_CBD_column_usage.py \
        --ckpt ckpts_CBD7/2026-08-03-09-09-59-NGBCv5 \
        --datasets ict mf_SEN mf_ROM --n_ict_identity 10

Compare four epoch-200 checkpoints::

    python eval_CBD_column_usage.py \
        --ckpt <baseline> <cage> <cyclic> <cage+cyclic> \
        --epoch 200 --datasets ict mf_SEN mf_ROM

Skip the ICT functional/pruning analysis for a faster neutral-only run::

    python eval_CBD_column_usage.py --ckpt <checkpoint> --skip_functional
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from eval_CBD_cif import Trainer as CIFTrainer
from eval_CBD_cif import to_jsonable
from eval_CBD_gc import GCEvalDataset, GC_collate_wrapper, vertex_area_weights


EPS = 1e-12


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global and local control-point weight-column utilization diagnostics"
    )
    parser.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoint folders")
    parser.add_argument("--epoch", type=int, default=None,
                        help="load model_<epoch>.pth instead of the checkpoint's *_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="eval_CBD_column_usage")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--datasets", nargs="+", default=["ict", "mf_SEN", "mf_ROM"])
    parser.add_argument("--n_ict_identity", type=int, default=10)
    parser.add_argument("--n_id_per_dataset", type=int, default=1)

    parser.add_argument(
        "--active_mass_ratios", type=float, nargs="+",
        default=[0.0, 0.01, 0.1, 0.5, 1.0],
        help="active if column mass > ratio * mean column mass; 0 uses exact nonzero support",
    )
    parser.add_argument("--mass_coverage", type=float, default=0.95)

    parser.add_argument("--skip_functional", action="store_true",
                        help="skip ICT blendshape motion-weighted usage and pruning curves")
    parser.add_argument("--n_functional_ict", type=int, default=5)
    parser.add_argument("--skip_pruning", action="store_true")
    parser.add_argument("--pruning_q", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256, 512])

    # Normally restored from train_opts.yml.  Explicit overrides are useful
    # for older checkpoints whose saved evaluation flags were incomplete.
    parser.add_argument("--last_activation", default=None,
                        choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"])
    parser.add_argument("--align_latent", dest="align_latent", action="store_true")
    parser.add_argument("--no_align_latent", dest="align_latent", action="store_false")
    parser.set_defaults(align_latent=None)
    parser.add_argument("--no_pou", dest="no_pou", action="store_true")
    parser.add_argument("--use_pou", dest="no_pou", action="store_false")
    parser.set_defaults(no_pou=None)
    parser.add_argument("--allow_partial_load", action="store_true")

    args = parser.parse_args()
    if not 0.0 < args.mass_coverage <= 1.0:
        parser.error("--mass_coverage must be in (0, 1]")
    if any(r < 0 for r in args.active_mass_ratios):
        parser.error("--active_mass_ratios must be non-negative")
    if args.n_ict_identity < 1 or args.n_id_per_dataset < 1 or args.n_functional_ict < 1:
        parser.error("identity counts must be positive")
    return args


def build_checkpoint_options(args: argparse.Namespace, ckpt: str) -> argparse.Namespace:
    config_path = Path(ckpt) / "train_opts.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint configuration not found: {config_path}")

    with config_path.open("r") as f:
        values = yaml.load(f, Loader=yaml.FullLoader)

    if int(values.get("version", 5)) != 5:
        raise ValueError("column-usage evaluation is defined only for NGBC/PDF version 5")

    # Fields consumed by CIFTrainer.__init__ and its identity-pool loader.
    values.update(
        ckpt=ckpt,
        device=args.device,
        seed=args.seed,
        version=5,
        continue_ckpt=args.epoch is not None,
        start_epoch=0 if args.epoch is None else args.epoch,
        allow_partial_load=args.allow_partial_load,
        quick=False,
        skip_part3=True,
        datasets=args.datasets,
        n_ict_identity=args.n_ict_identity,
        n_id_per_dataset=args.n_id_per_dataset,
        n_columns_sample=-1,
        n_pert_columns=1,
        active_mass_ratio=0.01,
        no_vis=True,
        log_dir=args.output_dir,
    )

    if args.last_activation is not None:
        values["last_activation"] = args.last_activation
    if args.align_latent is not None:
        values["align_latent"] = args.align_latent
    if args.no_pou is not None:
        values["no_pou"] = args.no_pou

    defaults = dict(
        last_activation="relu",
        align_latent=False,
        no_pou=False,
        optim_cage=False,
        out_type=1,
    )
    for key, default in defaults.items():
        if values.get(key) is None:
            values[key] = default

    return argparse.Namespace(**values)


def normalized_area(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Return nonnegative per-vertex area weights that sum to one."""
    area_np = vertex_area_weights(
        vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
    ).astype(np.float32)
    area = torch.from_numpy(area_np).to(device=vertices.device, dtype=vertices.dtype)
    return area / area.sum().clamp_min(EPS)


def normalize_mass(mass: torch.Tensor) -> torch.Tensor:
    return mass.clamp_min(0) / mass.clamp_min(0).sum().clamp_min(EPS)


def effective_count(distribution: torch.Tensor) -> Dict[str, float]:
    p = normalize_mass(distribution)
    entropy = -(p * torch.log(p + EPS)).sum()
    return {
        "entropy_effective_k": float(torch.exp(entropy).item()),
        "participation_ratio": float((1.0 / p.square().sum().clamp_min(EPS)).item()),
    }


def coverage_count(distribution: torch.Tensor, coverage: float) -> int:
    p = normalize_mass(distribution)
    cumulative = torch.cumsum(torch.sort(p, descending=True).values, dim=0)
    index = torch.searchsorted(
        cumulative,
        torch.tensor(coverage, device=p.device, dtype=p.dtype),
        right=False,
    )
    return min(int(index.item()) + 1, p.numel())


def identity_weight_metrics(
    W: torch.Tensor,
    area: torch.Tensor,
    active_mass_ratios: Sequence[float],
    coverage: float,
) -> Tuple[Dict, torch.Tensor, Dict[float, torch.Tensor]]:
    """Compute global mass and local-support diagnostics for one identity."""
    W = W.clamp_min(0)
    n_vert, K = W.shape
    row_sum = W.sum(dim=-1, keepdim=True)
    valid_rows = row_sum[:, 0] > EPS
    W_pou = W / row_sum.clamp_min(EPS)

    mass = normalize_mass((area[:, None] * W_pou).sum(dim=0))
    exact_active = (W > 0).any(dim=0)

    active_sets = {}
    active_counts = {}
    mean_mass = mass.mean()
    for ratio in active_mass_ratios:
        active = exact_active if ratio == 0 else mass > ratio * mean_mass
        active_sets[float(ratio)] = active
        active_counts[str(ratio)] = int(active.sum().item())

    local_entropy = -(W_pou * torch.log(W_pou + EPS)).sum(dim=-1)
    local_effective = torch.exp(local_entropy)
    valid_local = local_effective[valid_rows]
    if valid_local.numel() == 0:
        local_stats = dict(mean=None, median=None, p95=None)
    else:
        local_stats = {
            "mean": float((area[valid_rows] * valid_local).sum().item()
                          / area[valid_rows].sum().clamp_min(EPS).item()),
            "median": float(torch.quantile(valid_local, 0.5).item()),
            "p95": float(torch.quantile(valid_local, 0.95).item()),
        }

    result = {
        "n_vertices": n_vert,
        "num_control_points": K,
        "exact_active_columns": int(exact_active.sum().item()),
        "active_columns": active_counts,
        "effective": effective_count(mass),
        "coverage_k": coverage_count(mass, coverage),
        "coverage": coverage,
        "local_effective_support": local_stats,
        "invalid_zero_row_ratio": float((~valid_rows).float().mean().item()),
    }
    return result, mass, active_sets


def active_set_summary(
    active_sets_by_identity: Sequence[Dict[float, torch.Tensor]],
    ratios: Sequence[float],
) -> Dict[str, Dict]:
    summary = {}
    for ratio in ratios:
        sets = [record[float(ratio)] for record in active_sets_by_identity]
        stacked = torch.stack(sets)
        jaccards = []
        for a, b in itertools.combinations(sets, 2):
            union = (a | b).sum()
            if union > 0:
                jaccards.append(float(((a & b).sum() / union).item()))

        summary[str(ratio)] = {
            "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else None,
            "intersection_count": int(stacked.all(dim=0).sum().item()),
            "union_count": int(stacked.any(dim=0).sum().item()),
            "mean_active_count": float(stacked.sum(dim=1).float().mean().item()),
        }
    return summary


def prune_weights(
    W: torch.Tensor, importance: torch.Tensor, q: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    K = W.shape[1]
    q = min(max(int(q), 1), K)
    keep = torch.topk(importance, q).indices
    mask = torch.zeros(K, device=W.device, dtype=W.dtype)
    mask[keep] = 1
    pruned = W * mask[None]
    row_sum = pruned.sum(dim=-1, keepdim=True)
    uncovered = row_sum[:, 0] <= EPS
    return pruned / row_sum.clamp_min(EPS), uncovered


@torch.no_grad()
def functional_ict_metrics(
    trainer: CIFTrainer,
    n_identity: int,
    pruning_q: Sequence[int],
    run_pruning: bool,
    coverage: float,
) -> Dict:
    """Motion-weighted column usage on the shared 53 ICT blendshapes."""
    dataset = GCEvalDataset(n_identity=n_identity)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset.n_basis,
        collate_fn=partial(GC_collate_wrapper, device=trainer.device),
    )

    records = []
    for batch in loader:
        neutral = batch.template[0:1]
        neutral_norm = batch.template_normal[0:1]
        faces = batch.faces[0]
        W = trainer.model.predict_coordinate(neutral, neutral_norm)  # [1,N,K]

        _, v0 = trainer.model.retarget_animation(
            neutral, neutral_norm, neutral, neutral_norm, W, neutral
        )

        B = batch.vertices.shape[0]
        W_b = W.expand(B, -1, -1)
        neutral_b = neutral.expand(B, -1, -1)
        neutral_norm_b = neutral_norm.expand(B, -1, -1)
        _, v_exp = trainer.model.retarget_animation(
            neutral_b,
            neutral_norm_b,
            batch.vertices,
            batch.vertices_normal,
            W_b,
            neutral_b,
        )
        delta_v = v_exp - v0  # [53,K,3]

        area = normalized_area(neutral[0], faces)
        W0 = W[0].clamp_min(0)
        W0 = W0 / W0.sum(dim=-1, keepdim=True).clamp_min(EPS)

        # ||W_ik * Delta v_k||^2, averaged over surface area and expressions.
        weight_square_mass = (area[:, None] * W0.square()).sum(dim=0)
        cp_motion_energy = delta_v.square().sum(dim=-1).mean(dim=0)
        functional_contribution = normalize_mass(weight_square_mass * cp_motion_energy)

        full_prediction = trainer._apply_key_d(W_b, v_exp, neutral_b)
        full_mse = float(F.mse_loss(full_prediction, batch.vertices).item())

        pruning = []
        if run_pruning:
            for q in sorted(set(min(max(int(x), 1), trainer.K) for x in pruning_q)):
                W_q, uncovered = prune_weights(W0, functional_contribution, q)
                W_q_b = W_q[None].expand(B, -1, -1)
                pred_q = trainer._apply_key_d(W_q_b, v_exp, neutral_b)
                pruning.append({
                    "q": q,
                    "mse_to_gt": float(F.mse_loss(pred_q, batch.vertices).item()),
                    "mse_to_full": float(F.mse_loss(pred_q, full_prediction).item()),
                    "uncovered_vertex_ratio": float(uncovered.float().mean().item()),
                })

        records.append({
            "identity": batch.identity_name,
            "self_retargeting_mse": full_mse,
            "motion_effective": effective_count(cp_motion_energy),
            "functional_effective": effective_count(functional_contribution),
            "functional_coverage_k": coverage_count(functional_contribution, coverage),
            "coverage": coverage,
            "pruning": pruning,
            "cp_motion_energy": cp_motion_energy,
            "functional_contribution": functional_contribution,
        })

    numeric_keys = [
        "self_retargeting_mse",
        "functional_coverage_k",
    ]
    summary = {
        key: float(np.mean([record[key] for record in records]))
        for key in numeric_keys
    }
    summary["functional_entropy_effective_k"] = float(np.mean([
        record["functional_effective"]["entropy_effective_k"] for record in records
    ]))
    summary["functional_participation_ratio"] = float(np.mean([
        record["functional_effective"]["participation_ratio"] for record in records
    ]))

    if run_pruning and records:
        q_values = sorted({item["q"] for record in records for item in record["pruning"]})
        summary["mean_pruning_curve"] = []
        for q in q_values:
            rows = [
                item for record in records for item in record["pruning"] if item["q"] == q
            ]
            summary["mean_pruning_curve"].append({
                "q": q,
                "mse_to_gt": float(np.mean([row["mse_to_gt"] for row in rows])),
                "mse_to_full": float(np.mean([row["mse_to_full"] for row in rows])),
                "uncovered_vertex_ratio": float(np.mean([
                    row["uncovered_vertex_ratio"] for row in rows
                ])),
            })

    return {"summary": summary, "per_identity": records}


def write_identity_csv(path: Path, records: Sequence[Dict]) -> None:
    fields = [
        "identity", "dataset", "n_vertices", "num_control_points",
        "exact_active_columns", "entropy_effective_k", "participation_ratio",
        "coverage_k", "local_support_mean", "local_support_median",
        "local_support_p95", "invalid_zero_row_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "identity": record["identity"],
                "dataset": record["dataset"],
                "n_vertices": record["n_vertices"],
                "num_control_points": record["num_control_points"],
                "exact_active_columns": record["exact_active_columns"],
                "entropy_effective_k": record["effective"]["entropy_effective_k"],
                "participation_ratio": record["effective"]["participation_ratio"],
                "coverage_k": record["coverage_k"],
                "local_support_mean": record["local_effective_support"]["mean"],
                "local_support_median": record["local_effective_support"]["median"],
                "local_support_p95": record["local_effective_support"]["p95"],
                "invalid_zero_row_ratio": record["invalid_zero_row_ratio"],
            })


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace, ckpt: str) -> Dict:
    checkpoint_opts = build_checkpoint_options(args, ckpt)
    trainer = CIFTrainer(checkpoint_opts)
    trainer.model.eval()

    pool = trainer._load_identity_pool()
    per_identity = []
    masses = []
    active_sets = []

    for identity in pool:
        W = trainer.model.predict_coordinate(identity.neu_vert, identity.neu_norm)[0]
        area = normalized_area(identity.neu_vert[0], identity.faces)
        metrics, mass, sets = identity_weight_metrics(
            W, area, args.active_mass_ratios, args.mass_coverage
        )
        metrics.update(identity=identity.name, dataset=identity.dataset)
        per_identity.append(metrics)
        masses.append(mass)
        active_sets.append(sets)

    mass_tensor = torch.stack(masses)
    identity_balanced_mass = normalize_mass(mass_tensor.mean(dim=0))
    exact_counts = [record["exact_active_columns"] for record in per_identity]
    local_means = [
        record["local_effective_support"]["mean"] for record in per_identity
        if record["local_effective_support"]["mean"] is not None
    ]

    summary = {
        "checkpoint": ckpt,
        "epoch": args.epoch,
        "num_identities": len(per_identity),
        "num_control_points": trainer.K,
        "mean_exact_active_columns": float(np.mean(exact_counts)),
        "identity_balanced_effective": effective_count(identity_balanced_mass),
        "identity_balanced_coverage_k": coverage_count(
            identity_balanced_mass, args.mass_coverage
        ),
        "coverage": args.mass_coverage,
        "mean_local_effective_support": float(np.mean(local_means)) if local_means else None,
        "active_set_consistency": active_set_summary(
            active_sets, args.active_mass_ratios
        ),
        "identity_balanced_column_mass": identity_balanced_mass,
    }

    functional = None
    if not args.skip_functional:
        functional = functional_ict_metrics(
            trainer=trainer,
            n_identity=args.n_functional_ict,
            pruning_q=args.pruning_q,
            run_pruning=not args.skip_pruning,
            coverage=args.mass_coverage,
        )

    checkpoint_name = Path(ckpt).name
    result_dir = Path(args.output_dir) / checkpoint_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "summary": summary,
        "per_identity": per_identity,
        "functional_ict": functional,
    }
    with (result_dir / "column_usage_metrics.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    write_identity_csv(result_dir / "column_usage_per_identity.csv", per_identity)

    compact = {
        "checkpoint": checkpoint_name,
        "mean_exact_active": summary["mean_exact_active_columns"],
        "global_effective_k": summary["identity_balanced_effective"]["entropy_effective_k"],
        "global_participation_ratio": summary["identity_balanced_effective"]["participation_ratio"],
        "global_coverage_k": summary["identity_balanced_coverage_k"],
        "mean_local_support": summary["mean_local_effective_support"],
    }
    if functional is not None:
        compact.update(
            functional_effective_k=functional["summary"]["functional_entropy_effective_k"],
            functional_coverage_k=functional["summary"]["functional_coverage_k"],
            ict_self_mse=functional["summary"]["self_retargeting_mse"],
        )
    print(json.dumps(compact, indent=2))
    return result


def main() -> None:
    args = options()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []
    for ckpt in args.ckpt:
        results.append(evaluate_checkpoint(args, ckpt))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "column_usage_comparison.json").open("w") as f:
        json.dump(to_jsonable(results), f, indent=2)


if __name__ == "__main__":
    main()
