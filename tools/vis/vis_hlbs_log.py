"""
vis_hlbs_log.py — Parse HLBS log.txt and plot loss curves (Train components + Val).

Usage:
    python tools/vis/vis_hlbs_log.py \
        --ckpt_dir ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01

    # Multiple runs side-by-side (overlay totals + per-run subplots)
    python tools/vis/vis_hlbs_log.py \
        --ckpt_dir ckpts_hlbs/run_A ckpts_hlbs/run_B --out _tmp/cmp.png
"""
import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_TRAIN_PAT = re.compile(r"^\[(\d+)/\d+\]\[(\d+)\]\[Train\]\s+(.+)$")
_KV_PAT    = re.compile(r"([A-Za-z][A-Za-z0-9_\-]*)\s*:\s*([0-9eE.+\-]+)")
# Two supported Val formats:
#   (A) old: "[NNN] Val: 1.23e-04 (Best: X [N])"
#   (B) new: "[NNN] Val: recon-lbs: X recon-neu: X ... total: 1.23e-04 (Best: X [N])"
_VAL_HEAD = re.compile(r"^\[(\d+)\]\s+Val:\s+(.*?)\s*(?:\(Best:\s+([0-9eE.+\-]+)\s+\[(\d+)\]\))?\s*$")


def parse_log(log_path):
    """Return (epochs, train_curves dict[key]->array, val_eps, val_vals)."""
    train = defaultdict(lambda: defaultdict(list))
    val   = defaultdict(dict)   # epoch -> key -> value (new) or {'total': v} (old)
    best_ep = None
    best_v  = None
    with open(log_path) as f:
        for line in f:
            line = line.rstrip()
            m = _TRAIN_PAT.match(line)
            if m:
                ep = int(m.group(1))
                for k, v in _KV_PAT.findall(m.group(3)):
                    try:
                        train[ep][k].append(float(v))
                    except ValueError:
                        pass
                continue
            m = _VAL_HEAD.match(line)
            if m:
                ep = int(m.group(1))
                body = m.group(2)
                kvs = _KV_PAT.findall(body)
                if kvs:
                    # new format with per-component keys
                    for k, v in kvs:
                        try: val[ep][k] = float(v)
                        except ValueError: pass
                    if "total" not in val[ep]:
                        # If parser caught keys but no explicit total, skip.
                        pass
                else:
                    # old format: single number in body
                    try: val[ep]["total"] = float(body.strip())
                    except ValueError: pass
                if m.group(3) is not None:
                    best_v, best_ep = float(m.group(3)), int(m.group(4))

    epochs = sorted(train.keys())
    all_keys = set()
    for e in epochs:
        all_keys.update(train[e].keys())
    curves = {
        k: np.array([np.mean(train[e].get(k, [np.nan])) for e in epochs])
        for k in all_keys
    }
    val_eps  = sorted(val.keys())
    val_keys = set()
    for e in val_eps:
        val_keys.update(val[e].keys())
    val_curves = {
        k: np.array([val[e].get(k, np.nan) for e in val_eps])
        for k in val_keys
    }
    return epochs, curves, val_eps, val_curves, best_ep, best_v


_DEFAULT_KEYS = [
    "recon-lbs", "recon-neu", "recon-normal",
    "L_bind_reg", "L_sigma", "L_net_center",
]


def plot_single(log_path, out_path, keys=_DEFAULT_KEYS, title=None):
    epochs, curves, val_eps, val_curves, best_ep, best_v = parse_log(log_path)
    if not epochs:
        print(f"[WARN] no train rows in {log_path}")
        return

    keys = [k for k in keys if k in curves]
    n_panels = 2 + len(keys)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    total = curves.get("total")
    val_total = val_curves.get("total")
    # Fallback if log had no "(Best: X [N])" tag
    if best_ep is None and val_total is not None and len(val_eps):
        best_ep = val_eps[int(np.nanargmin(val_total))]
        best_v  = float(np.nanmin(val_total))

    # panel 0: train total + val overlay
    ax = axes[0]
    if total is not None:
        ax.plot(epochs, total, lw=1.4, label="Train total")
        ax.set_yscale("log")
    if val_total is not None and len(val_eps):
        ax2 = ax.twinx()
        ax2.plot(val_eps, val_total, "o-", color="tab:red", lw=1.6, ms=4, label="Val total")
        ax2.set_yscale("log")
        if best_ep is not None:
            ax2.scatter([best_ep], [best_v], color="tab:red", s=90, marker="*",
                        zorder=5, label=f"best @ep{best_ep}={best_v:.3e}")
            ax2.axhline(best_v, color="tab:red", ls=":", alpha=0.4)
        ax2.set_ylabel("Val", color="tab:red")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_xlabel("epoch"); ax.set_ylabel("Train total")
    ax.set_title("Train total vs Val"); ax.grid(True, alpha=0.3)

    # panel 1: Val components overlay (linear)
    ax = axes[1]
    if val_total is not None and len(val_eps):
        comp_keys = [k for k in val_curves if k != "total"]
        for k in comp_keys:
            ax.plot(val_eps, val_curves[k], "o-", lw=1.2, ms=3, label=k)
        ax.plot(val_eps, val_total, "o-", lw=1.8, ms=4, label="total", color="black")
        if best_ep is not None:
            ax.scatter([best_ep], [best_v], color="red", s=90, marker="*", zorder=5,
                       label=f"best @ep{best_ep}={best_v:.3e}")
            ax.axhline(best_v, color="red", ls=":", alpha=0.4)
        ax.set_yscale("log")
        last_total = val_total[-1] if len(val_total) else float("nan")
        ax.set_title(f"Val components — last total={last_total:.3e} @ep{val_eps[-1]}")
        ax.legend(fontsize=7, loc="best")
    else:
        ax.text(0.5, 0.5, "no Val rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3)

    # remaining: individual train components
    for i, k in enumerate(keys, start=2):
        ax = axes[i]
        ys = curves[k]
        if np.all(np.isnan(ys)) or np.nanmax(np.abs(ys)) == 0:
            ax.text(0.5, 0.5, f"{k}\n(zero/empty)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.plot(epochs, ys, lw=1.3)
        pos = ys[ys > 0]
        lin = max(1e-12, float(np.nanmin(pos))) if pos.size else 1e-12
        ax.set_yscale("symlog", linthresh=lin)
        ax.set_xlabel("epoch"); ax.set_title(k); ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    if title is None:
        title = Path(log_path).parent.name
    fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    best_str = f"{best_v:.4e} @ep{best_ep}" if best_v is not None else "n/a"
    print(f"Saved -> {out_path}  (train ep last={epochs[-1]}, val best={best_str})")


def plot_compare(log_paths, out_path):
    """Overlay multiple runs' Train total + Val on shared axes."""
    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for p in log_paths:
        epochs, curves, val_eps, val_curves, _, _ = parse_log(p)
        name = Path(p).parent.name
        if "total" in curves:
            ax_t.plot(epochs, curves["total"], lw=1.4, label=name)
        vt = val_curves.get("total")
        if vt is not None and len(val_eps):
            ax_v.plot(val_eps, vt, "o-", lw=1.4, ms=3, label=name)
    for ax, t in [(ax_t, "Train total"), (ax_v, "Val")]:
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title(t)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved overlay -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", nargs="+", required=True,
                    help="One or more ckpt directories (each must contain log.txt)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG. Default: <ckpt_dir>/loss_plot.png (single) "
                         "or _tmp/log_compare.png (multi)")
    ap.add_argument("--keys", nargs="+", default=_DEFAULT_KEYS,
                    help="Train metric keys to plot per panel")
    args = ap.parse_args()

    log_paths = []
    for d in args.ckpt_dir:
        lp = os.path.join(d, "log.txt")
        if not os.path.isfile(lp):
            print(f"[skip] log.txt not found in {d}")
            continue
        log_paths.append(lp)
    if not log_paths:
        raise SystemExit("No log.txt found in any provided --ckpt_dir")

    if len(log_paths) == 1:
        out = args.out or os.path.join(args.ckpt_dir[0], "loss_plot.png")
        plot_single(log_paths[0], out, keys=args.keys)
    else:
        for lp in log_paths:
            plot_single(lp, os.path.join(os.path.dirname(lp), "loss_plot.png"),
                        keys=args.keys)
        out = args.out or "_tmp/log_compare.png"
        plot_compare(log_paths, out)


if __name__ == "__main__":
    main()
