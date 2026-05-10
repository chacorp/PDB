"""
Summarize evaluation results from eval_CBD folder.
Produces one table per metric (MSE, MSE-in, MSE-out, Lap),
with models as rows and datasets as columns.
"""

import os
import re
import pandas as pd
from pathlib import Path

EVAL_DIR = Path("eval_CBD")

# Regex to extract metrics from log.txt
METRIC_RE = re.compile(
    r"\[Eval\]"
    r"(?:\s+MSE:\s*(?P<MSE>[+-]?\d+\.\d+e[+-]\d+))?"
    r"(?:\s+MSE-in:\s*(?P<MSE_in>[+-]?\d+\.\d+e[+-]\d+))?"
    r"(?:\s+MSE-out:\s*(?P<MSE_out>[+-]?\d+\.\d+e[+-]\d+))?"
    r"(?:\s+Lap:\s*(?P<Lap>[+-]?\d+\.\d+e[+-]\d+))?"
)
DATASET_RE = re.compile(r"\[Dataset\]:\s*(\S+)")


def parse_log(log_path: Path) -> dict:
    """Return dict with dataset and metric values, or empty dict if unparseable."""
    try:
        text = log_path.read_text()
    except Exception:
        return {}

    dataset_m = DATASET_RE.search(text)
    metric_m = METRIC_RE.search(text)
    if not metric_m:
        return {}

    result = {"dataset": dataset_m.group(1) if dataset_m else None}
    for key in ("MSE", "MSE_in", "MSE_out", "Lap"):
        val = metric_m.group(key)
        result[key] = float(val) if val is not None else None
    return result


def subfolder_config(subfolder_name: str, dataset: str) -> str:
    """
    Derive a short config tag from the subfolder name by stripping the
    dataset prefix and trailing _NN suffixes.
    e.g.  'biwi-masked-laplacian'  -> 'masked-laplacian'
          'mf_ROM'                  -> ''
          'ict-masked-laplacian_00' -> 'masked-laplacian_00'
    """
    s = subfolder_name
    # strip leading dataset name
    for prefix in sorted([dataset, dataset.replace("_", "-")], key=len, reverse=True):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.lstrip("-")


def collect_records():
    records = []
    for model_dir in sorted(EVAL_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for sub_dir in sorted(model_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            log_path = sub_dir / "log.txt"
            if not log_path.exists():
                continue
            parsed = parse_log(log_path)
            if not parsed or parsed.get("MSE") is None:
                continue
            dataset = parsed["dataset"] or sub_dir.name.split("-")[0]
            config = subfolder_config(sub_dir.name, dataset)
            records.append({
                "model":   model_name,
                "dataset": dataset,
                "config":  config,
                "sub":     sub_dir.name,
                "MSE":     parsed["MSE"],
                "MSE-in":  parsed["MSE_in"],
                "MSE-out": parsed["MSE_out"],
                "Lap":     parsed["Lap"],
            })
    return records


def make_tables(records):
    df = pd.DataFrame(records)

    # Use (dataset, config) as the column identifier
    df["col"] = df.apply(
        lambda r: r["dataset"] if not r["config"] else f"{r['dataset']} [{r['config']}]",
        axis=1,
    )

    metrics = ["MSE", "MSE-in", "MSE-out", "Lap"]
    tables = {}
    for metric in metrics:
        sub = df[df[metric].notna()][["model", "col", metric]].copy()
        if sub.empty:
            continue
        # If a (model, col) pair has multiple entries keep the first
        sub = sub.drop_duplicates(subset=["model", "col"], keep="first")
        pivot = sub.pivot(index="model", columns="col", values=metric)
        # Sort columns: plain dataset names first, then decorated ones
        cols = sorted(pivot.columns, key=lambda c: ("]" in c, c))
        tables[metric] = pivot[cols]
    return tables


def format_value(v):
    if pd.isna(v):
        return "-"
    return f"{v:.4e}"


def print_tables(tables: dict):
    pd.set_option("display.float_format", lambda x: f"{x:.4e}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 300)

    for metric, table in tables.items():
        print(f"\n{'='*80}")
        print(f"  Metric: {metric}")
        print(f"{'='*80}")
        print(table.apply(lambda col: col.map(format_value)).to_string())


def save_tables_csv(tables: dict, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    for metric, table in tables.items():
        fname = out_dir / f"eval_summary_{metric.replace('-', '_')}.csv"
        table.to_csv(fname)
        print(f"Saved {fname}")


def table_to_md(table: pd.DataFrame, metric: str) -> str:
    fmt = table.apply(lambda col: col.map(format_value))
    fmt.index.name = "model \\ dataset"
    lines = [f"# {metric}\n"]
    lines.append(fmt.to_markdown())
    return "\n".join(lines)


def save_tables_md(tables: dict, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    for metric, table in tables.items():
        fname = out_dir / f"eval_summary_{metric.replace('-', '_')}.md"
        fname.write_text(table_to_md(table, metric))
        print(f"Saved {fname}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize eval_CBD results")
    parser.add_argument("--save", action="store_true", help="Save tables as MD (and CSV) files")
    parser.add_argument("--out-dir", default="eval_summary", help="Output directory")
    args = parser.parse_args()

    records = collect_records()
    print(f"Parsed {len(records)} result entries from {len(set(r['model'] for r in records))} models.")
    tables = make_tables(records)
    print_tables(tables)

    if args.save:
        save_tables_md(tables, Path(args.out_dir))
        save_tables_csv(tables, Path(args.out_dir))
