import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def try_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_bulk_csv(path):
    data = {
        "iter": [],
        "train_loss": [],
        "val_loss": [],
        "mfu": [],
        "lr": [],
        "batch_size": [],
        "tokens": [],
        "peak_mb": [],
        "iter_ms": [],
    }
    with open(path, newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            iter_value = try_float(row[0])
            if iter_value is None:
                continue
            data["iter"].append(int(iter_value))
            data["train_loss"].append(try_float(row[2]) if len(row) > 2 else np.nan)
            data["val_loss"].append(try_float(row[3]) if len(row) > 3 else np.nan)
            data["mfu"].append(try_float(row[4]) if len(row) > 4 else np.nan)
            data["lr"].append(try_float(row[5]) if len(row) > 5 else np.nan)
            data["batch_size"].append(try_float(row[6]) if len(row) > 6 else np.nan)
            data["tokens"].append(try_float(row[7]) if len(row) > 7 else np.nan)
            data["peak_mb"].append(try_float(row[8]) if len(row) > 8 else np.nan)
            data["iter_ms"].append(try_float(row[-1]) if len(row) > 0 else np.nan)

    for key, values in data.items():
        data[key] = np.asarray(values, dtype=float)
    data["hours"] = np.cumsum(np.nan_to_num(data["iter_ms"], nan=0.0)) / 3_600_000.0
    return data


def load_side_csv(path):
    rows = []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed = {
                "iter": int(float(row["iter"])),
                "target_name": row["target_name"],
                "source_iter": int(float(row["source_iter"])),
                "base_probe_loss": float(row["base_probe_loss"]),
                "candidate_probe_loss": float(row["candidate_probe_loss"]),
                "delta": float(row["delta"]),
                "accepted": int(float(row["accepted"])),
                "piggyback_k": int(float(row["piggyback_k"])),
                "snapshot_mb": float(row["snapshot_mb"]),
                "lr": float(row["lr"]),
                "batch_size": float(row["batch_size"]),
                "tokens_trained": float(row["tokens_trained"]),
                "peak_gpu_mb": float(row["peak_gpu_mb"]),
                "iter_latency_ms": float(row["iter_latency_ms"]),
            }
            rows.append(parsed)
    return rows


def summarize_run(run):
    best_idx = int(np.nanargmin(run["val_loss"]))
    return {
        "best_val_loss": float(run["val_loss"][best_idx]),
        "best_iter": int(run["iter"][best_idx]),
        "hours_to_best": float(run["hours"][best_idx]),
        "peak_gpu_gb": float(np.nanmax(run["peak_mb"]) / 1024.0),
        "mean_iter_ms": float(np.nanmean(run["iter_ms"])),
    }


def format_summary_text(label, summary):
    return (
        f"{label}\n"
        f"best val: {summary['best_val_loss']:.4f}\n"
        f"best iter: {summary['best_iter']}\n"
        f"hours to best: {summary['hours_to_best']:.2f}\n"
        f"peak GPU: {summary['peak_gpu_gb']:.2f} GB\n"
        f"mean iter: {summary['mean_iter_ms']:.1f} ms"
    )


def plot_report(baseline, rehearsal, side_rows, args):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(baseline["iter"], baseline["val_loss"], label=args.baseline_label, linewidth=2.0)
    axes[0, 0].plot(rehearsal["iter"], rehearsal["val_loss"], label=args.rehearsal_label, linewidth=2.0)
    axes[0, 0].set_title("Validation Loss vs Iteration")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Validation loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(baseline["hours"], baseline["val_loss"], label=args.baseline_label, linewidth=2.0)
    axes[0, 1].plot(rehearsal["hours"], rehearsal["val_loss"], label=args.rehearsal_label, linewidth=2.0)
    axes[0, 1].set_title("Validation Loss vs Wall-Clock Time")
    axes[0, 1].set_xlabel("Wall-clock hours")
    axes[0, 1].set_ylabel("Validation loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    base_summary = summarize_run(baseline)
    rehearsal_summary = summarize_run(rehearsal)
    categories = ["Peak GPU (GB)", "Mean iter (ms)", "Hours to best"]
    base_values = [
        base_summary["peak_gpu_gb"],
        base_summary["mean_iter_ms"],
        base_summary["hours_to_best"],
    ]
    rehearsal_values = [
        rehearsal_summary["peak_gpu_gb"],
        rehearsal_summary["mean_iter_ms"],
        rehearsal_summary["hours_to_best"],
    ]
    x = np.arange(len(categories))
    width = 0.35
    axes[1, 0].bar(x - width / 2, base_values, width=width, label=args.baseline_label)
    axes[1, 0].bar(x + width / 2, rehearsal_values, width=width, label=args.rehearsal_label)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].set_title("Training Cost Summary")
    axes[1, 0].grid(alpha=0.3, axis="y")
    axes[1, 0].legend()

    if side_rows:
        side_iters = np.asarray([row["iter"] for row in side_rows], dtype=float)
        side_delta = np.asarray([row["delta"] for row in side_rows], dtype=float)
        side_accept = np.asarray([row["accepted"] for row in side_rows], dtype=float)
        axes[1, 1].plot(side_iters, side_delta, marker="o", linewidth=2.0, label="base loss - candidate loss")
        axes[1, 1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        acceptance_rate = np.cumsum(side_accept) / np.arange(1, len(side_accept) + 1)
        ax2 = axes[1, 1].twinx()
        ax2.plot(side_iters, acceptance_rate, color="tab:orange", linestyle="--", linewidth=2.0, label="acceptance rate")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Loss improvement")
        ax2.set_ylabel("Acceptance rate")
        axes[1, 1].set_title("Side-Network Effectiveness")
        axes[1, 1].grid(alpha=0.3)
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.02, 0.92, format_summary_text(args.baseline_label, base_summary), va="top", ha="left", fontsize=11)
        axes[1, 1].text(0.52, 0.92, format_summary_text(args.rehearsal_label, rehearsal_summary), va="top", ha="left", fontsize=11)
        axes[1, 1].set_title("Run Summaries")

    fig.suptitle(args.title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    report_path = os.path.join(args.out_dir, "side_rehearsal_report.png")
    fig.savefig(report_path, dpi=220)
    print("[OK] saved:", report_path)

    summary = {
        "baseline": base_summary,
        "rehearsal": rehearsal_summary,
        "relative_overheads": {
            "peak_gpu_pct": (
                (rehearsal_summary["peak_gpu_gb"] - base_summary["peak_gpu_gb"])
                / max(base_summary["peak_gpu_gb"], 1e-8)
            ) * 100.0,
            "iter_latency_pct": (
                (rehearsal_summary["mean_iter_ms"] - base_summary["mean_iter_ms"])
                / max(base_summary["mean_iter_ms"], 1e-8)
            ) * 100.0,
        },
    }
    if side_rows:
        summary["side_rehearsal"] = {
            "num_updates": len(side_rows),
            "mean_delta": float(np.mean([row["delta"] for row in side_rows])),
            "acceptance_rate": float(np.mean([row["accepted"] for row in side_rows])),
            "max_snapshot_mb": float(np.max([row["snapshot_mb"] for row in side_rows])),
        }
    summary_path = os.path.join(args.out_dir, "side_rehearsal_summary.json")
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)
    print("[OK] saved:", summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_bulk", required=True)
    parser.add_argument("--rehearsal_bulk", required=True)
    parser.add_argument("--side_eval_csv", default=None)
    parser.add_argument("--baseline_label", default="STE baseline")
    parser.add_argument("--rehearsal_label", default="Side rehearsal")
    parser.add_argument("--title", default="Side-Rehearsal QAT Report")
    parser.add_argument("--out_dir", default="plots/side_rehearsal")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    baseline = load_bulk_csv(args.baseline_bulk)
    rehearsal = load_bulk_csv(args.rehearsal_bulk)
    side_rows = load_side_csv(args.side_eval_csv) if args.side_eval_csv and os.path.exists(args.side_eval_csv) else []
    plot_report(baseline, rehearsal, side_rows, args)


if __name__ == "__main__":
    main()
