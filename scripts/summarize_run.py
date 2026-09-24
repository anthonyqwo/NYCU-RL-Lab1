"""Summarize completed training/evaluation CSVs and plot mean training scores."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_id")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
logs = root / "logs"
summary = {}
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for variant, label in (("td_state", "TD-state"), ("td_afterstate", "TD-after-state")):
    data = {}
    for phase, expected in (("train", 100000), ("eval", 1000)):
        source = logs / f"{variant}_100k_{args.run_id}_{phase}.csv"
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected or [int(r["episode"]) for r in rows] != list(range(1, expected + 1)):
            raise ValueError(f"Incomplete or invalid episodes in {source}")
        scores = [int(r["score"]) for r in rows]
        tiles = [int(r["max_tile"]) for r in rows]
        wins = sum(t >= 2048 for t in tiles)
        data[phase] = {
            "games": len(rows), "mean_score": statistics.mean(scores),
            "max_score": max(scores), "wins_2048": wins,
            "win_rate_2048_percent": 100 * wins / len(rows),
            "max_tile": max(tiles), "elapsed_seconds": float(rows[-1]["elapsed_seconds"]),
            "tile_reach_percent": {str(t): 100 * sum(v >= t for v in tiles) / len(rows)
                                   for t in (512, 1024, 2048, 4096, 8192)},
        }
        if phase == "train":
            x = list(range(1000, len(rows) + 1, 1000))
            y = [statistics.mean(scores[i - 1000:i]) for i in x]
            ax.plot(x, y, label=label, linewidth=1.7)
    summary[variant] = data
ax.set(xlabel="Training episodes", ylabel="Mean score (per 1,000 episodes)",
       title="2048 TD learning: 100,000 episodes per variant")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(logs / f"training_curve_{args.run_id}.png", dpi=180)
plt.close(fig)
(logs / f"results_{args.run_id}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
lines = ["# 100k training results", "", f"Run: `{args.run_id}`. Training seed: 42; evaluation seed: 2026; alpha: 0.1.",
         "Both variants trained from zero for 100,000 episodes, then evaluated for 1,000 games without learning.",
         "", "| Variant | Training seconds | Evaluation mean | Evaluation max | 2048 wins | 2048 rate |",
         "|---|---:|---:|---:|---:|---:|"]
for variant, data in summary.items():
    t, e = data["train"], data["eval"]
    lines.append(f"| {variant} | {t['elapsed_seconds']:.2f} | {e['mean_score']:.2f} | {e['max_score']} | {e['wins_2048']}/1000 | {e['win_rate_2048_percent']:.1f}% |")
lines += ["", "Times use the program's elapsed clock through the last game, excluding the final weight write.",
          "This is one training seed and one evaluation seed per variant; results are not a multi-seed comparison.",
          "Training curves show non-overlapping blocks of 1,000 games, not evaluation performance.", ""]
(logs / f"results_{args.run_id}.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, indent=2))
