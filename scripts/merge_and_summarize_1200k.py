"""
Merge fragmented training CSVs for TD-State and TD-Afterstate up to 1.2M episodes,
compute milestone statistics (100k, 500k, 1M, 1.2M), and plot full training curves.
"""
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

SEGMENTS = {
    "td_afterstate": [
        (LOGS / "td_afterstate_100k_20260924_202933_train.csv", 100000),
        (LOGS / "td_afterstate_1200k_20260924_232129_continuation.csv", 750000),
        (LOGS / "td_afterstate_1200k_resume.csv", 350000),
    ],
    "td_state": [
        (LOGS / "td_state_100k_20260924_202933_train.csv", 100000),
        (LOGS / "td_state_1200k_20260924_232129_continuation.csv", 330000),
        (LOGS / "td_state_1200k_resume.csv", 770000),
    ]
}

MILESTONES = [100000, 500000, 1000000, 1200000]

def merge_variant(variant):
    print(f"Merging {variant}...")
    combined_path = LOGS / f"{variant}_1200k_total_train.csv"
    total_count = 0
    elapsed_offset = 0.0
    
    scores = []
    max_tiles = []
    
    # Store 1000-ep block means for plotting
    curve_x = []
    curve_y = []
    
    milestone_stats = {}
    
    with combined_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["episode", "score", "max_tile", "moves", "elapsed_seconds"])
        writer.writeheader()
        
        block_scores = []
        
        for file_path, max_take in SEGMENTS[variant]:
            if not file_path.exists():
                print(f"Warning: {file_path} not found!")
                continue
            
            print(f"  Reading up to {max_take} rows from {file_path.name}...")
            local_elapsed = 0.0
            with file_path.open(newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                for local_idx, row in enumerate(reader, 1):
                    if local_idx > max_take:
                        break
                    total_count += 1
                    local_elapsed = float(row["elapsed_seconds"])
                    cum_elapsed = elapsed_offset + local_elapsed
                    
                    score = int(row["score"])
                    tile = int(row["max_tile"])
                    
                    row["episode"] = total_count
                    row["elapsed_seconds"] = f"{cum_elapsed:.6f}"
                    writer.writerow(row)
                    
                    scores.append(score)
                    max_tiles.append(tile)
                    block_scores.append(score)
                    
                    if total_count % 2000 == 0:
                        curve_x.append(total_count)
                        curve_y.append(statistics.mean(block_scores))
                        block_scores.clear()
                    
                    if total_count in MILESTONES:
                        # Compute statistics for this milestone window (last 10,000 games)
                        w_start = max(0, total_count - 10000)
                        w_scores = scores[w_start:total_count]
                        w_tiles = max_tiles[w_start:total_count]
                        
                        milestone_stats[str(total_count)] = {
                            "episodes": total_count,
                            "elapsed_seconds": cum_elapsed,
                            "window_games": len(w_scores),
                            "window_mean_score": round(statistics.mean(w_scores), 2),
                            "window_max_score": max(w_scores),
                            "overall_max_score": max(scores),
                            "window_win_rate_2048": round(100.0 * sum(t >= 2048 for t in w_tiles) / len(w_tiles), 2),
                            "window_rate_4096": round(100.0 * sum(t >= 4096 for t in w_tiles) / len(w_tiles), 2),
                            "window_rate_8192": round(100.0 * sum(t >= 8192 for t in w_tiles) / len(w_tiles), 2),
                            "window_rate_16384": round(100.0 * sum(t >= 16384 for t in w_tiles) / len(w_tiles), 2),
                        }
                        print(f"    Milestone {total_count}: Mean={milestone_stats[str(total_count)]['window_mean_score']}, 2048={milestone_stats[str(total_count)]['window_win_rate_2048']}%")
            
            elapsed_offset += local_elapsed
            print(f"  Finished segment: cumulative count = {total_count}, cumulative elapsed = {elapsed_offset:.1f}s")
            
    return {
        "total_episodes": total_count,
        "total_elapsed_seconds": elapsed_offset,
        "milestones": milestone_stats,
        "curve_x": curve_x,
        "curve_y": curve_y,
    }

def main():
    results = {}
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    
    labels = {
        "td_afterstate": "TD-after-state (V(s'))",
        "td_state": "TD-state (V(s))"
    }
    colors = {
        "td_afterstate": "#246B9C",
        "td_state": "#CC6A20"
    }
    
    for variant in ["td_afterstate", "td_state"]:
        res = merge_variant(variant)
        results[variant] = {
            "total_episodes": res["total_episodes"],
            "total_elapsed_seconds": res["total_elapsed_seconds"],
            "milestones": res["milestones"]
        }
        ax.plot(res["curve_x"], res["curve_y"], label=labels[variant], color=colors[variant], linewidth=1.5, alpha=0.9)
    
    # Add vertical lines for milestones
    for m in [100000, 500000, 1000000, 1200000]:
        ax.axvline(x=m, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(m, ax.get_ylim()[0] + 5000, f" {m//1000}k", rotation=90, color="gray", fontsize=9, va="bottom")
        
    ax.set_xlabel("Training Episodes", fontsize=11)
    ax.set_ylabel("Mean Score (per 2,000 episodes)", fontsize=11)
    ax.set_title("2048 TD Learning Training Curves (0 to 1.2M Episodes)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="lower right")
    
    plot_path = LOGS / "training_curve_1200k.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")
    
    json_path = LOGS / "results_1200k.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results JSON to {json_path}")

if __name__ == "__main__":
    main()
