# aggregate_and_plot.py
import os, json, math
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get("OUT_DIR", "./eval_out")
FIG_PATH = os.environ.get("FIG_PATH", os.path.join(OUT_DIR, "base_vs_sft_imdb.png"))

def load_stats(path):
    with open(path, "r") as f:
        obj = json.load(f)
    s = obj["stats"]
    s["file"] = os.path.basename(path)
    return s

def evaluate_sft_imdb_qwen_p2():
    base_path = os.path.join(OUT_DIR, "results_base.json")
    sft_path  = os.path.join(OUT_DIR, "results_sft.json")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Lipsește {base_path}. Rulează mai întâi gen_and_score.py --which base")
    if not os.path.exists(sft_path):
        raise FileNotFoundError(f"Lipsește {sft_path}. Rulează gen_and_score.py --which sft")

    res_base = load_stats(base_path)
    res_sft  = load_stats(sft_path)
    results = [res_base, res_sft]

    # tipărim sumar
    for r in results:
        print(f"{r['name']}: mean={r['mean_pos']:.4f}, std={r['std_pos']:.4f}, n={r['n']} (from {r['file']})")

    # plot
    names = [r["name"] for r in results]
    means = [r["mean_pos"] for r in results]
    stds  = [r["std_pos"] for r in results]
    ns    = [max(1, r["n"]) for r in results]
    sems  = [s/ math.sqrt(n) for s, n in zip(stds, ns)]

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure()
    x = np.arange(len(names))
    plt.bar(x, means, yerr=sems, capsize=6)
    plt.xticks(x, names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean P(positive) ± SEM")
    plt.title("Base vs SFT: P(positive) on Generated Text")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    print(f"Saved figure to: {FIG_PATH}")

if __name__ == "__main__":
    evaluate_sft_imdb_qwen_p2()