"""Aggregate HoloSD (ours) 10-seed i4_sepchan metrics -> mean +/- std.

Reads exp/bicycle/HoloSD/seeds/s*/eval/metrics.json (written by holosd_final eval),
backfills the `split` field (ring - ring_hard) so each file matches the baseline schema,
merges a "holosd" entry into URL/baselines/results/seed_summary.json, and prints the
headline numbers in the same style as scripts/agg_seeds.py.
"""
import os
import json
import glob
import numpy as np

HSD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # URL/HoloSD
SEED_GLOB = os.path.join(HSD, "exp/bicycle/HoloSD/seeds/s*/eval/metrics.json")
SUMMARY = "/media/satassd/grchoe/URL/baselines/results/seed_summary.json"

# headline scalars: name -> (group, key) in metrics.json
SCALARS = {
    "rand_reach":      ("nav_random", "reach"),
    "rand_pose":       ("nav_random", "pose"),
    "rand_pose_final": ("nav_random", "pose_final"),
    "ring_reach":      ("nav_ring", "reach"),
    "ring_pose":       ("nav_ring", "pose"),
    "hard_reach":      ("nav_ring_hard", "reach"),
    "hard_pose":       ("nav_ring_hard", "pose"),
    "split_reach":     ("split", "reach"),
    "split_pose":      ("split", "pose"),
}


def stat(v):
    v = np.asarray(v, float)
    n = len(v)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if n > 1 else 0.0)}


def main():
    files = sorted(glob.glob(SEED_GLOB))
    if not files:
        print(f"!! no metrics under {SEED_GLOB}")
        return
    ms = []
    for f in files:
        m = json.load(open(f))
        # backfill split = ring - ring_hard (heading-asymmetry gap), then persist
        m["split"] = {
            "reach": m["nav_ring"]["reach"] - m["nav_ring_hard"]["reach"],
            "pose":  m["nav_ring"]["pose"] - m["nav_ring_hard"]["pose"],
        }
        json.dump(m, open(f, "w"), indent=2)
        ms.append(m)
    n = len(ms)

    rec = {"n_seeds": n}
    for key, (g, k) in SCALARS.items():
        rec[key] = stat([m[g][k] for m in ms])
    sw = np.array([m["reach_vs_radius"]["reach"] for m in ms], float)
    rec["sweep_r"] = ms[0]["reach_vs_radius"]["r"]
    rec["sweep_mean"] = sw.mean(0).tolist()
    rec["sweep_std"] = (sw.std(0, ddof=1) if n > 1 else np.zeros(sw.shape[1])).tolist()

    # merge into the shared baseline summary
    summ = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    summ["holosd"] = rec
    json.dump(summ, open(SUMMARY, "w"), indent=1)

    def c(key):
        return f"{rec[key]['mean']*100:.1f}±{rec[key]['std']*100:.1f}"

    print(f"\nHoloSD (ours) — {n} seeds")
    print("| metric | random | ring | ring_hard |")
    print("|---|---|---|---|")
    print(f"| reach | {c('rand_reach')} | {c('ring_reach')} | {c('hard_reach')} |")
    print(f"| pose  | {c('rand_pose')} | {c('ring_pose')} | {c('hard_pose')} |")
    print(f"| pose_final (random) | {c('rand_pose_final')} | | |")
    print(f"| split reach/pose | {c('split_reach')} / {c('split_pose')} | | |")
    print("\nR_min sweep reach (mean%):", " ".join(f"{x*100:.0f}" for x in rec["sweep_mean"]))
    print(f"-> merged into {SUMMARY} [key: holosd]")


if __name__ == "__main__":
    main()
