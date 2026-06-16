"""i4_sepchan-style eval figures + nav videos for a HoloSD (clean) checkpoint.

Reproduces, from the standalone holosd_final.py policy (reusing its deploy/goal/metric
helpers), the same artifacts the archived ideas4 eval produced for i4_sepchan:

    eval/curves.png            -- Phase A volume + Cov eigenvalues, Phase B reward (train_log.jsonl)
    eval/nav_random.png        -- goal-nav trajectories (random / ring / ring_hard), drawn to closest step
    eval/nav_ring.png
    eval/nav_ring_hard.png
    eval/reach_vs_radius.png    -- reach as goal radius grows
    eval/eigenskill.png + .json -- drive the policy to each Cov eigen-axis pose (forward/lateral/rotation)
    eval/nav_random_grid.mp4    -- 3x3 full-horizon rollout videos
    eval/nav_ring_grid.mp4
    eval/nav_ring_hard_grid.mp4

CPU-only (small; spins up fast):
    JAX_PLATFORMS=cpu .venv/bin/python -m holosd.bicycle.eval_plots exp/bicycle/HoloSD/ckpt_final.pkl
"""
import os, sys, json, pickle, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from holosd.bicycle import holosd_final as F
from holosd.bicycle import chart as C

PI = np.pi
Z_RAD = 0.9   # ideas config default: ellipsoid command radius z = z_rad*sqrt(lam_i)*v_i


# ----------------------------------------------------------------- static figures
def plot_nav_closest(states, goals, succ, rho, path, title):
    """Each trajectory drawn only up to its closest-approach step (no post-arrival tail)."""
    d = np.linalg.norm(states[:, :, :2] - goals[:, None, :2], axis=-1)
    tc = d.argmin(1); ok = d.min(1) < succ
    fig, ax = plt.subplots(figsize=(7.5, 7.5)); th = np.linspace(0, 2 * PI, 100)
    ax.plot(rho * np.cos(th), rho * np.sin(th), "k--", lw=0.8, alpha=0.4)
    for i in range(min(len(goals), 40)):
        ax.plot(states[i, :tc[i] + 1, 0], states[i, :tc[i] + 1, 1],
                color=("tab:blue" if ok[i] else "tab:red"), lw=0.7, alpha=0.6)
    ax.scatter(goals[:, 0], goals[:, 1], marker="*", s=80,
               c=["tab:blue" if o else "tab:red" for o in ok], edgecolors="k", zorder=5)
    ax.plot(0, 0, "ko", ms=5); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"{title}: {ok.mean()*100:.0f}% reach <{succ:.2f} (traj to closest step)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_curves(log_path, out):
    if not os.path.exists(log_path):
        return
    rows = [json.loads(l) for l in open(log_path)]
    A = [r for r in rows if r.get("phase") == "A"]
    B = [r for r in rows if r.get("phase") == "B" and "r" in r]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    if A:
        ax[0].plot([r["epoch"] for r in A], [r["vol"] for r in A])
        ax[0].set_title("Phase A: -logdet volume"); ax[0].set_xlabel("epoch")
        eig = np.array([r["cov_eig"] for r in A])
        for k in range(eig.shape[1]):
            ax[1].plot([r["epoch"] for r in A], eig[:, k], label=f"eig{k+1}")
        ax[1].set_title("Phase A: Cov eigenvalues (ball-box)"); ax[1].legend(fontsize=7)
    if B:
        ax[2].plot([r["env_steps"] for r in B], [r["r"] for r in B])
        ax[2].set_title("Phase B: reward"); ax[2].set_xlabel("env steps")
    fig.tight_layout(); fig.savefig(os.path.join(out, "curves.png"), dpi=120); plt.close(fig)


# ----------------------------------------------------------------- B reward into train_log (clean log omits it)
def ensure_reward_log(ckpt_dir):
    """holosd_final logs Phase B rows without 'r'; recover (env_steps, r) from the text train log if present."""
    log_path = os.path.join(ckpt_dir, "train_log.jsonl")
    rows = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
    if any(rr.get("phase") == "B" and "r" in rr for rr in rows):
        return log_path
    import re, glob
    txt = None
    for cand in glob.glob(os.path.join(ckpt_dir, "*.log")) + \
            glob.glob(os.path.join(os.path.dirname(ckpt_dir), "*.log")):
        if any("[B]" in ln for ln in open(cand)):
            txt = cand; break
    if txt is None:
        return log_path
    bsteps = [(rr["epoch"], rr["env_steps"]) for rr in rows if rr.get("phase") == "B" and "env_steps" in rr]
    step_at = dict(bsteps)
    merged = os.path.join(ckpt_dir, "eval", "_curves_log.jsonl"); os.makedirs(os.path.dirname(merged), exist_ok=True)
    with open(merged, "w") as f:
        for rr in rows:
            if rr.get("phase") != "B" or "r" in rr:
                f.write(json.dumps(rr) + "\n")
        for ln in open(txt):
            m = re.search(r"\[B\]\s+(\d+)\s*\|\s*r\s*([+-]?[\d.]+)\s*q\s*([+-]?[\d.]+)", ln)
            if m:
                ep = int(m.group(1)); es = step_at.get(ep, ep * 1600)
                f.write(json.dumps(dict(phase="B", epoch=ep, env_steps=es,
                                        r=float(m.group(2)), q=float(m.group(3)))) + "\n")
    return merged


# ----------------------------------------------------------------- eigen-skill probe
def recompute_cov(cfg, pl, rng):
    """Cov_mu[psi(s,s')] (full matrix -> eigenvectors) by replaying the Phase-A data mu anchor."""
    net = F.Psi(cfg.phi_hidden, cfg.d); psi_p = pl["psi_p"]
    gen = F.build_mu(cfg); a_s, _ = gen(np.random.default_rng(cfg.seed), cfg.n_mu)
    S = a_s; n, Tp1 = S.shape[0], S.shape[1]; T = Tp1 - 1
    e = rng.integers(0, n, 4096); t0 = rng.integers(0, T, 4096)
    sep = rng.integers(1, T + 1, 4096); t1 = np.minimum(t0 + sep, T)
    end = np.asarray(net.apply(psi_p, jnp.asarray(F.pose_feat(jnp.asarray(S[e, t0]))),
                               jnp.asarray(F.pose_feat(jnp.asarray(S[e, t1])))))
    return np.cov(end.T) + 1e-4 * np.eye(cfg.d)


def fit_pose_to_chart(cfg, pl, rho, rng):
    """Linear map psi(0, g) ~ M @ [x,y,theta] + b over sampled goal poses (origin reference)."""
    net = F.Psi(cfg.phi_hidden, cfg.d); psi_p = pl["psi_p"]
    n = 4000
    r = 2.5 * rho * np.sqrt(rng.uniform(0, 1, n)); a = rng.uniform(0, 2 * PI, n)
    g = np.zeros((n, 4), np.float32)
    g[:, 0] = r * np.cos(a); g[:, 1] = r * np.sin(a); g[:, 2] = rng.uniform(-PI, PI, n)
    zero = np.zeros((n, 4), np.float32)
    ph = np.asarray(net.apply(psi_p, jnp.asarray(F.pose_feat(jnp.asarray(zero))),
                              jnp.asarray(F.pose_feat(jnp.asarray(g)))))
    X = np.c_[g[:, 0], g[:, 1], (g[:, 2] + PI) % (2 * PI) - PI, np.ones(n)]
    P, *_ = np.linalg.lstsq(X, ph, rcond=None)
    return P[:3, :].T, P[3, :]


def eigenskill(cfg, pl, deploy, rho, out):
    rng = np.random.default_rng(0)
    Cov = recompute_cov(cfg, pl, rng)
    w, V = np.linalg.eigh(Cov); order = np.argsort(w)[::-1]; lam = w[order]; Vec = V[:, order]
    M, b = fit_pose_to_chart(cfg, pl, rho, rng); Minv = np.linalg.pinv(M); d = cfg.d

    goals, meta = [], []
    for i in range(d):
        for sg in (1.0, -1.0):
            z = sg * Z_RAD * np.sqrt(lam[i]) * Vec[:, i]
            pose = Minv @ (z - b); xy = pose[:2]; nrm = np.linalg.norm(xy) + 1e-9
            if nrm > 1.8 * rho:
                xy = xy * (1.8 * rho / nrm)
            goals.append([xy[0], xy[1], (pose[2] + PI) % (2 * PI) - PI]); meta.append((i, sg))
    goals = np.array(goals, np.float32)
    states = deploy(goals)

    def classify(dx, dy, dth):
        return max({"forward": abs(dx), "lateral": abs(dy), "rotation": abs(dth) * rho}.items(),
                   key=lambda kv: kv[1])[0]

    fig, axes = plt.subplots(d, 2, figsize=(10.5, 4.4 * d), squeeze=False); thg = np.linspace(0, 2 * PI, 120)
    table = {}
    for k, (i, sg) in enumerate(meta):
        ax = axes[i, 0 if sg > 0 else 1]
        x, y = states[k, :, 0], states[k, :, 1]; thf = float(C.wrap_pi(states[k, -1, 2])); gth = goals[k, 2]
        ax.plot(rho * np.cos(thg), rho * np.sin(thg), "k:", lw=0.7, alpha=0.4)
        ax.plot(x, y, lw=1.4, color="tab:blue")
        for k2 in range(0, len(x) - 1, max(1, len(x) // 12)):
            hl = 0.13 * rho
            ax.annotate("", xy=(x[k2] + hl * np.cos(states[k, k2, 2]), y[k2] + hl * np.sin(states[k, k2, 2])),
                        xytext=(x[k2], y[k2]), arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))
        ax.plot(0, 0, "ko", ms=5); ax.plot(goals[k, 0], goals[k, 1], "g*", ms=13)
        ax.annotate("", xy=(goals[k, 0] + 0.2 * rho * np.cos(gth), goals[k, 1] + 0.2 * rho * np.sin(gth)),
                    xytext=(goals[k, 0], goals[k, 1]), arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.4))
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        dom = classify(x[-1], y[-1], thf)
        ax.set_title(f"axis {i+1} ({'+' if sg>0 else '-'}) lam={lam[i]:.0f} -> {dom}\n"
                     f"goal=({goals[k,0]:+.2f},{goals[k,1]:+.2f},{gth:+.2f}) "
                     f"reach=({x[-1]:+.2f},{y[-1]:+.2f},{thf:+.2f})", fontsize=9)
        if sg > 0:
            table[i] = dict(idx=i + 1, lam=float(lam[i]), maneuver=dom, goal=[float(v) for v in goals[k]],
                            reached=[float(x[-1]), float(y[-1]), thf])
    fig.suptitle(f"{cfg.exp_name}: eigen-skill = drive policy to each Cov eigen-axis POSE "
                 f"(green * = commanded pose+heading; effrank {(lam.sum()**2)/(lam**2).sum():.2f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(out, "eigenskill.png"), dpi=120); plt.close(fig)
    json.dump({"cov_eig": [float(x) for x in lam], "axes": [table[i] for i in range(d)]},
              open(os.path.join(out, "eigenskill.json"), "w"), indent=2)
    print("[eigenskill]", cfg.exp_name)
    for i in range(d):
        t = table[i]
        print(f"  ax{t['idx']} lam{t['lam']:.0f} {t['maneuver']:8s} "
              f"goal{tuple(round(v,2) for v in t['goal'])} -> reach{tuple(round(v,2) for v in t['reached'])}")


# ----------------------------------------------------------------- 3x3 grid nav video
def _draw_cell(ax, xi, goal, ok, rho, t):
    ax.clear(); th = np.linspace(0, 2 * PI, 80)
    ax.plot(rho * np.cos(th), rho * np.sin(th), "k--", lw=0.5, alpha=0.35)
    col = "tab:blue" if ok else "tab:red"
    ax.plot(xi[:t + 1, 0], xi[:t + 1, 1], color=col, lw=1.4)
    cx, cy, cth = xi[t]; hl = 0.16 * max(rho, 0.6)
    ax.annotate("", xy=(cx + hl * np.cos(cth), cy + hl * np.sin(cth)), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.8), zorder=6)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=130, c=col, edgecolors="k", zorder=5)
    ax.annotate("", xy=(goal[0] + hl * np.cos(goal[2]), goal[1] + hl * np.sin(goal[2])),
                xytext=(goal[0], goal[1]), arrowprops=dict(arrowstyle="->", color="k", lw=0.8), zorder=6)
    ax.plot(0, 0, "ko", ms=3)
    allx = np.concatenate([xi[:, 0], [goal[0], 0.0]]); ally = np.concatenate([xi[:, 1], [goal[1], 0.0]])
    cx0, cy0 = allx.mean(), ally.mean()
    span = max(allx.max() - allx.min(), ally.max() - ally.min(), 0.6 * rho) * 0.62 + 0.25
    ax.set_xlim(cx0 - span, cx0 + span); ax.set_ylim(cy0 - span, cy0 + span)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def make_grid_video(trajs, goals, ok, rho, path, title, fps=20, stride=3, hold=0.8):
    trajs = np.asarray(trajs); n = min(9, len(trajs)); L = trajs.shape[1]
    fig = Figure(figsize=(9.0, 9.4), dpi=100); canvas = FigureCanvasAgg(fig)
    axes = [fig.add_subplot(3, 3, i + 1) for i in range(9)]
    ts = list(range(0, L, stride))
    if ts[-1] != L - 1:
        ts.append(L - 1)
    frames = []
    for t in ts:
        for i in range(9):
            if i < n:
                _draw_cell(axes[i], trajs[i], goals[i], bool(ok[i]), rho, t)
            else:
                axes[i].clear(); axes[i].axis("off")
        fig.suptitle(f"{title}   ({ok[:n].mean()*100:.0f}% reach)   t={t}/{L-1}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); canvas.draw()
        w, h = canvas.get_width_height()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        frames.append(img)
    frames += [frames[-1]] * int(fps * hold)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=None)


# ----------------------------------------------------------------- driver
def main():
    a = argparse.ArgumentParser(); a.add_argument("ckpt"); a.add_argument("--T", type=int, default=300)
    a.add_argument("--no_video", action="store_true"); args = a.parse_args()
    pl = pickle.load(open(args.ckpt, "rb"))
    cfg = F.HoloSDConfig(**{k: v for k, v in pl["cfg"].items() if k in F.HoloSDConfig.__dataclass_fields__})
    rho = cfg.rho; succ = 0.2 * rho; ckpt_dir = os.path.dirname(args.ckpt)
    out = os.path.join(ckpt_dir, "eval"); os.makedirs(out, exist_ok=True)
    deploy = F.deploy_fn(cfg, pl, args.T)
    name = cfg.exp_name
    print(f"[eval_plots] {name}  -> {out}", flush=True)

    # 1) nav trajectory figures + 2) reach vs radius
    rng = np.random.default_rng(0)
    suites = [("nav_random", F.rand_goals(rng, 48, 0.7 * rho)),
              ("nav_ring", F.ring_goals(12, 0.7 * rho)),
              ("nav_ring_hard", F.ring_hard(12, 0.7 * rho))]
    for tag, goals in suites:
        st = deploy(goals); m = F.nav_metrics(st, goals, succ)
        plot_nav_closest(st, goals, succ, rho, os.path.join(out, f"{tag}.png"), f"{name} {tag[4:]}")
        print(f"  {tag:14s}: reach {m['reach']*100:3.0f}%  pose {m['pose']*100:3.0f}%  -> {tag}.png", flush=True)

    fr = np.linspace(0.4, 4.0, 10); rr = []
    for f in fr:
        g = F.rand_goals(np.random.default_rng(7), 32, f * rho)
        g[:, :2] *= (f * rho) / (np.linalg.norm(g[:, :2], axis=1, keepdims=True) + 1e-9)
        rr.append(F.nav_metrics(deploy(g.astype(np.float32)), g, succ)["reach"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); ax.plot(fr, rr, "o-"); ax.axvline(2.0, color="0.8", ls=":")
    ax.set_xlabel("goal radius / rho"); ax.set_ylabel("reach"); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
    ax.set_title(f"{name}: reach vs radius")
    fig.tight_layout(); fig.savefig(os.path.join(out, "reach_vs_radius.png"), dpi=120); plt.close(fig)
    print(f"  reach_vs_radius.png", flush=True)

    # 3) training curves
    plot_curves(ensure_reward_log(ckpt_dir), out)
    print(f"  curves.png", flush=True)

    # 4) eigen-skill probe
    eigenskill(cfg, pl, deploy, rho, out)
    print(f"  eigenskill.png + eigenskill.json", flush=True)

    # 5) nav grid videos
    if not args.no_video:
        rng = np.random.default_rng(1)
        for tag, goals in (("nav_random", F.rand_goals(rng, 9, 0.7 * rho)),
                           ("nav_ring", F.ring_goals(9, 0.7 * rho)),
                           ("nav_ring_hard", F.ring_hard(9, 0.7 * rho))):
            st = deploy(goals); xi = np.array(st[:, :, :3])
            ok = (np.linalg.norm(xi[:, :, :2] - goals[:, None, :2], axis=-1).min(1) < succ)
            path = os.path.join(out, f"{tag}_grid.mp4")
            make_grid_video(xi, goals, ok, rho, path, f"{name} {tag[4:]}")
            print(f"  {tag}_grid.mp4  ({ok.mean()*100:.0f}% reach)", flush=True)

    print(f"  -> all artifacts in {out}/", flush=True)


if __name__ == "__main__":
    main()
