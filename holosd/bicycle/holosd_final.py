"""HoloSD — final, self-contained algorithm (cleaned).

The adopted HoloSD (developed as i4_sepchan / HoloSD-v6): a max-volume effort-cocycle chart
discovered from reward-free data (Phase A), driven to goal poses by a chart POSITION carrot plus a
SEPARATE RAW HEADING channel (Phase B).  See docs/algorithms/HoloSD_ALGORITHM.md.

This module is dependency-minimal: it imports only the stable core (env, models, chart.wrap_pi) and
inlines everything else, so the whole `ideas/` experiment lineage can be archived.  Every constant that
was provably inert in the adopted config has been removed:
  - BCH structure tensor M + bch_init   (use_bch was False -> additive cocycle, M never applied)
  - Cov_half / z-address machinery       (z_rad, cov_metric_z -- Phase B is goal-conditioned)
  - trace penalty (trace_w, trace_s0), effort_cov + arc-length (cum_arc), time-Lipschitz lip_c
  - rotation-injection data knobs (rot_data, rot_frac, rot_dc, rot_inphase)
  - curriculum staging (rho_stages[:-1], epochs_per_stage), randomize_theta0, and all sibling-idea flags
The math of the live path is unchanged and the RNG usage mirrors the original, so retraining reaches
equivalent goal-nav metrics.

Train:  JAX_PLATFORMS=cuda HOLOSD_GPU=0 .venv/bin/python -m holosd.bicycle.holosd_final train \
            --out_dir exp/bicycle/HoloSD_clean
Eval :  JAX_PLATFORMS=cpu .venv/bin/python -m holosd.bicycle.holosd_final eval \
            exp/bicycle/HoloSD_clean/ckpt_final.pkl
"""
import os, sys, math, json, time, pickle, argparse
from dataclasses import dataclass, asdict
import numpy as np


def _gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("HOLOSD_GPU", "0")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")
_gpu()

import jax, jax.numpy as jnp, optax
import flax.linen as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from holosd.bicycle import env as E
from holosd.bicycle import chart as C
from holosd import models as M

PI = np.pi


# ===================================================================== config (minimal, live-only)
@dataclass
class HoloSDConfig:
    exp_name: str = "HoloSD_clean"
    out_dir: str = "exp/bicycle/HoloSD_clean"
    seed: int = 0
    quick: bool = False

    # --- env (R_min = L/tan(phi_max) = 1; rho = 0.5) ---
    dt: float = 0.1
    v_max: float = 1.0
    wheelbase: float = 0.5
    phi_max: float = math.atan(0.5)
    dphi_max: float = 1.5
    T: int = 200
    rho: float = 0.5

    # --- chart / Phase A geometry ---
    d: int = 3                  # chart dimension
    phi_hidden: int = 256       # chart MLP width (2 layers)
    phi_lr: float = 1e-3
    lip_w: float = 10.0
    cocycle_w: float = 1.0
    n_mu: int = 256             # reward-free random-oscillator episodes (data mu)
    phaseA_epochs: int = 1800

    # --- policy (SAC residual; oscillator carrier optional) ---
    osc_K: int = 3              # oscillator bank order (data mu rollouts always use it)
    osc_gain: float = 0.95
    with_prior: bool = False    # canonical HoloSD = pure SAC residual (NO oscillator
                                # carrier). data mu for Phase A is still oscillator-driven.
                                # Set True for the archived oscillator-bank policy variant.
    hidden: int = 512
    n_layers: int = 2

    # --- Phase B control / SAC ---
    phaseB_epochs: int = 3000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_sac: int = 256
    buffer_episodes: int = 2000
    target_entropy: float = -2.0
    init_alpha: float = 0.1
    episodes_per_epoch: int = 8
    grad_steps_per_epoch: int = 64
    init_random_epochs: int = 5
    log_every: int = 10

    @property
    def Tc(self) -> float:
        return float(np.clip(2.0 * PI * self.rho / (self.v_max * self.dt), 20.0, 120.0))

    def envp(self) -> E.EnvParams:
        return E.EnvParams(self.dt, self.v_max, self.wheelbase, self.phi_max, self.dphi_max, self.T)


# ===================================================================== nets / features
class Psi(nn.Module):
    """Two-point chart psi(s, s') -> R^d (learned log-map / privileged coords)."""
    hidden: int; d: int

    @nn.compact
    def __call__(self, xa, xb):
        return M.MLP(self.hidden, 2, self.d)(jnp.concatenate([xa, xb], -1))


class OscActor(nn.Module):
    """SAC actor: y = osc_prior(coef, t) + tanh(u), u ~ N(mean(obs), std(obs)).
    coef = oscillator-bank coefficients = gain * tanh(MLP(ctx)); ctx EXCLUDES the clock."""
    hidden: int; n_layers: int; act_dim: int; K: int; gain: float

    @nn.compact
    def __call__(self, obs, ctx):
        h = M.MLP(self.hidden, self.n_layers, 2 * self.act_dim)(obs)
        mean, ls = jnp.split(h, 2, -1)
        ls = jnp.clip(ls, M.LOG_STD_MIN, M.LOG_STD_MAX)
        coef = self.gain * jnp.tanh(M.MLP(64, 2, self.act_dim * (1 + 2 * self.K))(ctx))
        return mean, ls, coef


def osc_prior(coef, t, Tc, K, xp=jnp):
    c = coef.reshape(coef.shape[:-1] + (-1, 1 + 2 * K))
    ph = 2.0 * PI * t / Tc
    out = c[..., 0]
    for k in range(1, K + 1):
        out = out + c[..., 2 * k - 1] * xp.sin(k * ph[..., None]) + c[..., 2 * k] * xp.cos(k * ph[..., None])
    return out


def pose_feat(state, xp=jnp):
    """chart input: [x, y, cos th, sin th]."""
    return xp.stack([state[..., 0], state[..., 1], xp.cos(state[..., 2]), xp.sin(state[..., 2])], -1)


def carrot(u, xp=jnp):
    """Pi_{B(1)}: clip code-space error into the unit ball."""
    n = xp.sqrt(xp.sum(u * u, -1, keepdims=True)) + 1e-9
    return u * xp.minimum(1.0, 1.0 / n)


def obs_vec(state, vprev, ph, e, d, hraw, cfg, xp=jnp):
    """policy/critic obs: [cos th, sin th, phi/phi_max, v_prev, cos clk, sin clk, carrot(d), dist, hraw]."""
    base = xp.stack([xp.cos(state[..., 2]), xp.sin(state[..., 2]), state[..., 3] / cfg.phi_max,
                     vprev, xp.cos(ph), xp.sin(ph)], -1)
    return xp.concatenate([base, e, d[..., None], hraw[..., None]], -1)


def ctx_vec(state, vprev, e, d, hraw, cfg, xp=jnp):
    """oscillator-coefficient context (NO clock): [carrot(d), dist, cos th, sin th, phi/phi_max, v_prev, hraw]."""
    base = xp.stack([xp.cos(state[..., 2]), xp.sin(state[..., 2]), state[..., 3] / cfg.phi_max, vprev], -1)
    return xp.concatenate([e, d[..., None], base, hraw[..., None]], -1)


def obs_dim(cfg): return 7 + cfg.d + 1
def ctx_dim(cfg): return 5 + cfg.d + 1


def prior_term(coef, t, Tc, cfg, xp=jnp):
    """Policy oscillator carrier; 0 for canonical HoloSD (pure SAC residual). Only the
    with_prior variant adds it. Phase-A data mu always uses osc_prior regardless."""
    return osc_prior(coef, t, Tc, cfg.osc_K, xp) if cfg.with_prior else 0.0


# ===================================================================== Phase A: data mu
def build_mu(cfg):
    """Reward-free random-oscillator-bank rollouts (random per-episode coefficients, random theta0)."""
    p = cfg.envp(); Tc = cfg.Tc; K = cfg.osc_K; nc = E.ACT_DIM * (1 + 2 * K)

    def single(coef, theta0):
        state0 = jnp.array([0.0, 0.0, theta0, 0.0])

        def step(carry, t):
            state, _ = carry
            y = osc_prior(coef, t.astype(jnp.float32), Tc, cfg.osc_K)
            return (E.env_step(state, y, p), jnp.clip(y[0], -1, 1)), (state, y)
        (last, _), (states, ys) = jax.lax.scan(step, (state0, jnp.array(0.0)), jnp.arange(cfg.T))
        return jnp.concatenate([states, last[None]], 0), ys
    batched = jax.jit(jax.vmap(single, in_axes=(0, 0)))

    def gen(rng, n):
        coef = (cfg.osc_gain * rng.uniform(-1, 1, (n, nc))).astype(np.float32)
        th0 = rng.uniform(-PI, PI, n).astype(np.float32)
        s, y = batched(jnp.asarray(coef), jnp.asarray(th0))
        return np.asarray(s), np.asarray(y)
    return gen


def lip_bound(cfg, y_t):
    """Per-step effort budget for the Lipschitz hinge: cap = |v| (per-step arc-length rate)."""
    return (cfg.v_max * np.abs(np.clip(y_t[..., 0], -1, 1))).astype(np.float32)


def geom_batch(cfg, anchor, rng):
    """Multiscale pairs (volume), transitions (effort-Lip), ordered triples (cocycle)."""
    S, Y = anchor; n, Tp1 = S.shape[0], S.shape[1]; T = Tp1 - 1
    out = {}
    e = rng.integers(0, n, 1024)
    t0 = rng.integers(0, T, 1024); sep = rng.integers(1, T + 1, 1024); t1 = np.minimum(t0 + sep, T)
    out['pa0'] = S[e, t0]; out['pa1'] = S[e, t1]
    el = rng.integers(0, n, 512); tl = rng.integers(0, T, 512)
    out['lt0'] = S[el, tl]; out['lt1'] = S[el, tl + 1]; out['lipb'] = lip_bound(cfg, Y[el, tl])
    et = rng.integers(0, n, 512); tu = np.sort(rng.integers(0, Tp1, (512, 3)), 1)
    out['ca'] = S[et, tu[:, 0]]; out['cb'] = S[et, tu[:, 1]]; out['cc'] = S[et, tu[:, 2]]
    return {k: v.astype(np.float32) for k, v in out.items()}


def build_geom_train(cfg):
    """max logdet Cov_mu[psi(pair)] s.t. effort-Lipschitz + additive cocycle."""
    d = cfg.d
    net = Psi(cfg.phi_hidden, d)
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(cfg.phi_lr))

    def loss(psi_p, b):
        ap = lambda xa, xb: net.apply(psi_p, pose_feat(xa), pose_feat(xb))
        end = ap(b['pa0'], b['pa1'])
        cov = jnp.cov(end.T) + 1e-4 * jnp.eye(d)
        vol = -jnp.linalg.slogdet(cov)[1]
        step = ap(b['lt0'], b['lt1'])
        sn = jnp.sqrt(jnp.sum(step ** 2, -1) + 1e-12)
        lip = jnp.mean(jnp.maximum(0.0, sn - b['lipb']) ** 2)
        pab, pbc, pac = ap(b['ca'], b['cb']), ap(b['cb'], b['cc']), ap(b['ca'], b['cc'])
        coc = jnp.mean(jnp.sum((pac - (pab + pbc)) ** 2, -1))           # additive cocycle (no BCH)
        L = vol + cfg.lip_w * lip + cfg.cocycle_w * coc
        return L, (vol, lip, coc)

    @jax.jit
    def update(psi_p, ostate, b):
        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(psi_p, b)
        u, ostate = opt.update(g, ostate, psi_p)
        return optax.apply_updates(psi_p, u), ostate, l, aux
    return net, opt, update


def compute_eig(cfg, net, psi_p, anchor, rng):
    """Cov_mu[psi(s,s')] eigenvalues (controllability ellipsoid) + dscale = sqrt(mean eig)."""
    S = anchor[0]; n, Tp1 = S.shape[0], S.shape[1]; T = Tp1 - 1
    e = rng.integers(0, n, 4096); t0 = rng.integers(0, T, 4096)
    sep = rng.integers(1, T + 1, 4096); t1 = np.minimum(t0 + sep, T)
    end = np.asarray(net.apply(psi_p, jnp.asarray(pose_feat(jnp.asarray(S[e, t0]))),
                               jnp.asarray(pose_feat(jnp.asarray(S[e, t1])))))
    Cov = np.cov(end.T) + 1e-4 * np.eye(cfg.d)
    eig = np.sort(np.linalg.eigvalsh(Cov))[::-1]
    return eig.astype(np.float32)


# ===================================================================== Phase B: SAC over (carrot + raw heading)
class Buffer:
    def __init__(self, cap, T):
        self.cap, self.T = cap, T; self.ptr = self.size = 0
        self.states = np.zeros((cap, T + 1, E.STATE_DIM), np.float32)
        self.y = np.zeros((cap, T, E.ACT_DIM), np.float32)
        self.g = np.zeros((cap, 3), np.float32)

    def add(self, states, ys, g):
        n = states.shape[0]
        for i in range(n):
            j = (self.ptr + i) % self.cap
            self.states[j] = states[i]; self.y[j] = ys[i]; self.g[j] = g[i]
        self.ptr = (self.ptr + n) % self.cap; self.size = min(self.size + n, self.cap)


def init_state(cfg, key):
    act = OscActor(cfg.hidden, cfg.n_layers, E.ACT_DIM, cfg.osc_K, cfg.osc_gain)
    crit = M.DoubleCritic(cfg.hidden, cfg.n_layers)
    od, cd = obs_dim(cfg), ctx_dim(cfg)
    ks = jax.random.split(key, 2)
    act_p = act.init(ks[0], jnp.zeros((1, od)), jnp.zeros((1, cd)))
    q_p = crit.init(ks[1], jnp.zeros((1, od)), jnp.zeros((1, E.ACT_DIM)))
    clip = lambda lr: optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    ts = dict(act_p=act_p, act_o=clip(cfg.lr).init(act_p), q_p=q_p, q_o=clip(cfg.lr).init(q_p),
              qt_p=q_p, log_alpha=jnp.array(np.log(cfg.init_alpha), jnp.float32))
    ts["alpha_o"] = optax.adam(cfg.lr).init(ts["log_alpha"])
    return ts


def build_update(cfg):
    act = OscActor(cfg.hidden, cfg.n_layers, E.ACT_DIM, cfg.osc_K, cfg.osc_gain)
    crit = M.DoubleCritic(cfg.hidden, cfg.n_layers); Hent = cfg.target_entropy; Tc = cfg.Tc
    clip = lambda lr: optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    o_act, o_q, o_alpha = clip(cfg.lr), clip(cfg.lr), optax.adam(cfg.lr)

    def pol(act_p, obs, ctx, t, key):
        mean, ls, coef = act.apply(act_p, obs, ctx)
        u, logp = M.sample_action(mean, ls, key)
        return prior_term(coef, t, Tc, cfg) + u, logp

    def critic_loss(q_p, s, y, tq):
        q1, q2 = crit.apply(q_p, s, y); return jnp.mean((q1 - tq) ** 2 + (q2 - tq) ** 2)

    def actor_loss(act_p, q_p, la, s, ctx, t, key):
        y, logp = pol(act_p, s, ctx, t, key); q1, q2 = crit.apply(q_p, s, y)
        return jnp.mean(jnp.exp(la) * logp - jnp.minimum(q1, q2)), logp

    def step(carry, b):
        ts, key = carry; key, ka, kt = jax.random.split(key, 3)
        y1, logp1 = pol(ts['act_p'], b['s1'], b['ctx1'], b['t1'], ka)
        q1n, q2n = crit.apply(ts['qt_p'], b['s1'], y1); alpha = jnp.exp(ts['log_alpha'])
        tq = jax.lax.stop_gradient(b['r'] + cfg.gamma * b['mask'] * (jnp.minimum(q1n, q2n) - alpha * logp1))
        c_loss, c_grad = jax.value_and_grad(critic_loss)(ts['q_p'], b['s'], b['y'], tq)
        cupd, q_o = o_q.update(c_grad, ts['q_o'], ts['q_p']); q_p = optax.apply_updates(ts['q_p'], cupd)
        (a_loss, logp), a_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            ts['act_p'], q_p, ts['log_alpha'], b['s'], b['ctx'], b['t'], kt)
        aupd, act_o = o_act.update(a_grad, ts['act_o'], ts['act_p']); act_p = optax.apply_updates(ts['act_p'], aupd)
        alupd, alpha_o = o_alpha.update(
            jax.grad(lambda la, lp: jnp.mean(-la * jax.lax.stop_gradient(lp + Hent)))(ts['log_alpha'], logp),
            ts['alpha_o'], ts['log_alpha'])
        log_alpha = optax.apply_updates(ts['log_alpha'], alupd); qt_p = optax.incremental_update(q_p, ts['qt_p'], cfg.tau)
        return (dict(act_p=act_p, act_o=act_o, q_p=q_p, q_o=q_o, qt_p=qt_p, log_alpha=log_alpha, alpha_o=alpha_o), key), \
            dict(r=jnp.mean(b['r']), q=jnp.mean(jnp.minimum(q1n, q2n)))

    @jax.jit
    def run_epoch(ts, key, batches):
        (ts2, key2), met = jax.lax.scan(step, (ts, key), batches); return ts2, key2, met
    return run_epoch


def sample_pose_goals(cfg, rng, n):
    r = cfg.rho * (0.3 + 2.7 * rng.uniform(0, 1, n)); a = rng.uniform(0, 2 * PI, n)
    return np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-PI, PI, n)], -1).astype(np.float32)


def build_control(cfg, net):
    """Closed-loop rollout: position carrot = Pi(psi(s_t, g)); raw heading hraw = wrap(th_g - th)/pi."""
    act = OscActor(cfg.hidden, cfg.n_layers, E.ACT_DIM, cfg.osc_K, cfg.osc_gain)
    p = cfg.envp(); Tc = cfg.Tc

    def make_single(det):
        def single(act_p, psi_p, key, goal):
            fg = pose_feat(jnp.concatenate([goal, jnp.zeros(1)])[None])[0]

            def stepf(carry, t):
                state, vprev, k = carry; k, ka = jax.random.split(k)
                rem = net.apply(psi_p, pose_feat(state[None])[0], fg)
                e = carrot(rem, jnp); d = jnp.sqrt(jnp.sum(rem * rem, -1) + 1e-12)
                hraw = C.wrap_pi(goal[2] - state[2], jnp) / PI
                ph = 2.0 * PI * t / Tc
                obs = obs_vec(state, vprev, ph, e, d, hraw, cfg); ctx = ctx_vec(state, vprev, e, d, hraw, cfg)
                mean, ls, coef = act.apply(act_p, obs, ctx)
                u = jnp.tanh(mean) if det else M.sample_action(mean, ls, ka)[0]
                y = prior_term(coef, t.astype(jnp.float32), Tc, cfg) + u
                return (E.env_step(state, y, p), jnp.clip(y[0], -1, 1), k), (state, y)
            (last, _, _), (states, ys) = jax.lax.scan(stepf, (jnp.zeros(4), jnp.array(0.0), key), jnp.arange(cfg.T))
            return jnp.concatenate([states, last[None]], 0), ys
        return jax.jit(jax.vmap(single, in_axes=(None, None, 0, 0)))

    bs, bd = make_single(False), make_single(True)

    def collect(act_p, psi_p, key, goals, det):
        return (bd if det else bs)(act_p, psi_p, jax.random.split(key, goals.shape[0]), goals)
    return collect


def control_batch(cfg, net, pbuf, rng, psi_p, dscale):
    G, B, T = cfg.grad_steps_per_epoch, cfg.batch_sac, pbuf.T
    e = rng.integers(0, pbuf.size, (G, B)); t = rng.integers(0, T, (G, B))
    st = pbuf.states[e, t]; st1 = pbuf.states[e, t + 1]; g = pbuf.g[e]
    vp = np.where(t > 0, np.clip(pbuf.y[e, np.maximum(t - 1, 0), 0], -1, 1), 0.0); vp1 = np.clip(pbuf.y[e, t, 0], -1, 1)
    gf = pose_feat(np.concatenate([g.reshape(-1, 3), np.zeros((G * B, 1), np.float32)], -1), np)

    def comp(states):
        rem = np.asarray(net.apply(psi_p, jnp.asarray(pose_feat(jnp.asarray(states.reshape(-1, 4)))), jnp.asarray(gf)))
        d = np.sqrt((rem * rem).sum(-1) + 1e-12); ee = rem * np.minimum(1.0, 1.0 / (d[:, None] + 1e-9))
        hraw = C.wrap_pi(g.reshape(-1, 3)[:, 2] - states.reshape(-1, 4)[:, 2]) / PI
        sh = states.shape[:-1]
        return ee.reshape(sh + (-1,)), d.reshape(sh), hraw.reshape(sh)
    e_t, d_t, h_t = comp(st); e_t1, d_t1, h_t1 = comp(st1)
    r = (d_t - d_t1) + dscale * (np.abs(h_t) - np.abs(h_t1))
    ph, ph1 = 2 * PI * t / cfg.Tc, 2 * PI * (t + 1) / cfg.Tc
    obs = obs_vec(st, vp, ph, e_t, d_t, h_t, cfg, np); obs1 = obs_vec(st1, vp1, ph1, e_t1, d_t1, h_t1, cfg, np)
    ctx = ctx_vec(st, vp, e_t, d_t, h_t, cfg, np); ctx1 = ctx_vec(st1, vp1, e_t1, d_t1, h_t1, cfg, np)
    return dict(s=obs.astype(np.float32), ctx=ctx.astype(np.float32), y=pbuf.y[e, t], r=r.astype(np.float32),
                s1=obs1.astype(np.float32), ctx1=ctx1.astype(np.float32), t=t.astype(np.float32),
                t1=(t + 1).astype(np.float32), mask=np.ones_like(r, np.float32))


# ===================================================================== train
def train(cfg):
    out = cfg.out_dir; os.makedirs(out, exist_ok=True)
    print(f"[{cfg.exp_name}] dev {jax.devices()} | HoloSD clean -> {out}", flush=True)
    key = jax.random.PRNGKey(cfg.seed); rng = np.random.default_rng(cfg.seed)
    gen = build_mu(cfg); a_s, a_y = gen(rng, cfg.n_mu); anchor = (a_s, a_y)
    logf = open(os.path.join(out, "train_log.jsonl"), "w")
    log = lambda dd: (logf.write(json.dumps(dd) + "\n"), logf.flush())
    t0 = time.time()

    # ---- Phase A: geometry ----
    net, gopt, gupd = build_geom_train(cfg)
    key, k1 = jax.random.split(key); dum = pose_feat(jnp.zeros((1, 4)))
    psi_p = net.init(k1, dum, dum); gostate = gopt.init(psi_p)
    print("=== Phase A (geometry) ===", flush=True)
    nA = 38 if cfg.quick else cfg.phaseA_epochs
    for ep in range(nA):
        b = jax.tree_util.tree_map(jnp.asarray, geom_batch(cfg, anchor, rng))
        psi_p, gostate, gl, aux = gupd(psi_p, gostate, b)
        if ep % 200 == 0 or ep == nA - 1:
            eig = compute_eig(cfg, net, psi_p, anchor, rng)
            print(f"  [A] {ep:4d} L {float(gl):.2f} vol {float(aux[0]):+.2f} lip {float(aux[1]):.4f} "
                  f"coc {float(aux[2]):.4f} | eig " + "/".join(f"{x:.1f}" for x in eig), flush=True)
            log(dict(phase="A", epoch=ep, L=float(gl), vol=float(aux[0]), lip=float(aux[1]), coc=float(aux[2]),
                     cov_eig=[float(x) for x in eig]))
    eig = compute_eig(cfg, net, psi_p, anchor, rng)
    dscale = float(np.sqrt(np.mean(eig)))
    print(f"  Phase-A eig {['%.1f'%x for x in eig]} | dscale {dscale:.2f}", flush=True)

    # ---- Phase B: control ----
    ts = init_state(cfg, k1)
    run_epoch = build_update(cfg)
    collect = build_control(cfg, net)
    pbuf = Buffer(cfg.buffer_episodes, cfg.T); gep = cfg.episodes_per_epoch; env_steps = 0
    print("=== Phase B (control) ===", flush=True)
    nB = 38 if cfg.quick else cfg.phaseB_epochs
    for ep in range(nB):
        goals = sample_pose_goals(cfg, rng, gep)
        key, kc = jax.random.split(key)
        ss, ys = collect(ts['act_p'], psi_p, kc, jnp.asarray(goals), False)
        ss, ys = np.asarray(ss), np.asarray(ys); pbuf.add(ss, ys, goals); env_steps += gep * cfg.T
        met = {}
        if ep >= cfg.init_random_epochs and pbuf.size >= 2:
            batches = jax.tree_util.tree_map(jnp.asarray, control_batch(cfg, net, pbuf, rng, psi_p, dscale))
            key, ku = jax.random.split(key); ts, key, m = run_epoch(ts, ku, batches)
            met = {k: float(np.asarray(v).mean()) for k, v in m.items()}
        if ep % cfg.log_every == 0 or ep == nB - 1:
            log(dict(phase="B", epoch=ep, env_steps=env_steps, **met))
            print(f"  [B] {ep:4d} | r {met.get('r',0):+.3f} q {met.get('q',0):+.2f}", flush=True)
    logf.close()
    pickle.dump(dict(act_p=jax.device_get(ts['act_p']), q_p=jax.device_get(ts['q_p']),
                     psi_p=jax.device_get(psi_p), cov_eig=[float(x) for x in eig], dscale=dscale,
                     cfg=asdict(cfg)), open(os.path.join(out, "ckpt_final.pkl"), "wb"))
    print(f"\n[{cfg.exp_name}] done {(time.time()-t0)/60:.1f} min", flush=True)


# ===================================================================== eval (goal nav: reach / pose / pose_final)
def rand_goals(rng, n, r):
    rr = r * np.sqrt(rng.uniform(0, 1, n)); a = rng.uniform(0, 2 * PI, n)
    return np.stack([rr * np.cos(a), rr * np.sin(a), rng.uniform(-PI, PI, n)], -1).astype(np.float32)


def ring_goals(k, r):
    a = np.linspace(0, 2 * PI, k, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a), a], -1).astype(np.float32)


def ring_hard(k, r):
    a = np.linspace(0, 2 * PI, k, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a), (a + PI) % (2 * PI) - PI], -1).astype(np.float32)


def nav_metrics(states, goals, succ):
    d = np.linalg.norm(states[:, :, :2] - goals[:, None, :2], axis=-1); closest = d.min(1); tc = d.argmin(1)
    head = np.abs((states[np.arange(len(goals)), tc, 2] - goals[:, 2] + PI) % (2 * PI) - PI)
    return dict(reach=float((closest < succ).mean()), pose=float(((closest < succ) & (head < 0.3)).mean()))


def pose_final(states, goals):
    dxy = np.linalg.norm(states[:, -1, :2] - goals[:, :2], axis=-1)
    dth = np.abs((states[:, -1, 2] - goals[:, 2] + PI) % (2 * PI) - PI)
    return float(((dxy < 0.3) & (dth < 0.3)).mean())


def deploy_fn(cfg, pl, nav_T):
    act = OscActor(cfg.hidden, cfg.n_layers, E.ACT_DIM, cfg.osc_K, cfg.osc_gain)
    net = Psi(cfg.phi_hidden, cfg.d)
    p = E.EnvParams(cfg.dt, cfg.v_max, cfg.wheelbase, cfg.phi_max, cfg.dphi_max, nav_T); Tc = cfg.Tc
    act_p, psi_p = pl["act_p"], pl["psi_p"]

    def single(goal):
        fg = pose_feat(jnp.concatenate([goal, jnp.zeros(1)])[None])[0]

        def stepf(carry, t):
            state, vprev = carry
            rem = net.apply(psi_p, pose_feat(state[None])[0], fg)
            e = carrot(rem, jnp); d = jnp.sqrt(jnp.sum(rem * rem, -1) + 1e-12)
            hraw = C.wrap_pi(goal[2] - state[2], jnp) / PI
            ph = 2.0 * PI * t / Tc
            obs = obs_vec(state, vprev, ph, e, d, hraw, cfg); ctx = ctx_vec(state, vprev, e, d, hraw, cfg)
            mean, _, coef = act.apply(act_p, obs, ctx)
            y = prior_term(coef, t.astype(jnp.float32), Tc, cfg) + jnp.tanh(mean)
            return (E.env_step(state, y, p), jnp.clip(y[0], -1, 1)), state
        (last, _), states = jax.lax.scan(stepf, (jnp.zeros(4), jnp.array(0.0)), jnp.arange(nav_T))
        return jnp.concatenate([states, last[None]], 0)
    f = jax.jit(jax.vmap(single))
    return lambda goals: np.asarray(f(jnp.asarray(goals)))


def evaluate(ckpt, nav_T=300):
    pl = pickle.load(open(ckpt, "rb"))
    raw_cfg = dict(pl["cfg"])
    if "no_prior" in raw_cfg and "with_prior" not in raw_cfg:
        raw_cfg["with_prior"] = not raw_cfg["no_prior"]   # back-compat: old ckpts used no_prior
    cfg = HoloSDConfig(**{k: v for k, v in raw_cfg.items() if k in HoloSDConfig.__dataclass_fields__})
    rho = cfg.rho; succ = 0.2 * rho; deploy = deploy_fn(cfg, pl, nav_T)
    rng = np.random.default_rng(0); summ = {}
    suites = [("nav_random", rand_goals(rng, 48, 0.7 * rho)), ("nav_ring", ring_goals(12, 0.7 * rho)),
              ("nav_ring_hard", ring_hard(12, 0.7 * rho))]
    print(f"[eval] {cfg.exp_name}  (deploy==train)", flush=True)
    for tag, goals in suites:
        st = deploy(goals); m = nav_metrics(st, goals, succ); m["pose_final"] = pose_final(st, goals)
        summ[tag] = m
        print(f"  {tag:14s}: reach {m['reach']*100:3.0f}%  pose {m['pose']*100:3.0f}%  pose_final {m['pose_final']*100:3.0f}%", flush=True)
    # reach vs radius
    fr = np.linspace(0.4, 4.0, 10); rr = []
    for f in fr:
        g = rand_goals(np.random.default_rng(7), 32, f * rho)
        g[:, :2] *= (f * rho) / (np.linalg.norm(g[:, :2], axis=1, keepdims=True) + 1e-9)
        rr.append(nav_metrics(deploy(g.astype(np.float32)), g, succ)["reach"])
    summ["reach_vs_radius"] = dict(r=[float(x) for x in fr], reach=[float(x) for x in rr])
    summ["kind"] = "holosd_clean"
    avg = {k: float(np.mean([summ[s][k] for s in ("nav_random", "nav_ring", "nav_ring_hard")]))
           for k in ("reach", "pose", "pose_final")}
    summ["avg"] = avg
    print(f"  >>> AVG reach {avg['reach']*100:.1f}%  pose {avg['pose']*100:.1f}%  pose_final {avg['pose_final']*100:.1f}%", flush=True)
    out = os.path.join(os.path.dirname(ckpt), "eval"); os.makedirs(out, exist_ok=True)
    json.dump(summ, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"  -> {out}/metrics.json", flush=True)
    return summ


# ===================================================================== CLI
def main():
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--out_dir", default="exp/bicycle/HoloSD_clean")
    tr.add_argument("--exp_name", default="HoloSD_clean")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--phaseA", type=int, default=1800); tr.add_argument("--phaseB", type=int, default=3000)
    tr.add_argument("--quick", action="store_true")
    ev = sub.add_parser("eval")
    ev.add_argument("ckpt"); ev.add_argument("--T", type=int, default=300)
    ar = a.parse_args()
    if ar.mode == "train":
        cfg = HoloSDConfig(exp_name=ar.exp_name, out_dir=ar.out_dir, seed=ar.seed,
                           phaseA_epochs=ar.phaseA, phaseB_epochs=ar.phaseB, quick=ar.quick)
        train(cfg)
    else:
        evaluate(ar.ckpt, ar.T)


if __name__ == "__main__":
    main()
