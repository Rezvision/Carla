#!/usr/bin/env python3
"""
dlg_attack.py — Deep Leakage from Gradients (DLG), controlled MVP
=================================================================

Target: the GRU sequence-to-sequence autoencoder defined in fed_client_jax.py
(WINDOW_SIZE=20, N_FEATURES=8, GRU_HIDDEN=32). This script reconstructs the
*training input* of one federated client step from the gradient that step
emits. It establishes an upper-bound baseline of input leakage in your
federated IDS pipeline before any privacy-preserving mechanism (DP-SGD,
output perturbation, secure aggregation, gradient compression, ...) is added.

THREAT MODEL — honest-but-curious server / passive eavesdropper
    * Knows the global model weights θ (always true in any FedAvg setup).
    * Observes one client's per-batch gradient g = ∇θ L(x_batch; θ).
    * Wants to recover x_batch.

WORST-CASE-FOR-PRIVACY ASSUMPTIONS (deliberately so — this gives the
strongest attack regime; mitigations are added back as ablations later):
    * Batch size = 1.  Larger batches → harder (attacker must also
      untangle a permutation across the batch).
    * Single SGD step.  Real FedAvg sends weight deltas after K>1 local
      steps; K>1 is informationally lossy for DLG.
    * NOISE_STD = 0 (denoising-AE noise disabled).
    * L2_LAMBDA = 0 (regulariser disabled — typically a small effect).
    * Attacker has full knowledge of the architecture and weights.

ATTACK — gradient inversion (Geiping et al. 2020, "Inverting Gradients").
Optimise a dummy input x* by minimising
    L_attack(x*) = 1 − cos(∇θ L(x*; θ), g_observed)  +  λ_TV · TV_t(x*)
via Adam with K random restarts. Cos-similarity (rather than L2) handles
gradient-magnitude variation across layers; the temporal-TV prior reflects
the smoothness of CAN telemetry signals.

EVALUATION — N independent trials. Each trial draws fresh weights and a
fresh smooth-signal ground-truth window. Per-element error is bootstrapped
to give 95% CIs on:
    * MAE between recovered and true window
    * Pearson r between recovered and true (vectorised across all elements)
    * "Recovered fraction" at multiple tolerances ε
Per-feature recovery breakdown is also reported.

OUTPUTS  (all in --outdir)
    results.json              — every metric + bootstrap CIs + per-trial
    recovered_vs_true.png     — 8-channel overlay (best trial)
    recovery_vs_tolerance.png — recovery rate vs ε across all trials
    loss_curve.png            — Adam convergence (best trial)
    all_pairs.npz             — full (x_true, x_rec) for downstream analysis

NEXT EXPERIMENTS once this baseline is recorded
    * Re-enable NOISE_STD and L2 — measure leakage degradation.
    * Sweep batch size B ∈ {1, 4, 16, 32} — plot leakage vs B.
    * Sweep local-step count K ∈ {1, 5, 20} — plot leakage vs K.
    * Add DP-SGD (clip + Gaussian noise on g) — plot (ε, δ) vs leakage.
    * Replace synthetic data with real normalised CAN windows from CARLA.

USAGE
    # Stage 1 — sanity check on synthetic (5 min):
    python dlg_attack.py --outdir dlg_synth

    # Stage 2 — real CAN data, fresh-init weights (the paper baseline):
    python dlg_attack.py --parquet path/to/can_log.parquet --outdir dlg_real

    # Tighter CIs / longer optimisation:
    python dlg_attack.py --trials 50 --iters 6000 --restarts 8

The --parquet path expects columns
    speed_kmh, battery_level, throttle, brake, steering, gear,
    location_x, location_y
(matches DataCollector.FEATURE_NAMES in fed_client_jax.py). Z-score
normalisation is computed on the file (matches DataCollector._normalise);
the chosen mean/std are saved to results.json so metrics in normalised space
can be inverted to physical units later if needed.
"""
import argparse, json, os, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad

# ── Architectural constants (must match fed_client_jax.py) ──────────────────
WINDOW_SIZE = 20
N_FEATURES  = 8
GRU_HIDDEN  = 32

# Feature order — must match DataCollector.FEATURE_NAMES in fed_client_jax.py
FEATURE_NAMES = ['speed_kmh', 'battery_level', 'throttle', 'brake',
                 'steering',  'gear',          'location_x', 'location_y']

# ── Forward pass — copied verbatim from fed_client_jax.py for fidelity ──────
def _gru_step_enc(p, x, h):
    xh = jnp.concatenate([x, h], axis=1)
    r  = jax.nn.sigmoid(xh @ p['Wr']  + p['br'])
    z  = jax.nn.sigmoid(xh @ p['Wz']  + p['bz'])
    n  = jnp.tanh(x @ p['Wn_x'] + (r * h) @ p['Wn_h'] + p['bn'])
    return (1.0 - z) * h + z * n

def _gru_step_dec(p, prev, h):
    xh = jnp.concatenate([prev, h], axis=1)
    r  = jax.nn.sigmoid(xh @ p['Dec_Wr'] + p['Dec_br'])
    z  = jax.nn.sigmoid(xh @ p['Dec_Wz'] + p['Dec_bz'])
    n  = jnp.tanh(prev @ p['Dec_Wn_x'] + (r * h) @ p['Dec_Wn_h'] + p['Dec_bn'])
    h_new = (1.0 - z) * h + z * n
    return h_new, h_new @ p['Wo'] + p['bo']

def encoder_forward(p, x_seq):
    h0 = jnp.zeros((x_seq.shape[0], GRU_HIDDEN))
    h, _ = jax.lax.scan(lambda h, x_t: (_gru_step_enc(p, x_t, h), None),
                        h0, x_seq.transpose(1, 0, 2))
    return h

def decoder_forward_train(p, h_enc, x_seq):
    B = h_enc.shape[0]
    zeros = jnp.zeros((B, N_FEATURES))
    tf_in = jnp.concatenate([zeros[None], x_seq.transpose(1, 0, 2)[:-1]], axis=0)
    _, recon_T = jax.lax.scan(lambda h, x_p: _gru_step_dec(p, x_p, h),
                              h_enc, tf_in)
    return recon_T.transpose(1, 0, 2)

def loss_fn(p, x_seq):
    """Pure MSE — no noise, no L2 (worst case for privacy, as documented)."""
    h = encoder_forward(p, x_seq)
    return jnp.mean((decoder_forward_train(p, h, x_seq) - x_seq) ** 2)

def init_params(seed):
    F, H, XH = N_FEATURES, GRU_HIDDEN, N_FEATURES + GRU_HIDDEN
    keys = jax.random.split(jax.random.PRNGKey(seed), 12)
    w = lambda k, r, c: jax.random.normal(k, (r, c)) * jnp.sqrt(2.0 / (r + c))
    b = lambda n: jnp.zeros(n)
    return dict(
        Wr=w(keys[0], XH, H),     br=b(H),
        Wz=w(keys[1], XH, H),     bz=b(H),
        Wn_x=w(keys[2], F, H),    Wn_h=w(keys[3], H, H),    bn=b(H),
        Dec_Wr=w(keys[4], XH, H), Dec_br=b(H),
        Dec_Wz=w(keys[5], XH, H), Dec_bz=b(H),
        Dec_Wn_x=w(keys[6], F, H),Dec_Wn_h=w(keys[7], H, H),Dec_bn=b(H),
        Wo=w(keys[8], H, F),      bo=b(F),
    )

# ── Synthetic ground-truth windows ─────────────────────────────────────────
def synth_window(seed):
    """Smooth multi-channel signal in roughly [0, 1] — mimics normalised CAN
    telemetry (RPM, speed, throttle, brake, steering, ...). Each channel is
    a random sinusoid + slow drift: distinguishable across channels and
    smooth within each, matching the structure of real automotive signals."""
    rng = np.random.default_rng(seed)
    t   = np.arange(WINDOW_SIZE, dtype=np.float32)[:, None]
    freqs = rng.uniform(0.05, 0.5,    (1, N_FEATURES))
    phase = rng.uniform(0.0, 2*np.pi, (1, N_FEATURES))
    amp   = rng.uniform(0.20, 0.45,   (1, N_FEATURES))
    bias  = rng.uniform(0.30, 0.70,   (1, N_FEATURES))
    drift = rng.uniform(-0.005, 0.005,(1, N_FEATURES)) * t
    sig   = bias + amp * np.sin(2 * np.pi * freqs * t + phase) + drift
    return sig.astype(np.float32)[None, ...]    # (1, T, F)

def parquet_source(path, n_trials, seed):
    """Build a callable trial_idx → (1, T, F) from a parquet CAN log.
    Z-score normalisation matches DataCollector._normalise in fed_client.
    Returns (callable, meta) where meta records norm stats + chosen windows."""
    try:
        import pandas as pd      # noqa: F401
    except ImportError:
        raise SystemExit("Reading parquet needs pandas + pyarrow:  "
                         "pip install pandas pyarrow")
    import pandas as pd
    df = pd.read_parquet(path)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise SystemExit(f"Parquet missing required columns: {missing}\n"
                         f"Expected: {FEATURE_NAMES}")
    raw  = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    mean = raw.mean(axis=0)
    std  = np.where(raw.std(axis=0) > 1e-6, raw.std(axis=0), 1.0)
    data = ((raw - mean) / std).astype(np.float32)

    n_full = len(data) - WINDOW_SIZE + 1
    if n_full < n_trials:
        raise SystemExit(f"Parquet has only {n_full} possible windows; "
                         f"need {n_trials}")
    rng = np.random.default_rng(seed)
    # Prefer non-overlapping starts (matches the stride=20 used in training)
    nonoverlap = list(range(0, n_full, WINDOW_SIZE))
    if len(nonoverlap) >= n_trials:
        starts = list(rng.choice(nonoverlap, size=n_trials, replace=False))
    else:
        starts = list(rng.choice(n_full, size=n_trials, replace=False))
    starts = [int(s) for s in starts]
    windows = [data[s:s + WINDOW_SIZE][None, ...].astype(np.float32)
               for s in starts]
    print(f"[DLG] Parquet '{path}': {len(data)} frames; sampling "
          f"{n_trials} windows (non-overlap stride=20).")
    print(f"[DLG] z-score:  mean={np.array2string(mean, precision=3)}")
    print(f"[DLG] z-score:  std ={np.array2string(std,  precision=3)}")
    meta = {"source": "parquet", "path": str(path),
            "feature_names": FEATURE_NAMES,
            "norm_mean": mean.tolist(), "norm_std": std.tolist(),
            "picked_starts": starts}
    return (lambda i: windows[i]), meta


def cos_dist(g_a, g_b):
    """1 − cosine-similarity over flattened gradient pytrees."""
    num = sum(jnp.sum(g_a[k] * g_b[k]) for k in g_b)
    da  = sum(jnp.sum(g_a[k] ** 2)    for k in g_b)
    db  = sum(jnp.sum(g_b[k] ** 2)    for k in g_b)
    return 1.0 - num / (jnp.sqrt(da * db) + 1e-12)

def make_attack_runner(iters):
    """Build a JIT-compiled inner loop. Compiled once, reused per trial.
    Runs `iters` Adam steps minimising cos-distance + TV prior on time axis."""
    grad_loss = grad(loss_fn, argnums=0)

    def attack_obj(x, params, true_grad, tv_weight):
        g  = grad_loss(params, x)
        cd = cos_dist(g, true_grad)
        tv = jnp.mean(jnp.abs(x[:, 1:, :] - x[:, :-1, :]))
        return cd + tv_weight * tv

    obj_and_g = value_and_grad(attack_obj, argnums=0)

    @jit
    def run(x0, params, true_grad, tv_weight, lr):
        def body(i, state):
            x, m, v, t, curve = state
            loss, g = obj_and_g(x, params, true_grad, tv_weight)
            t  = t + 1
            m  = 0.9   * m + 0.1   * g
            v  = 0.999 * v + 0.001 * g**2
            mh = m / (1 - 0.9**t)
            vh = v / (1 - 0.999**t)
            x  = x - lr * mh / (jnp.sqrt(vh) + 1e-8)
            curve = curve.at[i].set(loss)
            return (x, m, v, t, curve)
        init = (x0,
                jnp.zeros_like(x0),
                jnp.zeros_like(x0),
                0,
                jnp.zeros(iters, dtype=jnp.float32))
        x_final, _, _, _, curve = jax.lax.fori_loop(0, iters, body, init)
        return x_final, curve
    return run

def attack_one_window(params, true_grad, *, runner, lr, tv_weight,
                      restarts, seed):
    """K random restarts, return the recovery with lowest final attack loss."""
    best = (None, None, np.inf)
    for r in range(restarts):
        key = jax.random.PRNGKey(seed * 1000 + r)
        x0  = jax.random.uniform(key, (1, WINDOW_SIZE, N_FEATURES))
        x_rec, curve = runner(x0, params, true_grad, tv_weight, lr)
        x_rec, curve = np.array(x_rec), np.array(curve)
        if curve[-1] < best[2]:
            best = (x_rec, curve, float(curve[-1]))
    return best

# ── Metrics & bootstrap ─────────────────────────────────────────────────────
def per_element_metrics(x_true, x_rec, tols):
    err = np.abs(x_rec - x_true).ravel()
    mae = float(err.mean())
    a, b = x_rec.ravel(), x_true.ravel()
    if a.std() > 1e-9 and b.std() > 1e-9:
        r = float(np.corrcoef(a, b)[0, 1])
    else:
        r = 0.0
    return {"mae": mae, "pearson_r": r,
            **{f"tol_{t}": float((err < t).mean()) for t in tols}}

def bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    rng   = np.random.default_rng(seed)
    arr   = np.asarray(values, dtype=np.float64)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (float(arr.mean()),
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))

# ── Plots ───────────────────────────────────────────────────────────────────
def save_plots(outdir, rep, all_errs_flat, tols, feature_names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fnames = feature_names or [f"feat {i}" for i in range(N_FEATURES)]

    plt.figure(figsize=(6, 3))
    plt.semilogy(rep["curve"])
    plt.xlabel("Adam iteration"); plt.ylabel("attack loss (1 − cos + λ·TV)")
    plt.title("DLG attack convergence (best trial)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/loss_curve.png", dpi=110); plt.close()

    fig, axes = plt.subplots(N_FEATURES, 1,
                             figsize=(7, 1.3 * N_FEATURES), sharex=True)
    for f in range(N_FEATURES):
        axes[f].plot(rep["x_true"][0, :, f], lw=2,   label="true")
        axes[f].plot(rep["x_rec"][0, :, f],  lw=1.4, ls="--", label="recovered")
        axes[f].set_ylabel(fnames[f], fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Gradient leakage: recovered vs true (best trial, "
                 "z-scored units)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/recovered_vs_true.png", dpi=110); plt.close()

    grid  = np.linspace(0, 0.2, 41)
    rates = [(all_errs_flat < t).mean() for t in grid]
    plt.figure(figsize=(6, 3))
    plt.plot(grid, rates, lw=2)
    for t in tols:
        r = (all_errs_flat < t).mean()
        plt.axvline(t, color='gray', ls=':', alpha=0.5)
        plt.text(t, r, f"  {r:.1%} @ ε={t}", fontsize=8, va="bottom")
    plt.xlabel("tolerance ε"); plt.ylabel("fraction of points recovered")
    plt.title("Recovery rate vs tolerance — pooled across all trials")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{outdir}/recovery_vs_tolerance.png", dpi=110); plt.close()

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",    type=int,   default=20)
    ap.add_argument("--iters",     type=int,   default=4000)
    ap.add_argument("--lr",        type=float, default=0.05)
    ap.add_argument("--restarts",  type=int,   default=4)
    ap.add_argument("--tv-weight", type=float, default=1e-3)
    ap.add_argument("--seed",      type=int,   default=0)
    ap.add_argument("--outdir",    default="./dlg_out")
    ap.add_argument("--parquet",   default=None,
                    help="path to a CAN-telemetry .parquet (columns: "
                         "speed_kmh, battery_level, throttle, brake, "
                         "steering, gear, location_x, location_y). "
                         "If omitted, uses synthetic smooth signals.")
    ap.add_argument("--tols", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[DLG] {args.trials} trials × {args.restarts} restarts × "
          f"{args.iters} iters")
    print(f"[DLG] tols={args.tols}  tv_weight={args.tv_weight}  lr={args.lr}")

    # Build the data source: real CAN parquet if --parquet, else synthetic
    if args.parquet:
        data_fn, data_meta = parquet_source(args.parquet, args.trials,
                                            args.seed)
    else:
        data_meta = {"source": "synthetic",
                     "feature_names": [f"feat_{i}" for i in range(N_FEATURES)]}
        data_fn   = lambda i: synth_window(seed=args.seed * 31337 + i)

    runner        = make_attack_runner(args.iters)
    grad_loss_jit = jit(grad(loss_fn, argnums=0))

    # Warm up JIT so wall-clock excludes compile time on the first real trial
    print("[DLG] JIT warmup...", end=" ", flush=True)
    p_warm = init_params(seed=99999)
    x_warm = jnp.array(synth_window(seed=99999))
    g_warm = grad_loss_jit(p_warm, x_warm)
    _      = runner(jax.random.uniform(jax.random.PRNGKey(0),
                                       (1, WINDOW_SIZE, N_FEATURES)),
                    p_warm, g_warm, args.tv_weight, args.lr)
    print("done.")

    trials = []
    t0     = time.time()
    for i in range(args.trials):
        params = init_params(seed=args.seed * 7919 + i)
        x_true = data_fn(i)
        true_g = grad_loss_jit(params, jnp.array(x_true))
        x_rec, curve, fin = attack_one_window(
            params, true_g, runner=runner, lr=args.lr,
            tv_weight=args.tv_weight, restarts=args.restarts,
            seed=args.seed * 977 + i)
        m = per_element_metrics(x_true, x_rec, args.tols)
        trials.append(dict(x_true=x_true, x_rec=x_rec, curve=curve,
                           final_attack_loss=fin, **m))
        print(f"  trial {i+1:3d}/{args.trials}: "
              f"MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}  "
              f"rec@0.05={m['tol_0.05']:.1%}  attack_loss={fin:.2e}")

    # ── Aggregate metrics with bootstrap 95% CIs ────────────────────────────
    keys    = ["mae", "pearson_r"] + [f"tol_{t}" for t in args.tols]
    summary = {}
    for k in keys:
        mean, lo, hi = bootstrap_ci([t[k] for t in trials], seed=args.seed)
        summary[k] = dict(mean=mean, ci95_lo=lo, ci95_hi=hi,
                          halfwidth=(hi - lo) / 2)

    # Per-feature recovery (at the tightest tolerance)
    eps      = args.tols[0]
    per_feat = {}
    for f in range(N_FEATURES):
        rates = [float((np.abs(t["x_rec"][:, :, f] - t["x_true"][:, :, f])
                        < eps).mean()) for t in trials]
        mean, lo, hi = bootstrap_ci(rates, seed=args.seed + f)
        per_feat[f"feat_{f}"] = dict(eps=eps, mean=mean,
                                     ci95_lo=lo, ci95_hi=hi)

    # ── Headline report ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"DLG ATTACK RESULTS — N = {args.trials} trials, batch=1, "
          f"no defence")
    print("=" * 64)
    print(f"  MAE:           {summary['mae']['mean']:.4f}  "
          f"(95% CI [{summary['mae']['ci95_lo']:.4f}, "
          f"{summary['mae']['ci95_hi']:.4f}])")
    print(f"  Pearson r:     {summary['pearson_r']['mean']:.4f}  "
          f"(95% CI [{summary['pearson_r']['ci95_lo']:.4f}, "
          f"{summary['pearson_r']['ci95_hi']:.4f}])")
    print(f"  'Deduced'-rate (fraction of (sample,t,f) recovered within ε):")
    for t in args.tols:
        s = summary[f"tol_{t}"]
        print(f"     ε = {t:<5}  →  {s['mean']*100:5.1f}%  "
              f"(95% CI ±{s['halfwidth']*100:.1f} pp)")
    print(f"\n  Per-feature recovered@ε={eps}:")
    fnames = data_meta.get("feature_names",
                           [f"feat_{i}" for i in range(N_FEATURES)])
    for f in range(N_FEATURES):
        pf = per_feat[f"feat_{f}"]
        print(f"     {fnames[f]:<14}: {pf['mean']*100:5.1f}%  "
              f"(95% CI [{pf['ci95_lo']*100:5.1f}%, "
              f"{pf['ci95_hi']*100:5.1f}%])")
    print(f"\n  Wall time:     {time.time()-t0:.1f}s")
    print("=" * 64)

    # ── Save ────────────────────────────────────────────────────────────────
    json_out = dict(
        args=vars(args), data=data_meta,
        summary=summary, per_feature=per_feat,
        per_trial=[{k: v for k, v in t.items()
                    if k not in ("x_true", "x_rec", "curve")}
                   for t in trials],
    )
    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    rep = min(trials, key=lambda t: t["mae"])
    all_errs_flat = np.concatenate(
        [np.abs(t["x_rec"] - t["x_true"]).ravel() for t in trials])
    save_plots(args.outdir, rep, all_errs_flat, args.tols,
               feature_names=data_meta.get("feature_names"))
    np.savez(f"{args.outdir}/all_pairs.npz",
             x_true=np.concatenate([t["x_true"] for t in trials], axis=0),
             x_rec =np.concatenate([t["x_rec"]  for t in trials], axis=0))

    print(f"\nWrote → {args.outdir}/")
    print(f"   results.json              all metrics + bootstrap CIs")
    print(f"   recovered_vs_true.png     8-channel overlay (best trial)")
    print(f"   recovery_vs_tolerance.png recovery rate vs ε")
    print(f"   loss_curve.png            attack convergence")
    print(f"   all_pairs.npz             every (x_true, x_rec) for "
          f"further analysis")

if __name__ == "__main__":
    main()
