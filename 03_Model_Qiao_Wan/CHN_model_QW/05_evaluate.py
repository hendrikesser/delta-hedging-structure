import importlib as _il
import os
import numpy as np
import pandas as pd
import torch

config              = _il.import_module('01_config')
_fnn                = _il.import_module('03_fnn_model')
ResidualHedgingNN   = _fnn.ResidualHedgingNN
init_weights_xavier = _fnn.init_weights_xavier
_trainer            = _il.import_module('04_trainer')
train_model         = _trainer.train_model


def _get_buckets(flag):
    if flag == config.Call:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]


def _assign_buckets(delta_bs, flag):
    """Deterministic bucket assignment: partition of [0.05, 0.95] incl. both
    boundaries (avoids float-noise gaps/overlaps of interval masks)."""
    sign = 1 if flag == config.Call else -1
    idx = np.clip(np.floor((np.abs(delta_bs) - 0.05) / 0.1 + 1e-9).astype(int), 0, 8)
    return sign * np.round(0.1 * (idx + 1), 1)


def _dump_daily_sse(delta_total, dV, dS, dbs, dates, run_id):
    """Save per-day SSEs of the residual NN and BS hedges (for 09_dm_test)."""
    d = pd.DataFrame({'date': pd.to_datetime(dates),
                      'se_nn': (dV - delta_total * dS) ** 2,
                      'se_bs': (dV - dbs * dS) ** 2})
    g = d.groupby('date').agg(sse_nn=('se_nn', 'sum'), sse_bs=('se_bs', 'sum'),
                              n=('se_nn', 'size'))
    os.makedirs('diagnostics_output/per_run_dumps', exist_ok=True)
    tag = str(getattr(config, 'TEST_CUTOFF', 'NA'))[:7]
    path = f'diagnostics_output/per_run_dumps/daily_sse_{config.MODEL_NAME}_{config.FLAG}_{tag}_run{run_id}.csv'
    g.to_csv(path)


def compute_gain_ratio(model, X_test_t, dbs_test_np, dV_test_np, dS_test_np, dates_test=None):
    """Compute overall Gain Ratio = 1 - MSE(NN) / MSE(BS) on the test set."""
    model.eval()
    with torch.no_grad():
        correction = model(X_test_t).numpy().squeeze()

    delta_total = dbs_test_np + correction

    mse_nn     = np.mean((dV_test_np - delta_total  * dS_test_np) ** 2)
    mse_bs     = np.mean((dV_test_np - dbs_test_np  * dS_test_np) ** 2)
    gain_ratio = 1 - mse_nn / mse_bs

    print(f"MSE (Residual NN):    {mse_nn:.6f}")
    print(f"MSE (Black-Scholes):  {mse_bs:.6f}")
    print("-" * 30)
    print(f"Overall Gain Ratio (Test Set): {gain_ratio:.4f}")
    if dates_test is not None:
        _dump_daily_sse(delta_total, dV_test_np, dS_test_np, dbs_test_np, dates_test, 1)
    return gain_ratio, delta_total


def compute_bucket_gains(delta_total, dV_test_np, dS_test_np, dbs_test_np, flag):
    """Compute Gain Ratio for each delta bucket and print a formatted table."""
    buckets = _get_buckets(flag)

    print(f"\n{'Delta Bucket':>15} {'Gain Ratio':>12} {'N':>8}")
    print("-" * 38)

    bucket_gains_run = {}
    sum_mse_nn = 0.0
    sum_mse_bs = 0.0

    assigned = _assign_buckets(dbs_test_np, flag)
    for b in buckets:
        mask = assigned == b
        if mask.sum() == 0:
            continue

        dV_b       = dV_test_np[mask]
        dS_b       = dS_test_np[mask]
        dt_b       = delta_total[mask]
        dbs_b      = dbs_test_np[mask]

        mse_nn_b = np.mean((dV_b - dt_b  * dS_b) ** 2)
        mse_bs_b = np.mean((dV_b - dbs_b * dS_b) ** 2)
        gain_b   = 1 - mse_nn_b / mse_bs_b

        sum_mse_nn += np.sum((dV_b - dt_b  * dS_b) ** 2)
        sum_mse_bs += np.sum((dV_b - dbs_b * dS_b) ** 2)

        bucket_gains_run[b] = gain_b
        print(f"{b:>15.1f} {gain_b:>12.4f} {mask.sum():>8}")

    overall = 1 - sum_mse_nn / sum_mse_bs
    print("-" * 38)
    print(f"{'overall':>15} {overall:>12.4f}")
    return buckets, bucket_gains_run


def run_multi_evaluation(n_features,
                         X_train_t, dV_train_t, dS_train_t, dbs_train_t,
                         X_val_t,   dV_val_t,   dS_val_t,   dbs_val_t,
                         X_test_t,  dV_test_np, dS_test_np, dbs_test_np,
                         first_gain, first_bucket_gains, buckets, flag,
                         dates_test=None):
    """Re-train from scratch NUM_RUNS times to assess result robustness.

    Run 1 results are passed in from the caller (already computed).
    Runs 2..NUM_RUNS use fresh random initialisations.

    Returns:
        overall_gains  - list of NUM_RUNS overall gain ratios
        bucket_gains   - dict {bucket: [gain_run1, ..., gain_runN]}
    """
    overall_gains = [first_gain]
    bucket_gains  = {b: [first_bucket_gains[b]] for b in buckets if b in first_bucket_gains}

    print(f"=== RUN 1/{config.NUM_RUNS} (from prior results) ===")
    print(f"Run 1 Completed. Overall Gain Ratio: {first_gain:.4f}\n")
    print(f"Starting remaining {config.NUM_RUNS - 1} independent training runs...\n")

    for run in range(2, config.NUM_RUNS + 1):
        print(f"=== RUN {run}/{config.NUM_RUNS} ===")

        run_model = ResidualHedgingNN(n_features)
        run_model.apply(init_weights_xavier)
        run_model = train_model(run_model,
                                X_train_t, dV_train_t, dS_train_t, dbs_train_t,
                                X_val_t,   dV_val_t,   dS_val_t,   dbs_val_t)

        run_model.eval()
        with torch.no_grad():
            correction_run = run_model(X_test_t).numpy().squeeze()
        delta_total_run = dbs_test_np + correction_run

        run_overall = 1 - (
            np.mean((dV_test_np - delta_total_run * dS_test_np) ** 2) /
            np.mean((dV_test_np - dbs_test_np     * dS_test_np) ** 2)
        )
        overall_gains.append(run_overall)
        if dates_test is not None:
            _dump_daily_sse(delta_total_run, dV_test_np, dS_test_np, dbs_test_np,
                            dates_test, run)

        assigned = _assign_buckets(dbs_test_np, flag)
        for b in buckets:
            mask = assigned == b
            if mask.sum() == 0:
                continue
            dV_b   = dV_test_np[mask]
            dS_b   = dS_test_np[mask]
            dt_b   = delta_total_run[mask]
            dbs_b  = dbs_test_np[mask]
            mse_nn_b = np.mean((dV_b - dt_b  * dS_b) ** 2)
            mse_bs_b = np.mean((dV_b - dbs_b * dS_b) ** 2)
            bucket_gains.setdefault(b, []).append(1 - mse_nn_b / mse_bs_b)

        print(f"Run {run} Completed. Overall Gain Ratio: {run_overall:.4f}\n")

    return overall_gains, bucket_gains


def report_frequency(model, X_test_t, df_test):
    """Rebalancing-frequency gains under the corrected mark-to-market construction.

    The hedge ratio is fitted at the daily horizon, the position is rebalanced
    every k trading days and held STALE between rebalances, and the pooled DAILY
    hedging errors are scored. For k = 1 this equals the standard daily gain.
    This supersedes the old HEDGE_FREQ='weekly'/'monthly' k-day-hold path (which
    is gamma/theta dominated and collapses). Single run; 13_freq_stale_delta.py
    gives the multi-run averages used in the paper."""
    import numpy as _np, pandas as _pd, torch as _t
    model.eval()
    with _t.no_grad():
        out = model(X_test_t).numpy().squeeze()
    dbs = df_test['delta'].values
    dm = dbs + out                                  # delta implied by the model
    d = _pd.DataFrame({'oid': df_test['optionid'].values,
                       'dt': _pd.to_datetime(df_test['date'].values),
                       'dV': df_test['delta_V'].values, 'dS': df_test['delta_S'].values,
                       'dbs': dbs, 'dm': dm}).sort_values(['oid', 'dt'])
    print("\nRebalancing frequency (stale-delta, daily mark-to-market, single run):")
    for _f, _k in [('daily', 1), ('weekly', 5), ('monthly', 21)]:
        _reb = _pd.Series(d.groupby('oid', sort=False).cumcount().values % _k == 0, index=d.index)
        _sm = d['dm'].where(_reb).groupby(d['oid']).ffill().values
        _sb = d['dbs'].where(_reb).groupby(d['oid']).ffill().values
        _g = 1 - _np.mean((d['dV'].values - _sm * d['dS'].values) ** 2) / \
                 _np.mean((d['dV'].values - _sb * d['dS'].values) ** 2)
        print(f"  {_f:>8} (k={_k:>2}): {_g:+.4f}")
