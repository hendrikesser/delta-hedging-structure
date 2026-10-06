import importlib as _il
import os
import numpy as np
import pandas as pd

config = _il.import_module('01_config')

OUT = 'diagnostics_output'
MIN_BUCKET_MONTH_OBS = 10   # bucket-month cells with fewer obs are skipped in the monthly average


def _get_buckets(flag):
    if flag == config.Call:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]


def _monthly_gains(err_mv, err_bs, months):
    """Per-test-month Gain = 1 - SSE_MV/SSE_BS (HW 2017 Table 1 metric)."""
    d = pd.DataFrame({'mv': err_mv ** 2, 'bs': err_bs ** 2, 'month': months})
    g = d.groupby('month').agg(mv=('mv', 'sum'), bs=('bs', 'sum'), n=('mv', 'size'))
    g['gain'] = 1 - g['mv'] / g['bs']
    return g[['gain', 'n']]


def evaluate(df_eval, flag):
    """Compute the minimum-variance delta and its gain ratio vs the BS delta.

    delta_MV = delta_BS + vega / (S * sqrt(T)) * (a + b*delta + c*delta^2)

    Two Gain definitions are reported:
      pooled:  1 - SSE(MV)/SSE(BS) over the whole evaluation sample
      monthly: per-test-month gains, then averaged across months
               (Hull & White 2017, Table 1). The monthly series is saved to
               diagnostics_output/ -- it shows WHEN the model works/breaks.
    """
    sqrt_T = np.sqrt(df_eval['TTM'].replace(0, 1e-6))
    df_eval = df_eval.copy()
    df_eval['delta_MV'] = df_eval['delta'] + (df_eval['vega'] / (df_eval['S'] * sqrt_T)) * (
        df_eval['a'] + df_eval['b'] * df_eval['delta'] + df_eval['c'] * df_eval['delta'] ** 2)

    err_mv = df_eval['delta_V'] - df_eval['delta_MV'] * df_eval['delta_S']
    err_bs = df_eval['delta_V'] - df_eval['delta']    * df_eval['delta_S']

    mse_mv, mse_bs = np.mean(err_mv ** 2), np.mean(err_bs ** 2)
    gain_ratio = 1 - mse_mv / mse_bs

    print(f"MSE (Minimum Variance): {mse_mv:.6f}")
    print(f"MSE (Black-Scholes):    {mse_bs:.6f}")
    print("-" * 52)
    print(f"Overall Gain Ratio (pooled): {gain_ratio:.4f}")

    # Per-test-month gains (HW Table 1 metric + break-timing series)
    monthly = None
    if 'month' in df_eval.columns:
        monthly = _monthly_gains(err_mv, err_bs, df_eval['month'])
        print(f"Overall Gain, monthly avg (HW Table 1): {monthly['gain'].mean():.4f}  "
              f"[{len(monthly)} months, {(monthly['gain'] < 0).sum()} negative]")
        os.makedirs(os.path.join(OUT, 'per_run_dumps'), exist_ok=True); os.makedirs(os.path.join(OUT, 'summaries'), exist_ok=True)
        path = os.path.join(OUT, 'summaries', f'monthly_gains_{flag}_{config.HEDGE_FREQ}_W{config.WINDOW_MONTHS}.csv')
        monthly.to_csv(path)
        print(f"monthly series saved: {path}")

    # Per-observation errors (date-keyed) for significance tests (09_dm_test)
    os.makedirs(os.path.join(OUT, 'per_run_dumps'), exist_ok=True); os.makedirs(os.path.join(OUT, 'summaries'), exist_ok=True)
    dump = pd.DataFrame({'date': df_eval['date'].values,
                         'err_mv': err_mv.values, 'err_bs': err_bs.values})
    wtag = f"{getattr(config, 'EVAL_START', None) or 'full'}_{getattr(config, 'EVAL_END', None) or 'end'}"
    dpath = os.path.join(OUT, 'per_run_dumps', f'errors_{flag}_{config.HEDGE_FREQ}_W{config.WINDOW_MONTHS}_{wtag}.csv.gz')
    dump.to_csv(dpath, index=False, compression='gzip')
    print(f"error dump saved: {dpath}")

    # Per-delta-bucket breakdown: pooled gain and monthly-averaged gain
    buckets = _get_buckets(flag)
    print(f"\n{'Delta Bucket':>13} {'Pooled':>10} {'MonthlyAvg':>12} {'N':>8}")
    print("-" * 52)

    # Deterministic bucket assignment (partition of [0.05, 0.95] incl. both
    # boundaries; avoids float-noise gaps/overlaps of interval masks).
    sign = 1 if flag == config.Call else -1
    idx = np.clip(np.floor((df_eval['delta'].abs() - 0.05) / 0.1 + 1e-9).astype(int), 0, 8)
    assigned = sign * np.round(0.1 * (idx + 1), 1)

    bucket_gains = {}
    sum_mv = sum_bs = 0.0
    for b in buckets:
        mask = assigned == b
        if mask.sum() == 0:
            continue
        e_mv, e_bs = err_mv[mask], err_bs[mask]
        gain_b = 1 - np.mean(e_mv ** 2) / np.mean(e_bs ** 2)
        sum_mv += np.sum(e_mv ** 2)
        sum_bs += np.sum(e_bs ** 2)
        bucket_gains[b] = gain_b

        moavg = float('nan')
        if 'month' in df_eval.columns:
            gb = _monthly_gains(e_mv, e_bs, df_eval.loc[mask, 'month'])
            gb = gb[gb['n'] >= MIN_BUCKET_MONTH_OBS]
            if len(gb):
                moavg = gb['gain'].mean()
        print(f"{b:>13.1f} {gain_b:>10.4f} {moavg:>12.4f} {mask.sum():>8}")

    print("-" * 52)
    print(f"{'overall':>13} {1 - sum_mv / sum_bs:>10.4f}")
    return gain_ratio, bucket_gains, df_eval
