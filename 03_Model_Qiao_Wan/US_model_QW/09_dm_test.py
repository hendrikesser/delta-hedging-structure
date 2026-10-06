"""Diebold-Mariano significance tests on daily MSE series: NN vs HW vs BS.

Prerequisites (same FLAG, same test window):
  - NN daily SSEs:  diagnostics_output/per_run_dumps/daily_sse_{MODEL_NAME}_{FLAG}_run*.csv
    (written automatically by 07/08 runners)
  - HW error dump:  ../../01_Model_HW/<market>_model_HW/diagnostics_output/per_run_dumps/errors_{FLAG}_*.csv.gz
    (written automatically by the HW runner; set EVAL_START/EVAL_END there to
    match this folder's TEST_CUTOFF window before running it)

Method: daily MSE loss differentials d_t = MSE_A(t) - MSE_B(t); DM statistic
= mean(d) / sqrt(NW_var / T) with Newey-West lags = floor(T^(1/3)).
Daily aggregation absorbs the cross-sectional correlation of same-day errors.
NN daily MSE is averaged across available runs.

Usage: python 09_dm_test.py
"""
import glob
import importlib as _il
import os
import numpy as np
import pandas as pd
from scipy import stats

config = _il.import_module('01_config')

_here = os.path.basename(os.getcwd())
HW_DIR = os.path.join('..', '..', '01_Model_HW',
                      'CHN_model_HW' if 'CHN' in _here else 'US_model_HW',
                      'diagnostics_output')


def _newey_west_var(d, lags):
    d = d - d.mean()
    T = len(d)
    v = np.sum(d ** 2) / T
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        v += 2 * w * np.sum(d[l:] * d[:-l]) / T
    return v


def dm_test(mse_a, mse_b, label_a, label_b):
    """H0: equal predictive accuracy. Negative DM => A has lower MSE."""
    d = (mse_a - mse_b).values
    T = len(d)
    lags = int(np.floor(T ** (1 / 3)))
    dm = d.mean() / np.sqrt(_newey_west_var(d, lags) / T)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    better = label_a if d.mean() < 0 else label_b
    print(f"  {label_a:>6} vs {label_b:<6}: DM = {dm:+.3f}  p = {p:.4f}  "
          f"(T = {T} days, NW lags = {lags})  -> lower MSE: {better}")
    return dm, p


def main():
    flag = config.FLAG

    # NN daily MSE, averaged across runs
    tag = str(getattr(config, 'TEST_CUTOFF', 'NA'))[:7]
    files = sorted(glob.glob(f'diagnostics_output/per_run_dumps/daily_sse_{config.MODEL_NAME}_{flag}_{tag}_run*.csv'))
    if not files:  # legacy untagged files
        files = sorted(glob.glob(f'diagnostics_output/per_run_dumps/daily_sse_{config.MODEL_NAME}_{flag}_run*.csv'))
    if not files:
        raise SystemExit(f"no daily_sse files for {config.MODEL_NAME}/{flag} -- run 07/08 first")
    runs = []
    for f in files:
        g = pd.read_csv(f, parse_dates=['date']).set_index('date')
        runs.append(g['sse_nn'] / g['n'])
    nn = pd.concat(runs, axis=1).mean(axis=1)
    bs = (g['sse_bs'] / g['n'])  # identical across runs

    # HW daily MSE
    hw_files = sorted(glob.glob(os.path.join(HW_DIR, 'per_run_dumps', f'errors_{flag}_*.csv.gz')))
    if not hw_files:
        raise SystemExit(f"no HW error dump in {HW_DIR} -- run the HW folder first")
    hw_raw = pd.read_csv(hw_files[-1], parse_dates=['date'])
    hw_g = hw_raw.assign(se_mv=hw_raw['err_mv'] ** 2,
                         se_bs=hw_raw['err_bs'] ** 2).groupby('date')
    hw_daily = hw_g['se_mv'].mean()
    hw_bs_daily = hw_g['se_bs'].mean()
    print(f"loaded: {len(files)} NN runs, HW dump {os.path.basename(hw_files[-1])}")

    # Align on common days (HW window should match TEST_CUTOFF period)
    idx = nn.index.intersection(hw_daily.index)
    if len(idx) < 30:
        raise SystemExit(f"only {len(idx)} overlapping days -- check EVAL_START/EVAL_END in the HW folder")
    nn_, bs_ = nn.loc[idx], bs.loc[idx]
    hw_, hw_bs_ = hw_daily.loc[idx], hw_bs_daily.loc[idx]

    print(f"\nDiebold-Mariano on {len(idx)} common days "
          f"({idx.min().date()} - {idx.max().date()}), FLAG = {flag}, "
          f"model = {config.MODEL_NAME}:")
    dm_test(nn_, bs_, 'NN', 'BS')
    dm_test(hw_, hw_bs_, 'HW', 'BS')
    # Cross-model comparison must be unit-free: the HW dump stores NORMALIZED
    # errors (S=1 convention), the NN dump raw price errors. Compare each
    # model's daily LOG MSE ratio vs its own BS benchmark (arithmetic ratios
    # are Jensen-biased against high-variance hedgers).
    rel_nn = np.log(nn_ / bs_)
    rel_hw = np.log(hw_ / hw_bs_)
    dm_test(rel_nn, rel_hw, 'lnNN/BS', 'lnHW/BS')
    zero = pd.Series(0.0, index=rel_nn.index)
    dm_test(rel_nn, zero, 'lnNN/BS', 'zero')   # typical-day test vs BS
    dm_test(rel_hw, zero, 'lnHW/BS', 'zero')

    print("\nNotes: NN MSE is the across-run average; HW/BS are deterministic. "
          "The NN-vs-HW line compares BS-relative daily MSEs (unit-free); "
          "negative DM means the NN's relative accuracy is better. "
          "Daily equal-weighting complements the pooled gain, which "
          "over-weights high-volatility days.")


if __name__ == '__main__':
    main()
