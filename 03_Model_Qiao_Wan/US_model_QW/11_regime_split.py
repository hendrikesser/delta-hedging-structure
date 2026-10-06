"""Volatility-regime split of hedging gains (thesis RQ3b).

Splits the test days into low / mid / high volatility terciles and reports
the pooled gain of the NN (this folder's MODEL_NAME/FLAG) and the HW
benchmark within each regime. Two split variables:
  1. the market volatility index (VIX for the US, VOL_SERIES for China);
  2. the daily BS hedging MSE (a model-free "hedging difficulty" proxy),
     as a robustness variant.

Needs only existing artifacts: the daily_sse_* dumps (07/08 runners) and the
HW errors_* dump on the matching window. No training. Run: python 11_regime_split.py
"""
import glob
import importlib as _il
import os
import numpy as np
import pandas as pd

config = _il.import_module('01_config')

_here = os.path.basename(os.getcwd())
HW_DIR = os.path.join('..', '..', '01_Model_HW',
                      'CHN_model_HW' if 'CHN' in _here else 'US_model_HW',
                      'diagnostics_output')


def load_nn():
    tag = str(getattr(config, 'TEST_CUTOFF', 'NA'))[:7]
    files = sorted(glob.glob(f'diagnostics_output/per_run_dumps/daily_sse_{config.MODEL_NAME}_{config.FLAG}_{tag}_run*.csv'))
    if not files:
        raise SystemExit(f"no daily_sse files for {config.MODEL_NAME}/{config.FLAG}/{tag}")
    runs = []
    for f in files:
        g = pd.read_csv(f, parse_dates=['date']).set_index('date')
        runs.append(g['sse_nn'])
    sse_nn = pd.concat(runs, axis=1).mean(axis=1)          # across-run average SSE
    return pd.DataFrame({'sse_nn': sse_nn, 'sse_bs': g['sse_bs'], 'n': g['n']})


def load_hw():
    files = sorted(glob.glob(os.path.join(HW_DIR, 'per_run_dumps', f'errors_{config.FLAG}_*.csv.gz')))
    if not files:
        raise SystemExit(f"no HW dump in {HW_DIR}")
    raw = pd.read_csv(files[-1], parse_dates=['date'])
    g = raw.assign(se_mv=raw['err_mv'] ** 2, se_bs=raw['err_bs'] ** 2).groupby('date')
    return pd.DataFrame({'sse_mv': g['se_mv'].sum(), 'sse_hwbs': g['se_bs'].sum()})


def load_vol_index():
    """Daily volatility-index level; returns a Series or None."""
    try:
        if hasattr(config, 'PATH_VIX'):                       # US
            v = pd.read_csv(config.PATH_VIX)
            dcol = 'Date' if 'Date' in v.columns else 'date'
            vcol = 'vix' if 'vix' in v.columns else v.columns[-1]
            v['date'] = pd.to_datetime(v[dcol])
            return pd.to_numeric(v[vcol], errors='coerce').groupby(v['date']).first()
        if hasattr(config, 'PATH_VOL'):                       # China (Wind export)
            v = pd.read_excel(config.PATH_VOL)
            v.columns = ['date', 'vol_300_index_options', 'vol_50etf_option',
                         'vol_500etf', 'vol_300etf_option']
            v = v[pd.to_datetime(v['date'], errors='coerce').notna()]
            v['date'] = pd.to_datetime(v['date'])
            s = pd.to_numeric(v[config.VOL_SERIES], errors='coerce')
            return s.groupby(v['date']).first().dropna()
    except Exception as e:
        print(f"(vol index unavailable: {e})")
    return None


def report(df, split, name):
    """df: daily frame with sse columns; split: Series aligned to df.index."""
    q = split.rank(pct=True)
    labels = pd.cut(q, [0, 1 / 3, 2 / 3, 1.0], labels=['low', 'mid', 'high'])
    print(f"\nRegime split by {name}:")
    print(f"{'regime':>7} {'days':>6} {'NN gain':>9} {'HW gain':>9}")
    for reg in ['low', 'mid', 'high']:
        m = labels == reg
        if m.sum() == 0:
            continue
        d = df[m.values]
        g_nn = 1 - d['sse_nn'].sum() / d['sse_bs'].sum()
        g_hw = 1 - d['sse_mv'].sum() / d['sse_hwbs'].sum()
        print(f"{reg:>7} {int(m.sum()):>6} {g_nn:>+9.4f} {g_hw:>+9.4f}")


def main():
    nn, hw = load_nn(), load_hw()
    df = nn.join(hw, how='inner')
    if len(df) < 30:
        raise SystemExit(f"only {len(df)} overlapping days -- check the HW dump window")
    print(f"{len(df)} common test days ({df.index.min().date()} - {df.index.max().date()}), "
          f"model = {config.MODEL_NAME}, FLAG = {config.FLAG}")
    print(f"full-window pooled gains: NN {1 - df['sse_nn'].sum() / df['sse_bs'].sum():+.4f}, "
          f"HW {1 - df['sse_mv'].sum() / df['sse_hwbs'].sum():+.4f}")

    vol = load_vol_index()
    if vol is not None:
        common = df.index.intersection(vol.index)
        report(df.loc[common], vol.loc[common], 'volatility index level')

    # Model-free robustness variant: daily BS MSE as hedging-difficulty proxy
    report(df, df['sse_bs'] / df['n'], 'daily BS MSE (difficulty proxy)')


if __name__ == '__main__':
    main()
