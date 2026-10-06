"""Bucketed rebalancing-frequency gain table for the Hull & White MV delta.

For calls and puts, prints the stale-delta gain ratio per |delta| bucket at
daily / weekly / monthly frequency (single MV-model column). Same stale-delta
construction as 12_rebalancing_frequency; delta_MV is computed exactly as in
05_evaluate.py, so the daily overall matches the headline gain.

Run:  KMP_DUPLICATE_LIB_OK=TRUE python bucket_freq_table.py
"""
import importlib as _il
import numpy as np, pandas as pd

config = _il.import_module('01_config')
dp     = _il.import_module('02_data_pipeline')
hf     = _il.import_module('03_hw_features')
ro     = _il.import_module('04_rolling_ols')

KS = {'daily': 1, 'weekly': 5, 'monthly': 21}

def _levels(flag):
    s = 1 if flag == config.Call else -1
    return [round(s * 0.1 * i, 1) for i in range(1, 10)]

def _assign(dbs, flag):
    s = 1 if flag == config.Call else -1
    idx = np.clip(np.floor((np.abs(dbs) - 0.05) / 0.1 + 1e-9).astype(int), 0, 8)
    return s * np.round(0.1 * (idx + 1), 1)

def _stale_sse(df, k, col):
    pos = df.groupby('optionid', sort=False).cumcount().values
    reb = pd.Series(pos % k == 0, index=df.index)
    sm = df[col].where(reb).groupby(df['optionid']).ffill().values
    sb = df['dbs'].where(reb).groupby(df['optionid']).ffill().values
    dV, dS = df['delta_V'].values, df['delta_S'].values
    return (dV - sm * dS) ** 2, (dV - sb * dS) ** 2

def run_flag(flag):
    config.FLAG = flag
    config.HEDGE_FREQ = 'daily'
    df = dp.build_pipeline(pairing='ndg')
    df = hf.add_regression_variables(df)
    df_eval, _ = ro.rolling_ols(df)
    df_eval = df_eval.copy()
    sqrt_T = np.sqrt(df_eval['TTM'].replace(0, 1e-6))
    df_eval['delta_MV'] = df_eval['delta'] + (df_eval['vega'] / (df_eval['S'] * sqrt_T)) * (
        df_eval['a'] + df_eval['b'] * df_eval['delta'] + df_eval['c'] * df_eval['delta'] ** 2)
    test = df_eval.sort_values(['optionid', 'date']).copy()  # df_eval is already the OOS evaluation window
    test['dbs'] = test['delta'].values
    test['dm']  = test['delta_MV'].values
    bk = _assign(test['dbs'].values, flag)
    side = 'CALLS' if flag == config.Call else 'PUTS'
    cells = {}
    for f, k in KS.items():
        em2, eb2 = _stale_sse(test, k, 'dm')
        cells[f] = {'overall': 1 - em2.sum() / eb2.sum()}
        for b in _levels(flag):
            m = bk == b
            if m.sum(): cells[f][b] = 1 - em2[m].sum() / eb2[m].sum()
    print(f"\n==== {side} | HW minimum-variance delta (stale-delta, mark-to-market daily) ====")
    print(f"{'delta':>7} {'daily':>10} {'weekly':>10} {'monthly':>10}")
    for b in _levels(flag):
        print(f"{b:>7.1f} " + " ".join(f"{cells[f].get(b, float('nan')):>10.4f}" for f in KS))
    print(f"{'overall':>7} " + " ".join(f"{cells[f]['overall']:>10.4f}" for f in KS))

def main():
    for flag in (config.Call, config.Put):
        run_flag(flag)

if __name__ == '__main__':
    main()
