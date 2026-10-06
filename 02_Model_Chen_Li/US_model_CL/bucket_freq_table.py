"""Bucketed rebalancing-frequency gain table (Chen & Li Table 5/6 style, plus daily).

For calls and puts, prints the RUNS-run mean stale-delta gain ratio per |delta|
bucket at daily / weekly / monthly frequency, for four feature specifications
(columns): DNN2 [TTM, delta], DNN3-VIX [+vix], DNN3-R [+index return],
DNN3-IV [+contract IV]. Construction matches 12_rebalancing_frequency /
13_freq_stale_delta (rebalance every k trading days, hold the delta STALE
between rebalances, mark to market DAILY, pool the daily errors). Averaging over
RUNS = config.NUM_RUNS matches the 10-run methodology of the headline tables.

Run:  KMP_DUPLICATE_LIB_OK=TRUE python bucket_freq_table.py
Set RESIDUAL = True in the Qiao & Wan / ResNet folders (baked in per folder).
Lower RUNS below for a faster preview.
"""
import importlib as _il
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

config      = _il.import_module('01_config')
dp          = _il.import_module('02_data_pipeline')
build_model = _il.import_module('03_fnn_model').build_model
train_model = _il.import_module('04_trainer').train_model

RESIDUAL = False
RUNS = config.NUM_RUNS          # 10-run mean, as in the headline tables; lower for a quick preview
KS = {'daily': 1, 'weekly': 5, 'monthly': 21}
VARIANTS = [('DNN2',     ['TTM', 'delta']),
            ('DNN3-VIX', ['TTM', 'delta', 'vix']),
            ('DNN3-R',   ['TTM', 'delta', 'R']),
            ('DNN3-IV',  ['TTM', 'delta', 'impl_volatility'])]


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

def _prep(features, df):
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    tv = df[df['date'] < cutoff]
    X, dV, dS, dbs = tv[features].values, tv['delta_V'].values, tv['delta_S'].values, tv['delta'].values
    Xtr, Xv, dVtr, dVv, dStr, dSv, dbstr, dbsv = train_test_split(
        X, dV, dS, dbs, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE, shuffle=True)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    st = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    cl = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1)
    targs = (st(Xtr), cl(dVtr), cl(dStr), cl(dbstr), st(Xv), cl(dVv), cl(dSv), cl(dbsv))
    return targs, mu, sd

def run_flag(flag):
    config.FLAG = flag
    config.HEDGE_FREQ = 'daily'
    filt = dp.filter_data(dp.format_data(*dp.load_raw_data()), flag)
    df = dp.pair_options_ndg(filt)
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    test0 = df[df['date'] >= cutoff].sort_values(['optionid', 'date']).copy()
    test0['dbs'] = test0['delta'].values
    bk = _assign(test0['dbs'].values, flag)
    levels = _levels(flag)
    side = 'PUT' if flag == config.Put else 'CALL'
    res = {}
    for label, feats in VARIANTS:
        targs, mu, sd = _prep(feats, df)
        Xte = torch.tensor((test0[feats].values - mu) / sd, dtype=torch.float32)
        acc = {f: {'overall': []} for f in KS}
        for f in KS:
            for b in levels: acc[f][b] = []
        for r in range(RUNS):
            model = build_model(n_features=len(feats))
            if RESIDUAL:
                model = train_model(model, targs[0], targs[1], targs[2], targs[3],
                                    targs[4], targs[5], targs[6], targs[7])
            else:
                model = train_model(model, targs[0], targs[1], targs[2],
                                    targs[4], targs[5], targs[6])
            model.eval()
            with torch.no_grad():
                out = model(Xte).numpy().squeeze()
            test0['dm'] = (test0['dbs'].values + out) if RESIDUAL else out
            for f, k in KS.items():
                em2, eb2 = _stale_sse(test0, k, 'dm')
                acc[f]['overall'].append(1 - em2.sum() / eb2.sum())
                for b in levels:
                    m = bk == b
                    if m.sum(): acc[f][b].append(1 - em2[m].sum() / eb2[m].sum())
            print(f"  {label} {side} run {r + 1}/{RUNS} done")
        res[label] = {f: {key: (float(np.mean(v)) if len(v) else float('nan'))
                          for key, v in acc[f].items()} for f in KS}
    labels = [l for l, _ in VARIANTS]
    for f in KS:
        print(f"\n==== {'CALLS' if flag==config.Call else 'PUTS'} | {f.upper()} hedging "
              f"({RUNS}-run mean, stale-delta mark-to-market daily) ====")
        print(f"{'delta':>7} " + " ".join(f"{l:>10}" for l in labels))
        for b in levels:
            print(f"{b:>7.1f} " + " ".join(f"{res[l][f].get(b, float('nan')):>10.4f}" for l in labels))
        print(f"{'overall':>7} " + " ".join(f"{res[l][f]['overall']:>10.4f}" for l in labels))

def main():
    for flag in (config.Call, config.Put):
        run_flag(flag)

if __name__ == '__main__':
    main()
