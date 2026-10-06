"""Bucketed rebalancing-frequency gain table for the residual-GRU hedger.

Same layout and metrics as the Chen & Li / Qiao & Wan bucket grid (so the numbers
drop straight next to the residual-FNN table), but the model is the GRU hedger:
a GRU encodes each contract's trailing SEQ_LEN feature vectors, its last hidden
state is concatenated with the time-t features, and an MLP head outputs the
residual correction (delta_total = delta_BS + f).

Sequences are built per contract over the WHOLE sample, normalised with TRAIN
statistics, so a test row still sees its genuine pre-cutoff path (all values <= t,
no look-ahead). Everything else -- filter, next-day pairing, chronological cut,
stale-delta mark-to-market, CVaR -- is identical to the FNN grid.

Run:  KMP_DUPLICATE_LIB_OK=TRUE python bucket_freq_table.py
      UNDERLYING=INDEX  KMP_DUPLICATE_LIB_OK=TRUE python bucket_freq_table.py
      SEQ_LEN=20        KMP_DUPLICATE_LIB_OK=TRUE python bucket_freq_table.py
"""
import importlib as _il
import random
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

config      = _il.import_module('01_config')
dp          = _il.import_module('02_data_pipeline')
build_model = _il.import_module('03_gru_model').build_model
train_model = _il.import_module('04_trainer').train_model

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'supplementary')))
import cvar
from sequences import build_sequences

RESIDUAL = getattr(config, 'RESIDUAL', True)
RUNS = config.NUM_RUNS
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

def _stale_err(df, k, col):
    pos = df.groupby('optionid', sort=False).cumcount().values
    reb = pd.Series(pos % k == 0, index=df.index)
    sm = df[col].where(reb).groupby(df['optionid']).ffill().values
    sb = df['dbs'].where(reb).groupby(df['optionid']).ffill().values
    dV, dS = df['delta_V'].values, df['delta_S'].values
    return (dV - sm * dS), (dV - sb * dS)

def _prep(features, df):
    """Return (targs, mu, sd, seq_all) for a feature set.

    seq_all is (N, SEQ_LEN, F) aligned to df's row order (df['_pos']); the trainer
    receives 3-D train/val tensors, and run_flag slices seq_all for the test set.
    """
    T = config.SEQ_LEN
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    tv_mask = (df['date'] < cutoff).values
    tv_idx = np.where(tv_mask)[0]
    # train/val split on positions within the pre-cutoff block (shuffled, seeded)
    tr_rel, val_rel = train_test_split(np.arange(len(tv_idx)),
                                       test_size=config.VAL_SPLIT,
                                       random_state=config.RANDOM_STATE, shuffle=True)
    tr_idx, val_idx = tv_idx[tr_rel], tv_idx[val_rel]

    Xall = df[features].values.astype(np.float64)
    mu, sd = Xall[tr_idx].mean(0), Xall[tr_idx].std(0)
    sd = np.where(sd == 0, 1.0, sd)

    dfn = df.copy()
    dfn[features] = (Xall - mu) / sd
    seq_all = build_sequences(dfn, features, T)              # (N, T, F) float32

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    col  = lambda idx, c: torch.tensor(df[c].values[idx], dtype=torch.float32).view(-1, 1)
    targs = (to_t(seq_all[tr_idx]),  col(tr_idx, 'delta_V'),  col(tr_idx, 'delta_S'),  col(tr_idx, 'delta'),
             to_t(seq_all[val_idx]), col(val_idx, 'delta_V'), col(val_idx, 'delta_S'), col(val_idx, 'delta'))
    return targs, mu, sd, seq_all

def run_flag(flag):
    config.FLAG = flag
    config.HEDGE_FREQ = 'daily'
    if getattr(config, 'UNDERLYING', 'ETF') == 'INDEX':
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.abspath(_o.path.join(_o.path.dirname(__file__), '..', '..', 'supplementary')))
        import index_loader
        merged = index_loader.build_merged(config)
    else:
        merged = dp.format_data(*dp.load_raw_data())
    filt = dp.filter_data(merged, flag)
    df = dp.pair_options_ndg(filt).reset_index(drop=True)
    df['_pos'] = np.arange(len(df))
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    test0 = df[df['date'] >= cutoff].sort_values(['optionid', 'date']).copy()
    _te = __import__('os').environ.get('TEST_END')
    if _te:
        test0 = test0[test0['date'] <= pd.Timestamp(_te)].copy()
    test0['dbs'] = test0['delta'].values
    te_pos = test0['_pos'].values
    bk = _assign(test0['dbs'].values, flag)
    levels = _levels(flag)
    side = 'PUT' if flag == config.Put else 'CALL'
    res = {}
    for label, feats in VARIANTS:
        targs, mu, sd, seq_all = _prep(feats, df)
        Xte = torch.tensor(seq_all[te_pos], dtype=torch.float32)
        acc = {f: {'overall': [], 'cvar95': [], 'cvar99': []} for f in KS}
        for f in KS:
            for b in levels: acc[f][b] = []
        for r in range(RUNS):
            model = build_model(n_features=len(feats))
            if RESIDUAL:
                # residual: pass dbs (train, val) so delta_total = delta_BS + f
                model = train_model(model, targs[0], targs[1], targs[2],
                                    targs[4], targs[5], targs[6],
                                    dbs_train_t=targs[3], dbs_val_t=targs[7])
            else:
                # direct: omit dbs so the model output is the delta itself
                model = train_model(model, targs[0], targs[1], targs[2],
                                    targs[4], targs[5], targs[6])
            model.eval()
            with torch.no_grad():
                out = model(Xte).numpy().squeeze()
            test0['dm'] = (test0['dbs'].values + out) if RESIDUAL else out
            for f, k in KS.items():
                em2, eb2 = _stale_sse(test0, k, 'dm')
                acc[f]['overall'].append(1 - em2.sum() / eb2.sum())
                em, eb = _stale_err(test0, k, 'dm')
                acc[f]['cvar95'].append(cvar.cvar_gain(em, eb, 0.95))
                acc[f]['cvar99'].append(cvar.cvar_gain(em, eb, 0.99))
                for b in levels:
                    m = bk == b
                    if m.sum(): acc[f][b].append(1 - em2[m].sum() / eb2[m].sum())
            print(f"  {label} {side} run {r + 1}/{RUNS} done")
        res[label] = {f: {key: (float(np.mean(v)) if len(v) else float('nan'))
                          for key, v in acc[f].items()} for f in KS}
    labels = [l for l, _ in VARIANTS]
    kind = 'residual' if RESIDUAL else 'direct'
    for f in KS:
        print(f"\n==== {'CALLS' if flag==config.Call else 'PUTS'} | {f.upper()} hedging "
              f"({kind} GRU, T={config.SEQ_LEN}, stale-delta mark-to-market daily) ====")
        print(f"{'delta':>7} " + " ".join(f"{l:>10}" for l in labels))
        for b in levels:
            print(f"{b:>7.1f} " + " ".join(f"{res[l][f].get(b, float('nan')):>10.4f}" for l in labels))
        print(f"{'overall':>7} " + " ".join(f"{res[l][f]['overall']:>10.4f}" for l in labels))
        print(f"{'cvar95':>7} " + " ".join(f"{res[l][f]['cvar95']:>10.4f}" for l in labels))
        print(f"{'cvar99':>7} " + " ".join(f"{res[l][f]['cvar99']:>10.4f}" for l in labels))

def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    _set_seed(getattr(config, 'SEED', 42))
    print(f"GRU bucket grid | UNDERLYING={getattr(config,'UNDERLYING','ETF')} "
          f"| residual={RESIDUAL} | SEQ_LEN={config.SEQ_LEN} | SEED={getattr(config,'SEED',42)} "
          f"| TEST_CUTOFF={config.TEST_CUTOFF}")
    for flag in (config.Call, config.Put):
        run_flag(flag)

if __name__ == '__main__':
    main()
