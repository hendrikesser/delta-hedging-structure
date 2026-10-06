"""Full-feature (kitchen-sink) residual-net test on the CSI 300 ETF.

The headline grid caps each net at THREE features (base [TTM, delta] + one of
VIX / R / IV). This one-off asks the question that grid never does: does stacking
ALL the features into one residual net beat the best single-feature net?

Specs (residual FNN, daily, single seed), calls and puts:
    DNN2      [TTM, delta]                                        (base)
    DNN3-IV   [TTM, delta, impl_volatility]                       (best single)
    Fea6      [TTM, delta, Moneyness, impl_volatility, theta, vega]
    Fea7      [TTM, delta, Moneyness, impl_volatility, theta, vega, gamma]

Prints overall daily gain ratio + CVaR95/99 gain per spec.

Run:  cd 03_Model_Qiao_Wan/CHN_model_QW
      KMP_DUPLICATE_LIB_OK=TRUE python fea_stack_test.py
"""
import importlib as _il, random
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

config      = _il.import_module('01_config')
dp          = _il.import_module('02_data_pipeline')
build_model = _il.import_module('03_fnn_model').build_model
train_model = _il.import_module('04_trainer').train_model

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'supplementary')))
import cvar

SPECS = [
    ('DNN2',    ['TTM', 'delta']),
    ('DNN3-IV', ['TTM', 'delta', 'impl_volatility']),
    ('Fea6',    ['TTM', 'delta', 'Moneyness', 'impl_volatility', 'theta', 'vega']),
    ('Fea7',    ['TTM', 'delta', 'Moneyness', 'impl_volatility', 'theta', 'vega', 'gamma']),
]


def _prep(features, df):
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    tv = df[df['date'] < cutoff]
    X, dV, dS, dbs = tv[features].values, tv['delta_V'].values, tv['delta_S'].values, tv['delta'].values
    Xtr, Xv, dVtr, dVv, dStr, dSv, dbstr, dbsv = train_test_split(
        X, dV, dS, dbs, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE, shuffle=True)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    st = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    cl = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1)
    targs = (st(Xtr), cl(dVtr), cl(dStr), cl(dbstr), st(Xv), cl(dVv), cl(dSv), cl(dbsv))
    return targs, mu, sd


def _overall_and_cvar(test0):
    dV, dS = test0['delta_V'].values, test0['delta_S'].values
    em = dV - test0['dm'].values * dS
    eb = dV - test0['dbs'].values * dS
    gain = 1 - (em ** 2).sum() / (eb ** 2).sum()
    return gain, cvar.cvar_gain(em, eb, 0.95), cvar.cvar_gain(em, eb, 0.99)


def run_flag(flag):
    config.FLAG = flag
    config.HEDGE_FREQ = 'daily'
    merged = dp.format_data(*dp.load_raw_data())
    filt = dp.filter_data(merged, flag)
    df = dp.pair_options_ndg(filt)
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    test0 = df[df['date'] >= cutoff].sort_values(['optionid', 'date']).copy()
    test0['dbs'] = test0['delta'].values
    side = 'CALLS' if flag == config.Call else 'PUTS'
    print(f"\n==== {side} | CSI 300 ETF | residual FNN, daily (single seed) ====")
    print(f"{'spec':>9} {'nfeat':>5} {'overall':>9} {'cvar95':>9} {'cvar99':>9}")
    for name, feats in SPECS:
        targs, mu, sd = _prep(feats, df)
        Xte = torch.tensor((test0[feats].values - mu) / sd, dtype=torch.float32)
        model = build_model(n_features=len(feats))
        model = train_model(model, targs[0], targs[1], targs[2], targs[3],
                            targs[4], targs[5], targs[6], targs[7])
        model.eval()
        with torch.no_grad():
            out = model(Xte).numpy().squeeze()
        test0['dm'] = test0['dbs'].values + out
        g, c95, c99 = _overall_and_cvar(test0)
        print(f"{name:>9} {len(feats):>5} {g:>9.4f} {c95:>9.4f} {c99:>9.4f}")


def main():
    seed = getattr(config, 'SEED', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"UNDERLYING={getattr(config,'UNDERLYING','ETF')}  TEST_CUTOFF={config.TEST_CUTOFF}  SEED={seed}")
    for flag in (config.Call, config.Put):
        run_flag(flag)


if __name__ == '__main__':
    main()
