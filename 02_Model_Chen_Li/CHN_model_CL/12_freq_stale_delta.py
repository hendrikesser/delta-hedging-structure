"""Rebalancing-frequency, matched to Chen & Li / Qiao & Wan (mark-to-market daily).

Why this differs from 08_run_ndgf and the superseded 12_freq_fixed_delta (now archived in 05_old_code)
--------------------------------------------------------
Those hold the hedge for the FULL k days and score a single k-day P&L
(dV_k - delta*dS_k). Over 5-21 days that error is dominated by gamma and time
decay, which are common to the model and the BS delta, so the delta advantage
washes out and the gain collapses. That is not what Chen & Li report
(DNN3 calls: 0.308 daily -> 0.284 weekly -> 0.107 monthly, only a mild decline).

The standard desk meaning of "hedging frequency k" is: you REBALANCE every k
trading days, but you mark the portfolio to market EVERY day. Between rebalances
the hedge ratio is simply held (goes stale). The gain is then computed on the
pooled DAILY hedging errors, using the stale (last-rebalanced) delta:

    err_t = dV_{t->t+1} - delta_stale * dS_{t->t+1}

with delta_stale refreshed only on rebalancing days. The BS benchmark is held
stale at the same frequency, so the comparison is fair. Daily gamma is identical
for model and BS; only staleness of the delta grows with k, which is why the
gain declines mildly rather than collapsing.

For k = 1 this reduces to the standard daily gain.

Usage
-----
    # set FLAG / MODEL_NAME in 01_config.py (or via -c override)
    python 13_freq_stale_delta.py

Set RESIDUAL = True in the Qiao & Wan / ResNet folders.
"""
import importlib as _il
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

config      = _il.import_module('01_config')
dp          = _il.import_module('02_data_pipeline')
build_model = _il.import_module('03_fnn_model').build_model
train_model = _il.import_module('04_trainer').train_model

RESIDUAL = False
RUNS  = 10
KS    = {'daily': 1, 'weekly': 5, 'monthly': 21}
# Chen & Li reference (2019 test): DNN3 calls 0.308 / 0.284 / 0.107


def stale_gain(df, k, delta_col):
    """Pooled DAILY-error gain when the hedge ratio (model and BS) is refreshed
    only every k-th observation of each contract and held stale in between."""
    pos = df.groupby('optionid', sort=False).cumcount().values
    reb = pd.Series(pos % k == 0, index=df.index)
    sm = df[delta_col].where(reb).groupby(df['optionid']).ffill().values  # stale model delta
    sb = df['dbs'].where(reb).groupby(df['optionid']).ffill().values      # stale BS delta
    dV, dS = df['delta_V'].values, df['delta_S'].values
    em = dV - sm * dS
    eb = dV - sb * dS
    return 1.0 - np.mean(em ** 2) / np.mean(eb ** 2)


def main():
    features = config.get_feature_sets(config.FLAG)[config.MODEL_NAME]
    cutoff   = pd.Timestamp(config.TEST_CUTOFF)

    # ---- load + filter, then DAILY pairing only (frequency lives in evaluation)
    raw = dp.load_raw_data()
    filt = dp.filter_data(dp.format_data(*raw), config.FLAG)
    config.HEDGE_FREQ = 'daily'
    df_daily = dp.pair_options_ndg(filt)

    # ---- train once on the daily training split -----------------------------
    df_tv  = df_daily[df_daily['date'] < cutoff]
    X_tv, dV_tv, dS_tv = df_tv[features].values, df_tv['delta_V'].values, df_tv['delta_S'].values
    dbs_tv = df_tv['delta'].values
    X_tr, X_val, dV_tr, dV_val, dS_tr, dS_val, dbs_tr, dbs_val = train_test_split(
        X_tv, dV_tv, dS_tv, dbs_tv, test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE, shuffle=True)
    X_mean, X_std = X_tr.mean(axis=0), X_tr.std(axis=0)

    def _std(a):
        return torch.tensor((a - X_mean) / X_std, dtype=torch.float32)

    def _col(a):
        return torch.tensor(a, dtype=torch.float32).view(-1, 1)

    X_tr_t, X_val_t = _std(X_tr), _std(X_val)
    dV_tr_t, dV_val_t = _col(dV_tr), _col(dV_val)
    dS_tr_t, dS_val_t = _col(dS_tr), _col(dS_val)
    dbs_tr_t, dbs_val_t = _col(dbs_tr), _col(dbs_val)

    # ---- daily TEST frame (features at t, 1-day dV/dS, contract id, date) ----
    test = df_daily[df_daily['date'] >= cutoff].sort_values(['optionid', 'date']).copy()
    X_test_t = _std(test[features].values)
    test['dbs'] = test['delta'].values

    gains = {f: [] for f in KS}
    for r in range(1, RUNS + 1):
        model = build_model(n_features=len(features))
        if RESIDUAL:
            model = train_model(model, X_tr_t, dV_tr_t, dS_tr_t, dbs_tr_t,
                                 X_val_t, dV_val_t, dS_val_t, dbs_val_t)
        else:
            model = train_model(model, X_tr_t, dV_tr_t, dS_tr_t, X_val_t, dV_val_t, dS_val_t)
        model.eval()
        with torch.no_grad():
            raw = model(X_test_t).numpy().squeeze()
        test['delta_model'] = (test['dbs'].values + raw) if RESIDUAL else raw
        for f, k in KS.items():
            gains[f].append(stale_gain(test, k, 'delta_model'))
        print(f"run {r}/{RUNS}: " + " | ".join(f"{f} {gains[f][-1]:+.4f}" for f in KS))

    print("\n" + "=" * 62)
    print(f"STALE-DELTA (daily mark-to-market) frequency gains  "
          f"(FLAG={config.FLAG}, {config.MODEL_NAME}, {'residual' if RESIDUAL else 'direct'}, {RUNS} runs)")
    print(f"{'freq':>8} {'k':>3} {'gain (mean+/-sd)':>22}")
    for f, k in KS.items():
        g = np.array(gains[f])
        print(f"{f:>8} {k:>3} {g.mean():>+12.4f} +/- {g.std():.4f}")
    print("=" * 62)
    print("Target (Chen & Li DNN3 calls): daily 0.308 | weekly 0.284 | monthly 0.107")


if __name__ == '__main__':
    main()
