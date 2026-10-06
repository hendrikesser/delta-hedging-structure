"""Transaction-cost analysis (thesis RQ3c) -- Chen & Li DIRECT-net variant.

Same analysis as the residual folders' 10_transaction_costs.py, but this
folder's network outputs the hedge ratio DIRECTLY (delta_NN = model(x)), not a
correction to the BS delta, and its trainer takes no BS-delta argument.
Reconstructs each contract's hedge path, measures turnover for the BS delta and
the direct NN, and reports the net-of-cost gain across a cost sweep.

Cost model: c * |delta_t - delta_{t-1}| * S_t per rebalance (c in bps of
notional traded); opening cost c*|delta|*S at a contract's first observation.
Net one-step P&L = (dV - delta*dS) - cost; net-of-cost gain =
1 - sum(net_NN^2)/sum(net_BS^2). Costs applied identically to BS and NN, so the
comparison isolates the extra turnover the learned hedge incurs.

Reuses build_pipeline / build_model / train_model. Trains one model.
Run: python 10_transaction_costs.py
"""
import importlib as _il
import numpy as np
import pandas as pd
import torch

config         = _il.import_module('01_config')
build_pipeline = _il.import_module('02_data_pipeline').build_pipeline
build_model    = _il.import_module('03_fnn_model').build_model
train_model    = _il.import_module('04_trainer').train_model

PAIRING  = 'ndg'                    # 'ndg' (strict headline) | 'random' (dgf)
COST_BPS = [0, 1, 2, 5, 10, 20]     # proportional cost per unit notional traded (bps)


def main():
    arrays, features = build_pipeline(pairing=PAIRING)
    (X_train, X_val, X_test,
     dV_train, dV_val, dV_test,
     dS_train, dS_val, dS_test,
     df_test, X_mean, X_std) = arrays

    dbs_te = df_test['delta'].values

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    v = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1)

    model = build_model(n_features=len(features))
    model = train_model(model,
                        t(X_train), v(dV_train), v(dS_train),
                        t(X_val),   v(dV_val),   v(dS_val))          # direct net: no BS-delta arg
    model.eval()
    with torch.no_grad():
        dnn_te = model(t(X_test)).numpy().squeeze()                 # direct net: output IS the delta

    d = pd.DataFrame({
        'optionid': df_test['optionid'].values,
        'date':     pd.to_datetime(df_test['date'].values),
        'S':        df_test['S'].values,
        'dbs': dbs_te, 'dnn': dnn_te,
        'dV':  dV_test, 'dS': dS_test,
    }).sort_values(['optionid', 'date'])

    g = d.groupby('optionid')
    d['turn_bs'] = (d['dbs'] - g['dbs'].shift(1)).abs().fillna(d['dbs'].abs())
    d['turn_nn'] = (d['dnn'] - g['dnn'].shift(1)).abs().fillna(d['dnn'].abs())
    d['err_bs']  = d['dV'] - d['dbs'] * d['dS']
    d['err_nn']  = d['dV'] - d['dnn'] * d['dS']

    gross      = 1 - (d['err_nn'] ** 2).sum() / (d['err_bs'] ** 2).sum()
    tb, tn     = d['turn_bs'].sum(), d['turn_nn'].sum()
    bs_mse_sum = (d['err_bs'] ** 2).sum()

    print(f"\nPairing={PAIRING}  Model={config.MODEL_NAME}  Flag={config.FLAG}  "
          f"| {len(d):,} pairs, {d['optionid'].nunique():,} contracts  [DIRECT net]")
    print(f"Gross gain (no cost):     {gross:+.4f}")
    print(f"Turnover  sum|d hedge|:   BS {tb:,.0f}  |  NN {tn:,.0f}  |  NN/BS {tn / tb:.2f}x")
    print(f"\n  cost(bps) | net gain | NN cost-drag | BS cost-drag   (drag = sum(cost^2)/sum(BS err^2))")
    prev = None
    for bp in COST_BPS:
        c       = bp / 1e4
        cost_bs = c * d['turn_bs'] * d['S']
        cost_nn = c * d['turn_nn'] * d['S']
        net_bs  = d['err_bs'] - cost_bs
        net_nn  = d['err_nn'] - cost_nn
        gnet    = 1 - (net_nn ** 2).sum() / (net_bs ** 2).sum()
        flag    = '  <- crosses 0' if (prev is not None and prev > 0 >= gnet) else ''
        print(f"  {bp:>8} | {gnet:>+8.4f} | {(cost_nn**2).sum()/bs_mse_sum:>12.4f} | "
              f"{(cost_bs**2).sum()/bs_mse_sum:>12.4f}{flag}")
        prev = gnet


if __name__ == '__main__':
    main()
