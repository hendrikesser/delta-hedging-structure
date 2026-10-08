"""Residual-objective ablation + significance test (peer-review points 2.1, 2.2).

The paper attributes the residual network's China edge to "anchoring on the BS
delta", but the Qiao & Wan network also differs from Chen & Li in activation
(sigmoid vs ReLU), output constraint (unconstrained vs sign-constrained) and
learning rate. That is a CONFOUND: the gap could come from those choices, not
the residual objective.

This script removes the confound. It trains ONE identical backbone (3 x 128,
ReLU, BatchNorm, unconstrained linear output, same optimiser/lr/epochs) in two
variants that differ ONLY in the objective:
    direct   : delta = f(x)
    residual : delta = delta_BS + f(x)
so any difference is attributable to the residual anchor alone.

It then runs a Diebold-Mariano test on the daily MSE series of the two variants
(log-ratio, Newey-West) to see whether the residual advantage is statistically
distinguishable from zero -- and, for context, each variant against the BS delta.

Usage (US calls / China puts):
    python -c "import importlib;c=importlib.import_module('01_config');c.FLAG=c.Call;c.MODEL_NAME='DNN3';importlib.import_module('14_residual_ablation').main()"
"""
import importlib as _il
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import train_test_split

config = _il.import_module('01_config')
dp     = _il.import_module('02_data_pipeline')

NRUNS  = 6
EPOCHS = 40
BATCH  = 4096
LR     = 1e-3


class MLP(nn.Module):
    """Fixed backbone shared by both variants (unconstrained output)."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train(Xtr, dVtr, dStr, dbstr, residual):
    m = MLP(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=LR)
    n = len(Xtr)
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            out = m(Xtr[b])
            delta = (dbstr[b] + out) if residual else out
            loss = torch.mean((dVtr[b] - delta * dStr[b]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def _dm(mse_a, mse_b):
    """Diebold-Mariano on log-MSE ratio (a - b). d<0 => a (residual) better."""
    d = np.log(mse_a) - np.log(mse_b)
    T = len(d); dbar = d.mean(); L = max(1, int(np.floor(T ** (1 / 3))))
    var = np.mean((d - dbar) ** 2)
    for l in range(1, L + 1):
        cov = np.mean((d[l:] - dbar) * (d[:-l] - dbar))
        var += 2 * (1 - l / (L + 1)) * cov
    stat = dbar / np.sqrt(var / T)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def main():
    feats  = config.get_feature_sets(config.FLAG)[config.MODEL_NAME]
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    filt = dp.filter_data(dp.format_data(*dp.load_raw_data()), config.FLAG)
    config.HEDGE_FREQ = 'daily'
    df = dp.pair_options_ndg(filt)

    start_date = pd.to_datetime(df['date'])
    target_date = pd.to_datetime(df['date_target'])

    train_mask = (start_date < cutoff) & (target_date < cutoff)
    tv = df.loc[train_mask].copy()

    Xtv, dVtv, dStv, dbstv = (
        tv[feats].values,
        tv['delta_V'].values,
        tv['delta_S'].values,
        tv['delta'].values,
    )
    Xtr, _, dVtr, _, dStr, _, dbstr, _ = train_test_split(
        Xtv, dVtv, dStv, dbstv,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    mu, sd = Xtr.mean(0), Xtr.std(0)

    test_mask = (start_date >= cutoff) & (target_date >= cutoff)
    te = df.loc[test_mask].sort_values(['optionid', 'date']).copy()    
    Xte = torch.tensor((te[feats].values - mu) / sd, dtype=torch.float32)
    dVte, dSte, dbste = te['delta_V'].values, te['delta_S'].values, te['delta'].values
    dates = pd.to_datetime(te['date'].values)

    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    dVtr_t = torch.tensor(dVtr, dtype=torch.float32)
    dStr_t = torch.tensor(dStr, dtype=torch.float32)
    dbstr_t = torch.tensor(dbstr, dtype=torch.float32)

    print(f"training on {len(Xtr_t):,} rows, {EPOCHS} epochs x {NRUNS} runs x 2 variants ...")
    preds = {}
    for name, res in [('direct', False), ('residual', True)]:
        print(f"  {name:>8}: ", end="", flush=True)
        acc = np.zeros(len(te))
        for r in range(NRUNS):
            m = _train(Xtr_t, dVtr_t, dStr_t, dbstr_t, res); m.eval()
            with torch.no_grad():
                out = m(Xte).numpy()
            acc += (dbste + out) if res else out
            print(f"{r + 1}", end=" ", flush=True)
        print("done")
        preds[name] = acc / NRUNS          # run-averaged delta

    se = {k: (dVte - preds[k] * dSte) ** 2 for k in preds}
    se['bs'] = (dVte - dbste * dSte) ** 2
    mse = {k: v.mean() for k, v in se.items()}
    gain = {k: 1 - mse[k] / mse['bs'] for k in ('direct', 'residual')}

    daily = pd.DataFrame({'date': dates, **{k: se[k] for k in se}}).groupby('date').mean()

    print("\n" + "=" * 64)
    print(f"RESIDUAL-OBJECTIVE ABLATION  (FLAG={config.FLAG}, {config.MODEL_NAME}, "
          f"identical backbone, {NRUNS} runs, {len(te):,} obs, {len(daily)} days)")
    print(f"  direct   gain vs BS : {gain['direct']:+.4f}")
    print(f"  residual gain vs BS : {gain['residual']:+.4f}")
    print(f"  residual - direct   : {gain['residual'] - gain['direct']:+.4f}")
    s, p = _dm(daily['residual'].values, daily['direct'].values)
    print(f"  DM residual vs direct: stat {s:+.2f}, p = {p:.3f}   "
          f"({'residual sig. better' if (p < 0.05 and s < 0) else 'NOT significant'})")
    s, p = _dm(daily['residual'].values, daily['bs'].values)
    print(f"  DM residual vs BS    : stat {s:+.2f}, p = {p:.3f}")
    s, p = _dm(daily['direct'].values, daily['bs'].values)
    print(f"  DM direct   vs BS    : stat {s:+.2f}, p = {p:.3f}")
    print("=" * 64)


if __name__ == '__main__':
    main()
