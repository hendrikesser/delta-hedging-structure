"""seed_sweep.py -- seed-sensitivity of the headline daily gain.

Yu asked how much the single-seed numbers move across seeds. This retrains the
model over N seeds and reports, per side, the overall daily gain ratio for each
feature spec: every seed's value plus mean / std / min / max. It reuses the
folder's own bucket_freq_table internals, so the training, pairing, cut and
stale-delta scoring are identical to the headline grid -- only the seed changes.

RESIDUAL is inherited from bucket_freq_table (True here in Qiao & Wan, False in
Chen & Li), so drop this same file in either folder.

Run:  KMP_DUPLICATE_LIB_OK=TRUE python seed_sweep.py
      SEEDS=42,0,1,2,3            KMP_DUPLICATE_LIB_OK=TRUE python seed_sweep.py
      UNDERLYING=INDEX SPECS=DNN3-IV KMP_DUPLICATE_LIB_OK=TRUE python seed_sweep.py
"""
import os, importlib as _il
import numpy as np, pandas as pd, torch

bft = _il.import_module('bucket_freq_table')   # reuse the exact grid pipeline
config = bft.config

SEEDS = [int(s) for s in os.environ.get('SEEDS', '42,0,1,2,3,4,5,6,7,8').split(',')]
# which feature specs to sweep (default: all four, matching the grid columns)
_want = os.environ.get('SPECS')
SPECS = [v for v in bft.VARIANTS if (_want is None or v[0] in _want.split(','))]


def _overall_daily(test0):
    em2, eb2 = bft._stale_sse(test0, 1, 'dm')       # k=1 -> daily
    return 1 - em2.sum() / eb2.sum()


def sweep_flag(flag):
    config.FLAG = flag
    config.HEDGE_FREQ = 'daily'
    if getattr(config, 'UNDERLYING', 'ETF') == 'INDEX':
        import sys, os as _o
        sys.path.insert(0, _o.path.abspath(_o.path.join(_o.path.dirname(bft.__file__), '..', '..', 'supplementary')))
        import index_loader
        merged = index_loader.build_merged(config)
    else:
        merged = bft.dp.format_data(*bft.dp.load_raw_data())
    filt = bft.dp.filter_data(merged, flag)
    df = bft.dp.pair_options_ndg(filt)
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    test0 = df[df['date'] >= cutoff].sort_values(['optionid', 'date']).copy()
    test0['dbs'] = test0['delta'].values
    side = 'CALLS' if flag == config.Call else 'PUTS'
    print(f"\n==== {side} | {getattr(config,'UNDERLYING','ETF')} | overall daily gain across {len(SEEDS)} seeds "
          f"({'residual' if bft.RESIDUAL else 'direct'} FNN) ====")
    hdr = "  ".join(f"s{sd}" for sd in SEEDS)
    print(f"{'spec':>9} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}   {hdr}")
    for label, feats in SPECS:
        targs, mu, sd = bft._prep(feats, df)
        Xte = torch.tensor((test0[feats].values - mu) / sd, dtype=torch.float32)
        gains = []
        for seed in SEEDS:
            bft._set_seed(seed)
            model = bft.build_model(n_features=len(feats))
            if bft.RESIDUAL:
                model = bft.train_model(model, targs[0], targs[1], targs[2], targs[3],
                                        targs[4], targs[5], targs[6], targs[7])
            else:
                model = bft.train_model(model, targs[0], targs[1], targs[2],
                                        targs[4], targs[5], targs[6])
            model.eval()
            with torch.no_grad():
                out = model(Xte).numpy().squeeze()
            test0['dm'] = (test0['dbs'].values + out) if bft.RESIDUAL else out
            gains.append(_overall_daily(test0))
        g = np.array(gains)
        cells = "  ".join(f"{x:.3f}" for x in g)
        print(f"{label:>9} {g.mean():>7.4f} {g.std():>7.4f} {g.min():>7.4f} {g.max():>7.4f}   {cells}")


def main():
    print(f"seed sweep | UNDERLYING={getattr(config,'UNDERLYING','ETF')} "
          f"| residual={bft.RESIDUAL} | seeds={SEEDS} | TEST_CUTOFF={config.TEST_CUTOFF}")
    for flag in (config.Call, config.Put):
        sweep_flag(flag)


if __name__ == '__main__':
    main()
