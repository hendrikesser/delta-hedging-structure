"""Diagnostics for the CHN HW model: why are the out-of-sample gains negative?

Checks, in order of diagnostic value:
  1. Price-vol relation: full-sample and rolling correlation of d(vix) with the
     ETF daily return. HW's model requires a stable, strongly negative relation
     (S&P 500: rolling corr ~ -0.7 and never flips sign). A weak or
     sign-flipping relation makes negative OOS gains structural.
  2. Rolling OLS coefficients (a, b, c) per evaluation month (paper Fig. 1
     analogue). Wild swings => estimation instability.
  3. In-sample ceiling: (a, b, c) fitted once on the full sample, gain measured
     in-sample. Upper bound for any rolling-window scheme; if ~0, no window
     choice can produce positive OOS gains.
  4. Data coverage: observations per month. Months with no data silently shift
     the rolling evaluation start (calls started 2021-12, puts 2022-12).

Outputs: console + CSV/PNG in ./diagnostics_output/.
Run:     python 06_diagnostics.py   (uses FLAG / HEDGE_FREQ from 01_config.py)
"""
import importlib as _il
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

config                   = _il.import_module('01_config')
build_pipeline           = _il.import_module('02_data_pipeline').build_pipeline
add_regression_variables = _il.import_module('03_hw_features').add_regression_variables
rolling_ols              = _il.import_module('04_rolling_ols').rolling_ols

OUT = 'diagnostics_output/diagnostics'
os.makedirs(OUT, exist_ok=True)
ROLL_DAYS = 252  # ~12 trading months


def _plot(fig_fn, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig_fn(plt)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    saved {path}")
    except ImportError:
        print("    (matplotlib not available -- plot skipped)")


def price_vol_relation(df_final):
    print("\n[1] Price-vol relation (drives the whole HW correction)")
    daily = (df_final.groupby('date')
             .agg(S=('S', 'first'), vix=('vix', 'first'))
             .sort_index())
    daily['R']    = daily['S'].pct_change()
    daily['dvix'] = daily['vix'].diff()
    daily = daily.dropna()

    full_corr = daily['R'].corr(daily['dvix'])
    print(f"    full-sample corr(R, dVIX) = {full_corr:+.3f} over {len(daily)} days")
    print("    (S&P 500 benchmark: ~ -0.7, stable; near 0 or sign-flipping =>")
    print("     negative OOS gains are structural, not an estimation problem)")

    daily['roll_corr'] = daily['R'].rolling(ROLL_DAYS).corr(daily['dvix'])
    by_year = daily['roll_corr'].groupby(daily.index.year).mean()
    print(f"    rolling {ROLL_DAYS}d corr, mean by year:")
    for y, v in by_year.dropna().items():
        print(f"      {y}: {v:+.3f}")
    pos_share = (daily['roll_corr'] > 0).mean()
    print(f"    share of days with rolling corr > 0: {pos_share:.1%}")

    daily.to_csv(os.path.join(OUT, f'price_vol_relation_{config.FLAG}.csv'))
    _plot(lambda plt: (plt.figure(figsize=(9, 3)),
                       plt.plot(daily.index, daily['roll_corr']),
                       plt.axhline(0, color='k', lw=0.5),
                       plt.title(f'Rolling {ROLL_DAYS}d corr(R, dVIX)'),
                       plt.ylabel('corr')),
          os.path.join(OUT, f'rolling_corr_{config.FLAG}.png'))
    return daily


def coefficient_stability(coeffs_df):
    print("\n[2] Rolling OLS coefficients (paper Fig. 1 analogue)")
    print(coeffs_df.describe().loc[['mean', 'std', 'min', 'max']].round(3))
    for c in ['a', 'b', 'c']:
        flips = (np.sign(coeffs_df[c]).diff().abs() > 0).sum()
        print(f"    {c}: {flips} sign changes across {len(coeffs_df)} months")

    coeffs_df.to_csv(os.path.join(OUT, f'coefficients_{config.FLAG}.csv'))
    idx = coeffs_df.index.to_timestamp()
    _plot(lambda plt: (plt.figure(figsize=(9, 3)),
                       [plt.plot(idx, coeffs_df[c], label=c) for c in ['a', 'b', 'c']],
                       plt.axhline(0, color='k', lw=0.5),
                       plt.legend(), plt.title('Rolling (a, b, c) per evaluation month')),
          os.path.join(OUT, f'coefficients_{config.FLAG}.png'))


def in_sample_ceiling(df_final):
    print("\n[3] In-sample ceiling (full-sample fit, gain measured in-sample)")
    X, y = df_final[['X1', 'X2', 'X3']], df_final['y']
    model = LinearRegression(fit_intercept=False).fit(X, y)
    resid = y - model.predict(X)
    gain = 1 - np.sum(resid ** 2) / np.sum(y ** 2)
    a, b, c = model.coef_
    print(f"    full-sample (a, b, c) = ({a:+.3f}, {b:+.3f}, {c:+.3f})")
    print(f"    in-sample gain = {gain:+.4f}")
    print("    (upper bound for any window; ~0 => the relation is absent,")
    print("     clearly positive => OOS failure comes from instability)")
    return gain


def data_coverage(df_final):
    print("\n[4] Data coverage by month")
    counts = df_final.groupby(df_final['date'].dt.to_period('M')).size()
    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq='M')
    missing = full_range.difference(counts.index)
    print(f"    {counts.index.min()} - {counts.index.max()}: "
          f"median {counts.median():.0f} obs/month, min {counts.min()}")
    if len(missing):
        print(f"    months with ZERO observations ({len(missing)}) -- these shift "
              f"the rolling eval start: {list(missing.astype(str))}")
    else:
        print("    no gap months")
    counts.to_csv(os.path.join(OUT, f'obs_per_month_{config.FLAG}.csv'))


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Diagnostics -- FLAG: {config.FLAG} | HEDGE_FREQ: {config.HEDGE_FREQ} | "
          f"WINDOW_MONTHS: {config.WINDOW_MONTHS}")

    df_final = build_pipeline(pairing='ndg')
    df_final = add_regression_variables(df_final)

    price_vol_relation(df_final)
    _, coeffs_df = rolling_ols(df_final)
    coefficient_stability(coeffs_df)
    in_sample_ceiling(df_final)
    data_coverage(df_final)

    print(f"\nAll outputs in ./{OUT}/")


if __name__ == '__main__':
    main()
