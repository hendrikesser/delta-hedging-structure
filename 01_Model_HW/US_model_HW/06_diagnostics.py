"""US price-volatility relation (confirms the value cited in Section 5.1 on our own data).

Computes the correlation between the daily S&P 500 return and the daily change in the
VIX over the 2010-2019 hedging sample, full-sample and on a rolling 252-day window.
This is the US counterpart to the CHN HW price-vol diagnostic: Hull & White's correction
requires a strongly negative, stable relation, so the US should show a large negative,
sign-stable correlation (contrast with China's weak, sign-flipping -0.24).

Run:  KMP_DUPLICATE_LIB_OK=TRUE python 06_diagnostics.py
"""
import importlib as _il
import numpy as np
import pandas as pd

config = _il.import_module('01_config')
ROLL = 252  # ~12 trading months


def main():
    # Read only the index and VIX series (no options needed for the price-vol relation).
    SPX = pd.read_csv(config.PATH_SPX)
    VIX = pd.read_csv(config.PATH_VIX)

    SPX['date'] = pd.to_datetime(SPX['caldt'] if 'caldt' in SPX.columns else SPX['date'])
    SPX['R'] = SPX['sprtrn'] if 'sprtrn' in SPX.columns else SPX['spindx'].pct_change()

    dcol = 'Date' if 'Date' in VIX.columns else 'date'
    VIX['date'] = pd.to_datetime(VIX[dcol])
    if 'vix' in VIX.columns:
        vcol = 'vix'
    else:
        vcol = [c for c in VIX.columns if 'vix' in c.lower() or c.lower() in ('close', 'adj close')][0]
    VIX['vix'] = VIX[vcol]

    m = pd.merge(SPX[['date', 'R']], VIX[['date', 'vix']], on='date').sort_values('date')
    m = m[m['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()
    m['dvix'] = m['vix'].diff()
    m = m.dropna(subset=['R', 'dvix'])

    full = m['R'].corr(m['dvix'])
    roll = m['R'].rolling(ROLL).corr(m['dvix']).dropna()

    print("\nUS price-volatility relation (S&P 500 return vs change in VIX), "
          f"{config.START_YEAR}-{config.END_YEAR}, {len(m):,} trading days")
    print(f"  full-sample correlation      : {full:+.3f}")
    print(f"  rolling {ROLL}-day correlation : "
          f"min {roll.min():+.3f} | mean {roll.mean():+.3f} | max {roll.max():+.3f} | "
          f"% positive {100 * (roll > 0).mean():.1f}%")
    print("\n(Compare: China corr = -0.24, unstable and sign-flipping; see CHN 06_diagnostics.)")


if __name__ == '__main__':
    main()
