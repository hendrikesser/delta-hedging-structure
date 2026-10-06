"""Sample-size matrix: which filter combination reproduces the papers' samples?

Builds the paired dataset (no training) for every combination of
  MIN_VOLUME in {0, 1}  x  AM_ONLY in {False, True}  x  pairing in {ndg, dgf}
and prints pair counts against the papers' reference totals:
  Chen & Li calls: 973,246 total (612,770 / 153,193 / 207,283 train/val/test)
  Chen & Li puts:  1,397,335 total (878,237 / 219,560 / 299,538)
  Qiao & Wan calls: 838,706 (test 174,303); puts: 1,234,959 (test 258,968)

Loads the raw data ONCE, then re-filters in memory. Runtime is dominated by
the initial CSV load (~1-2 min). Uses the FLAG from 01_config.py.

Usage: python 10_sample_matrix.py
"""
import importlib as _il
import pandas as pd

config = _il.import_module('01_config')
_dp    = _il.import_module('02_data_pipeline')

PAPER = {
    config.Call: {'CL_total': 973246, 'CL_test': 207283, 'QW_total': 838706, 'QW_test': 174303},
    config.Put:  {'CL_total': 1397335, 'CL_test': 299538, 'QW_total': 1234959, 'QW_test': 258968},
}


def main():
    flag = config.FLAG
    ref = PAPER[flag]
    print(f"FLAG = {flag} | paper targets: CL {ref['CL_total']:,} (test {ref['CL_test']:,}), "
          f"QW {ref['QW_total']:,} (test {ref['QW_test']:,})\n")

    df_raw, SPX_df, VIX_df = _dp.load_raw_data()
    merged = _dp.format_data(df_raw, SPX_df, VIX_df)
    cutoff = pd.Timestamp(config.TEST_CUTOFF)

    print(f"\n{'MIN_VOL':>7} {'AM_ONLY':>8} {'pairing':>8} {'pairs':>12} {'test':>10} "
          f"{'vs CL':>8} {'vs QW':>8}")
    print('-' * 68)
    for min_vol in [0, 1]:
        for am_only in [False, True]:
            config.MIN_VOLUME = min_vol
            config.AM_ONLY = am_only
            filt = _dp.filter_data(merged, flag)
            for pairing, fn in [('ndg', _dp.pair_options_ndg),
                                ('dgf<=7', _dp.pair_options_random)]:
                pairs = fn(filt)
                n, n_test = len(pairs), (pairs['date'] >= cutoff).sum()
                print(f"{min_vol:>7} {str(am_only):>8} {pairing:>8} {n:>12,} {n_test:>10,} "
                      f"{n/ref['CL_total']-1:>+7.1%} {n/ref['QW_total']-1:>+7.1%}")

    print("\nInterpretation: the combination closest to 0% 'vs CL' is Chen & Li's "
          "implicit sample; closest to 0% 'vs QW' is Qiao & Wan's.")


if __name__ == '__main__':
    main()
