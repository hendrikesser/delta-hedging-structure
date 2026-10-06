"""Main runner -- calendar-day pairing with the Hull & White MV delta.

Usage:
    Edit FLAG and HEDGE_FREQ in 01_config.py, then:
        python 07_run_dgf.py
"""
import importlib as _il

config                   = _il.import_module('01_config')
build_pipeline           = _il.import_module('02_data_pipeline').build_pipeline
add_regression_variables = _il.import_module('03_hw_features').add_regression_variables
rolling_ols              = _il.import_module('04_rolling_ols').rolling_ols
evaluate                 = _il.import_module('05_evaluate').evaluate


def main():
    df_final = build_pipeline(pairing='random')       # load, filter, pair
    df_final = add_regression_variables(df_final)     # y, X1, X2, X3
    df_eval, coeffs_df = rolling_ols(df_final)        # 36-month rolling OLS

    print(f"\nFlag: {config.FLAG}  |  Hedge frequency: {config.HEDGE_FREQ}")
    evaluate(df_eval, config.FLAG)                    # gain ratio + buckets


if __name__ == '__main__':
    main()
