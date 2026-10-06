import importlib as _il
import pandas as pd
from sklearn.linear_model import LinearRegression

config = _il.import_module('01_config')


def rolling_ols(df_final):
    """Rolling out-of-sample estimation of the Hull & White coefficients.

    For each month m (starting after the first WINDOW_MONTHS months), (a, b, c)
    are estimated by OLS without intercept on the previous WINDOW_MONTHS months
    and applied to month m only. This is the out-of-sample scheme of
    Hull & White (2017); with a 36-month window and data from 2010, evaluation
    covers 2013-2019.

    Returns:
        df_eval:   evaluation sample (months with coefficients) incl. a, b, c
        coeffs_df: one (a, b, c) row per evaluation month
    """
    df_final = df_final.copy()
    df_final['month'] = df_final['date'].dt.to_period('M')
    monthly_data = {m: g for m, g in df_final.groupby('month')}
    months = sorted(monthly_data.keys())

    W = config.WINDOW_MONTHS
    coeffs = {}
    for i in range(W, len(months)):
        window_months = months[i - W:i]                       # previous W months
        train_df = pd.concat([monthly_data[m] for m in window_months])

        model = LinearRegression(fit_intercept=False)
        model.fit(train_df[['X1', 'X2', 'X3']], train_df['y'])
        coeffs[months[i]] = model.coef_                        # applied to month i

    coeffs_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=['a', 'b', 'c'])
    coeffs_df.index.name = 'month'

    # Optional common evaluation start (config.EVAL_START = 'YYYY-MM'): truncate
    # the evaluation sample so different WINDOW_MONTHS are scored on the same
    # months (window-length comparisons are otherwise confounded by period).
    eval_start = getattr(config, 'EVAL_START', None)
    if eval_start:
        coeffs_df = coeffs_df[coeffs_df.index >= pd.Period(eval_start, freq='M')]
    eval_end = getattr(config, 'EVAL_END', None)
    if eval_end:
        coeffs_df = coeffs_df[coeffs_df.index <= pd.Period(eval_end, freq='M')]

    # Keep only months with estimated coefficients (out-of-sample evaluation set)
    df_eval = df_final.merge(coeffs_df, left_on='month', right_index=True)
    print(f"Rolling OLS: {len(coeffs_df)} evaluation months "
          f"({coeffs_df.index.min()} - {coeffs_df.index.max()}), "
          f"{len(df_eval):,} observations")
    return df_eval, coeffs_df
