import importlib as _il
import numpy as np

config = _il.import_module('01_config')


def add_regression_variables(df_final):
    """Construct the Hull & White (2017) regression variables.

    The BS hedging error is regressed on a quadratic function of delta:

        y = dV - delta_BS * dS
        y = (vega / sqrt(T)) * (dS / S) * (a + b*delta + c*delta^2) + eps

    so the regressors are X_i = (vega / sqrt(T)) * (dS / S) * delta^(i-1).
    No intercept is used (fit_intercept=False in 04_rolling_ols.py).
    """
    sqrt_T = np.sqrt(df_final['TTM'].replace(0, 1e-6))  # guard against divide-by-zero (same as 05_evaluate)

    # Target: hedging error of the plain BS delta
    df_final['y'] = df_final['delta_V'] - df_final['delta'] * df_final['delta_S']

    # Common factor: vega/sqrt(T) scaled by the index return over the period
    base_term = (df_final['vega'] / sqrt_T) * (df_final['delta_S'] / df_final['S'])

    df_final['X1'] = base_term
    df_final['X2'] = base_term * df_final['delta']
    df_final['X3'] = base_term * df_final['delta'] ** 2

    df_final = df_final.dropna(subset=['y', 'X1', 'X2', 'X3'])
    print(f"Regression sample: {len(df_final):,} observations")
    return df_final
