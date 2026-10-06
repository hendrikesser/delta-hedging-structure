import os

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'US_data'))

PATHS_OPTIONS = [
    os.path.join(DATA_DIR, 'Option_data_SPX_06_10.csv'),
    os.path.join(DATA_DIR, 'Option_data_SPX_11_15.csv'),
    os.path.join(DATA_DIR, 'Option_data_SPX_16_20.csv'),
]
PATH_SPX = os.path.join(DATA_DIR, 'SPX_daily_close_return.csv')
PATH_VIX = os.path.join(DATA_DIR, 'VIX_SPX.csv')

# ── Option type ──────────────────────────────────────────────────────────────
Call = 'C'
Put  = 'P'
FLAG = Put  # ← change to Call or Put before running

# ── Hedging frequency ────────────────────────────────────────────────────────
# Rebalancing interval in trading days: daily = 1, weekly = 5, monthly = 21.
# For weekly/monthly, both entry points pair each quote with the observation
# exactly k trading days ahead (the calendar-gap method applies to daily only).
HEDGE_FREQ  = 'monthly'   # 'daily' | 'weekly' | 'monthly'
HEDGE_STEPS = {'daily': 1, 'weekly': 5, 'monthly': 21}

# ── Time window ──────────────────────────────────────────────────────────────
START_YEAR = 2010
END_YEAR   = 2019

# ── Rolling estimation ───────────────────────────────────────────────────────
# Coefficients (a, b, c) are re-estimated each month by OLS on the previous
# WINDOW_MONTHS months and applied out-of-sample to the current month
# (Hull & White 2017 use a 36-month rolling window).
WINDOW_MONTHS = 36   # frozen benchmark choice (best in both markets, paper-faithful)

# Common evaluation start ('YYYY-MM' or None). Truncates the evaluation sample
# after the rolling estimation, so different WINDOW_MONTHS settings are scored
# on the same months. None = evaluate all months with coefficients.
EVAL_START = '2019-01'

# Last evaluation month ('YYYY-MM' or None): with EVAL_START, brackets the
# evaluation to a fixed window (e.g. a single NN test year).
EVAL_END =  None

# ── Normalization (Hull & White 2017) ────────────────────────────────────────
# Scale each pair so the underlying price on the first day is 1: delta_V,
# delta_S and vega are divided by S, then S := 1 (delta and TTM are
# scale-free; K never enters the regression). This is HW's data convention;
# it equal-weights the pooled SSE across time, whereas raw prices over-weight
# high-index periods (SPX roughly tripled over 2010-2019).
# Set False to reproduce the earlier unnormalized results.
NORMALIZE = True

# ── Volume filter ────────────────────────────────────────────────────────────
# Keep only quotes with volume >= MIN_VOLUME. NOTE: this is a deviation from
# Hull & White (2017), who apply no volume filter (they keep all quotes with
# valid bid/ask/IV/Greeks). Applied symmetrically in the CHN model, where it is
# needed because close prices are stale without trades. Set to 0 to disable.
MIN_VOLUME = 1

# ── Settlement filter ────────────────────────────────────────────────────────
# Keep only AM-settled options (standard monthly SPX series) and drop the
# PM-settled weeklys/EOM series (SPXW etc.), which grew from ~12% of quotes in
# 2006-10 to ~71% in 2016-20. The papers' sample sizes (Chen & Li ~973k call
# pairs; HW ~1.3M quotes 2004-2015) are only reconcilable with a mostly-AM
# universe. Use 10_sample_matrix.py (Chen_Li folder) to see the counts.
AM_ONLY = False
