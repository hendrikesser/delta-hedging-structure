import os

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Chinese_data'))

# Wind exports: .csv extension but xlsx content (read with openpyxl)
PATH_OPTIONS_GLOB = os.path.join(DATA_DIR, 'CSI_ETF_Options', 'CSI300_*.csv')
PATH_UNDERLYING   = os.path.join(DATA_DIR, 'Underlying', 'CSI300_ETF_Index.xlsx')   # Huatai-PB 300ETF (510300)
PATH_VOL          = os.path.join(DATA_DIR, 'Volatility', 'Volatility_300_Index_Options.xlsx')
PATH_SHIBOR       = os.path.join(DATA_DIR, 'SHIBOR', 'SHIBOR_3M.xlsx')

# CSI 300 INDEX options (CFFEX, cash-settled). Wind provides no gamma/vega/theta
# for these, so Greeks are self-computed via index_loader.py + greeks.py.
PATH_INDEX_OPTIONS_GLOB = os.path.join(DATA_DIR, 'CSI300_Index', 'CSI300_*.csv')
PATH_INDEX_UNDERLYING   = os.path.join(DATA_DIR, 'Underlying', '000300.SH-History Price-20260413.xlsx')

# Data source switch: 'ETF' = Huatai-PB 300ETF options (Wind Greeks);
#                     'INDEX' = CSI 300 index options (self-computed Greeks).
UNDERLYING = os.environ.get('UNDERLYING', 'INDEX')

# Volatility series used as the 'vix' column. Options:
#   'vol_300_index_options'  Volatility: 300 Index Options
#   'vol_300etf_option'      Volatility: Huatai-PB 300ETF Option
VOL_SERIES = 'vol_300_index_options'

# ── SSE 50 ETF options (510050, SSE-listed, Wind Greeks) ─────────────────────
# Same ETF pipeline as the 300 ETF (4th-Wed expiry, '510050'+C/P+YYMM+M/A+strike);
# underlying is the 510050 ETF price (NOT the 000016.SH index).
PATH_ETF50_OPTIONS_GLOB = os.path.join(DATA_DIR, 'SSE50', 'SSE50_*.xlsx')
PATH_ETF50_UNDERLYING   = os.path.join(DATA_DIR, 'Underlying', 'SSE50_ETF.xlsx')
if UNDERLYING == 'ETF50':
    PATH_OPTIONS_GLOB = PATH_ETF50_OPTIONS_GLOB
    PATH_UNDERLYING   = PATH_ETF50_UNDERLYING
    VOL_SERIES        = 'vol_50etf_option'


# Option price used as 'mid' (the export has no bid/ask):
#   'Close (D)' or 'Settlement Price'
PRICE_COL = 'Close (D)'

# Exclude dividend-adjusted contracts (Trading Code suffix 'A' instead of 'M'):
# their strike and per-unit price jump on the ex-dividend date, which would
# corrupt (delta_V, delta_S) pairs formed within the same optionid.
EXCLUDE_ADJUSTED = True

# Compute Black-Scholes implied volatility (the export has no IV column).
# The HW regression itself only needs the exchange-provided vega, so this is
# optional here; uses SHIBOR 3M as the risk-free rate.
COMPUTE_IV = False

# ── Option type ──────────────────────────────────────────────────────────────
Call = 'C'
Put  = 'P'
FLAG = Put if os.environ.get('FLAG','C').strip().upper() in ('P','PUT') else Call  # env-overridable: FLAG=P for puts

# ── Hedging frequency ────────────────────────────────────────────────────────
# Rebalancing interval in trading days: daily = 1, weekly = 5, monthly = 21.
# For weekly/monthly, both entry points pair each quote with the observation
# exactly k trading days ahead (the calendar-gap method applies to daily only).
HEDGE_FREQ  = 'daily'   # 'daily' | 'weekly' | 'monthly'
HEDGE_STEPS = {'daily': 1, 'weekly': 5, 'monthly': 21}

# ── Time window ──────────────────────────────────────────────────────────────
# SSE and 300ETF options were listed 23 Dec 2019; include the full sample.
START_YEAR = 2019
END_YEAR   = 2026

# ── Rolling estimation ───────────────────────────────────────────────────────
# Coefficients (a, b, c) are re-estimated each month by OLS on the previous
# WINDOW_MONTHS months and applied out-of-sample to the current month.
# Hull & White (2017) use 36 months but report that 12-60 month windows perform
# Set to 36 for the HW-baseline robustness check. Env-overridable so the ETF
# (short history) can use a smaller window for a Zhao-era cut, e.g. WINDOW_MONTHS=12.
WINDOW_MONTHS = int(os.environ.get('WINDOW_MONTHS', '36'))

# Common evaluation start ('YYYY-MM' or None). Truncates the evaluation sample
# after the rolling estimation, so different WINDOW_MONTHS settings are scored
# on the same months. Use '2022-12' (first month reachable by W=36) for the
# window sweep; None = evaluate all months with coefficients.
# Env-overridable so the OOS window can be set from the command line without
# editing this file (e.g. EVAL_START=2021-01 EVAL_END=2022-12 for a Zhao-style
# cut). Note: with WINDOW_MONTHS=36 the first reachable month is 36 months after
# each instrument's listing (index/SSE 50 reach a 2021-22 window; the CSI 300 ETF,
# listed Dec 2019, cannot -- shorten WINDOW_MONTHS for the ETF if needed).
EVAL_START = os.environ.get('EVAL_START', '2025-08')

# Last evaluation month ('YYYY-MM' or None): with EVAL_START, brackets the
# evaluation to a fixed window (e.g. a single NN test year).
EVAL_END = os.environ.get('EVAL_END') or None

# ── Volume filter ────────────────────────────────────────────────────────────
# Keep only quotes with volume >= MIN_VOLUME. NOTE: this is a deviation from
# Hull & White (2017), who apply no volume filter. It is kept here because the
# Wind export has close prices only, which are stale on days without trades.
# Set to 0 to disable.
MIN_VOLUME = 1

# ── Normalization (Hull & White 2017) ────────────────────────────────────────
# Scale each pair so the underlying price on the first day is 1: delta_V,
# delta_S and vega are divided by S, then S := 1 (delta and TTM are
# scale-free; K never enters the regression). HW's data convention; matters
# less here than in the US (the 300ETF trades in a narrow ~3.5-6 band) but is
# applied symmetrically. Set False for the earlier unnormalized results.
NORMALIZE = True
