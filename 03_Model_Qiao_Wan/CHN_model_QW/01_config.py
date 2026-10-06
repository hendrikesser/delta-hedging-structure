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

# SSE 50 ETF options (510050, SSE-listed, Wind Greeks). Same ETF pipeline as the
# 300 ETF (4th-Wednesday expiry, '510050'+C/P+YYMM+M/A+strike code); underlying
# is the 510050 ETF price (NOT the 000016.SH index).
PATH_ETF50_OPTIONS_GLOB = os.path.join(DATA_DIR, 'SSE50', 'SSE50_*.xlsx')
PATH_ETF50_UNDERLYING   = os.path.join(DATA_DIR, 'Underlying', 'SSE50_ETF.xlsx')

# Data source switch: 'ETF'   = Huatai-PB 300ETF options (Wind Greeks);
#                     'INDEX' = CSI 300 index options (self-computed Greeks);
#                     'ETF50' = SSE 50 ETF options (Wind Greeks).
UNDERLYING = os.environ.get('UNDERLYING', 'ETF')

# Volatility series used as the 'vix' feature. Options:
#   'vol_300_index_options'  Volatility: 300 Index Options (IV_1M)
#   'vol_300etf_option'      Volatility: Huatai-PB 300ETF Option
#   'vol_50etf_option'       Volatility: 50ETF Option (used for ETF50)
VOL_SERIES = 'vol_300_index_options'

# ETF50 uses the same format_data (ETF) branch, just with 50ETF paths + vol.
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
# Needed for the Fea4-Fea7 feature sets; uses SHIBOR 3M as the risk-free rate.
COMPUTE_IV = True

# ── Option type ──────────────────────────────────────────────────────────────
Call = 'C'
Put  = 'P'
FLAG = Put if os.environ.get('FLAG','C').strip().upper() in ('P','PUT') else Call  # env-overridable: FLAG=P for puts

# ── Filters ──────────────────────────────────────────────────────────────────
# Keep only traded quotes (volume >= MIN_VOLUME). Qiao & Wan explicitly exclude
# untraded quotes, so volume >= 1 is paper-faithful here (not a deviation, as it
# is in Chen & Li). Kept configurable for parity across the model folders.
MIN_VOLUME = 1

# ── Pairing (07_run_dgf only) ────────────────────────────────────────────────
# Max calendar days between a quote and its next quote when forming
# (delta_V, delta_S) pairs. Set to None for uncapped next-quote pairing.
# Ignored by 08_run_ndgf.
MAX_DAY_GAP = 4

# ── Hedging frequency ────────────────────────────────────────────────────────
# Rebalancing interval in trading days: daily = 1, weekly = 5, monthly = 21.
# For weekly/monthly, both entry points pair each quote with the observation
# exactly k trading days ahead (the calendar-gap method applies to daily only).
HEDGE_FREQ  = 'daily'   # 'daily' | 'weekly' | 'monthly'
HEDGE_STEPS = {'daily': 1, 'weekly': 5, 'monthly': 21}

# ── Time window ──────────────────────────────────────────────────────────────
# SSE 300ETF options were listed in Dec 2019: ~5 years train/val, 1 year test.
# Data runs 23 Dec 2019 - mid 2026. Primary split as in CHN_model_CL:
# train Dec19-Dec24, test 2025-01 - 2026-04 (~16 months).
START_YEAR  = 2019
END_YEAR    = 2026
TEST_CUTOFF = '2025-08-01'

# ── Feature sets (Qiao & Wan 2024, Table 2) ──────────────────────────────────
Fea2 = ['TTM', 'delta']
Fea3 = ['TTM', 'delta', 'Moneyness']
Fea4 = ['TTM', 'delta', 'Moneyness', 'impl_volatility']
Fea5 = ['TTM', 'delta', 'Moneyness', 'impl_volatility', 'theta']
Fea6 = ['TTM', 'delta', 'Moneyness', 'impl_volatility', 'theta', 'vega']
Fea7 = ['TTM', 'delta', 'Moneyness', 'impl_volatility', 'theta', 'vega', 'gamma']


def get_feature_sets(flag):
    """Returns all feature sets; Fea3_CL is flag-dependent (VIX for calls, R for puts)."""
    if flag == Call:
        Fea3_CL = ['TTM', 'delta', 'vix']
    else:
        Fea3_CL = ['TTM', 'delta', 'R']
    return {
        'Fea2':    Fea2,
        'Fea3':    Fea3,
        'Fea3_CL': Fea3_CL,
        'Fea4':    Fea4,
        'Fea5':    Fea5,
        'Fea6':    Fea6,
        'Fea7':    Fea7,
    }


MODEL_NAME = 'Fea3_CL'   # ← key into get_feature_sets() dict

# ── Training hyperparameters ─────────────────────────────────────────────────
# Hidden layers: 3 (paper, 10y data); 2 for small samples (paper's 3y setup,
# candidate for the Chinese sample).
N_LAYERS   = 3

LR         = 0.0001   # Qiao & Wan use 0.0001 (vs 0.0005 in Chen & Li)
N_EPOCHS   = 100  # cap only; patience-20 early stopping governs (parity with Chen & Li)
BATCH_SIZE = 1024
PATIENCE   = 20 # Introduce patience for early stopping
NUM_RUNS   = 1    # single fixed-seed run (Yu): reproducible, no multi-run averaging

# ── Data split ───────────────────────────────────────────────────────────────
VAL_SPLIT    = 0.20
RANDOM_STATE = 42
SEED = int(os.environ.get('SEED', '42'))
