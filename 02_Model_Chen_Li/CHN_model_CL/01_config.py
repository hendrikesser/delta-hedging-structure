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

# Volatility series used as the 'vix' feature. Options:
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
# Needed for the DNN3minusIV feature set; uses SHIBOR 3M as the risk-free rate.
COMPUTE_IV = True

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
# SSE 300ETF options were listed 23 Dec 2019; data runs to mid-2026.
# TEST_CUTOFF splits train+val (before) from test (from cutoff onward).
# Primary split: train Dec19-Dec24, test 2025-01 - 2026-04 (~16 months).
# Robustness splits (rerun with): '2024-01-01' (longer test, shorter train)
# or '2026-01-01' (max train, ~4 test months only).
START_YEAR  = 2019
END_YEAR    = 2026
TEST_CUTOFF = '2025-08-01'


# ── Feature sets (Chen & Li 2023, Table 2) ───────────────────────────────────
DNN2        = ['TTM', 'delta']
DNN2plus    = ['TTM', 'delta', 'Moneyness']
DNN3star    = ['TTM', 'delta', 'vix', 'R', 'Moneyness']
DNN3minusIV = ['TTM', 'delta', 'impl_volatility']


def get_feature_sets(flag):
    """Returns flag-dependent feature sets; DNN3/DNN3plus differ by call/put."""
    if flag == Call:
        DNN3     = ['TTM', 'delta', 'vix']
        DNN3plus = ['TTM', 'delta', 'vix', 'Moneyness']
    else:
        DNN3     = ['TTM', 'delta', 'R']
        DNN3plus = ['TTM', 'delta', 'R', 'Moneyness']
    return {
        'DNN2':        DNN2,
        'DNN2plus':    DNN2plus,
        'DNN3':        DNN3,
        'DNN3plus':    DNN3plus,
        'DNN3star':    DNN3star,
        'DNN3minusIV': DNN3minusIV,
        # DNN3-VIX (paper): VIX regardless of flag; used for puts under
        # weekly/monthly hedging, where it replaces DNN3 (Chen & Li, Sec. 4.3).
        'DNN3VIX':     ['TTM', 'delta', 'vix'],
        # DNN3-R (ours): index return regardless of flag. Mechanism test for
        # CHN calls -- if calls with R approach the puts' gain, the exploitable
        # Chinese signal is in return dynamics, not the vol-index relation.
        'DNN3R':       ['TTM', 'delta', 'R'],
    }


MODEL_NAME = 'DNN3'   # ← key into get_feature_sets() dict

# ── Training hyperparameters ─────────────────────────────────────────────────
LR         = 0.0005
N_EPOCHS   = 100  # cap only; patience-20 early stopping governs (test C)
BATCH_SIZE = 1024
PATIENCE   = 20 # Introduce patience for early stopping
NUM_RUNS   = 1    # single fixed-seed run (Yu): reproducible, no multi-run averaging

# ── Data split ───────────────────────────────────────────────────────────────
VAL_SPLIT    = 0.20
RANDOM_STATE = 42
SEED         = int(os.environ.get('SEED', '42'))   # global seed (python/numpy/torch) for a reproducible single run

# ── Volume filter ────────────────────────────────────────────────────────────────────
# Keep only quotes with volume >= MIN_VOLUME. NOTE: deviation from Chen & Li
# (2023), who follow Hull & White's filters (no volume screen) -- explains the
# smaller paired sample vs the paper (~712k vs ~973k call pairs). Kept at 1 for
# consistency with the frozen HW benchmark; set 0 for a paper-fidelity check.
MIN_VOLUME = 1

