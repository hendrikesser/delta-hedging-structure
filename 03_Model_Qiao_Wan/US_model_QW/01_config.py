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
FLAG = Call  # ← change to Call or Put before running

# ── Filters ──────────────────────────────────────────────────────────────────
# Keep only traded quotes (volume >= MIN_VOLUME). Qiao & Wan explicitly exclude
# untraded quotes, so volume >= 1 is paper-faithful here (not a deviation, as it
# is in Chen & Li). Kept configurable for parity across the model folders.
MIN_VOLUME = 1

# ── Pairing (07_run_dgf only) ────────────────────────────────────────────────
# Max calendar days between a quote and its next quote when forming
# (delta_V, delta_S) pairs. A cap of 4 reproduces the paper's sample size
# almost exactly (836,160 vs 838,706 call observations, -0.3%).
# Set to None for uncapped next-quote pairing. Ignored by 08_run_ndgf.
MAX_DAY_GAP = 4

# ── Hedging frequency ────────────────────────────────────────────────────────
# Rebalancing interval in trading days: daily = 1, weekly = 5, monthly = 21.
# For weekly/monthly, both entry points pair each quote with the observation
# exactly k trading days ahead (the calendar-gap method applies to daily only).
HEDGE_FREQ  = 'monthly'   # 'daily' | 'weekly' | 'monthly'
HEDGE_STEPS = {'daily': 1, 'weekly': 5, 'monthly': 21}

# ── Time window ──────────────────────────────────────────────────────────────
START_YEAR  = 2010
END_YEAR    = 2019
TEST_CUTOFF = '2019-01-01'

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


MODEL_NAME = 'Fea3'   # ← key into get_feature_sets() dict

# ── Training hyperparameters ─────────────────────────────────────────────────
# Hidden layers: 3 (paper, 10y data); 2 for small samples (paper's 3y setup,
# candidate for the Chinese sample).
N_LAYERS   = 3

LR         = 0.0001   # Qiao & Wan use 0.0001 (vs 0.0005 in Chen & Li)
N_EPOCHS   = 100  # cap only; patience-20 early stopping governs (parity with Chen & Li)
BATCH_SIZE = 1024
PATIENCE   = 20 # Introduce patience for early stopping
NUM_RUNS   = 10 

# ── Data split ───────────────────────────────────────────────────────────────
VAL_SPLIT    = 0.20
RANDOM_STATE = 42
