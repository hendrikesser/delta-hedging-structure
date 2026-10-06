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
FLAG = Call # ← change to Call or Put before running

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
NUM_RUNS   = 10

# ── Data split ───────────────────────────────────────────────────────────────
VAL_SPLIT    = 0.20
RANDOM_STATE = 42

# ── Volume filter ────────────────────────────────────────────────────────────────────
# Keep only quotes with volume >= MIN_VOLUME. NOTE: deviation from Chen & Li
# (2023), who follow Hull & White's filters (no volume screen) -- explains the
# smaller paired sample vs the paper (~712k vs ~973k call pairs). Kept at 1 for
# consistency with the frozen HW benchmark; set 0 for a paper-fidelity check.
MIN_VOLUME = 1

# ── Settlement filter ────────────────────────────────────────────────────────
# Keep only AM-settled options (standard monthly SPX series) and drop the
# PM-settled weeklys/EOM series (SPXW etc.), which grew from ~12% of quotes in
# 2006-10 to ~71% in 2016-20. The papers' sample sizes (Chen & Li ~973k call
# pairs; HW ~1.3M quotes 2004-2015) are only reconcilable with a mostly-AM
# universe. Use 10_sample_matrix.py (Chen_Li folder) to see the counts.
AM_ONLY = False

