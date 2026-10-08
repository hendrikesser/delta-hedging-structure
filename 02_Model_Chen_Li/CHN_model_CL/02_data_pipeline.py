import importlib as _il
import glob
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

config = _il.import_module('01_config')


# ── I. Load raw data ─────────────────────────────────────────────────────────

def load_raw_data():
    files = sorted(glob.glob(config.PATH_OPTIONS_GLOB))
    dfs = []
    for f in files:
        d = pd.read_excel(f, engine='openpyxl')   # .csv extension, xlsx content (Wind export)
        d = d[pd.to_datetime(d['Date'], errors='coerce').notna()]  # drop footer / repeated-header rows
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    ETF_df = pd.read_excel(config.PATH_UNDERLYING)
    ETF_df = ETF_df[pd.to_datetime(ETF_df['Date'], errors='coerce').notna()].copy()

    # Volatility file: row 0 = names, row 1 = units, data from row 2
    VOL_df = pd.read_excel(config.PATH_VOL)
    VOL_df.columns = ['date', 'vol_300_index_options', 'vol_50etf_option',
                      'vol_500etf', 'vol_300etf_option']
    VOL_df = VOL_df[pd.to_datetime(VOL_df['date'], errors='coerce').notna()].copy()

    SHIBOR_df = pd.read_excel(config.PATH_SHIBOR)

    print(f"Shape CSI300 ETF Options data: {df.shape}")
    print(f"Shape CSI300 ETF data:         {ETF_df.shape}")
    print(f"Shape Volatility data:         {VOL_df.shape}")
    print(f"Shape SHIBOR data:             {SHIBOR_df.shape}")
    return df, ETF_df, VOL_df, SHIBOR_df


# ── II. Format / rename columns and merge ────────────────────────────────────

def _fourth_wednesday(expiry_ym):
    """SSE ETF options expire on the 4th Wednesday of the expiry month."""
    first_wed = expiry_ym + pd.to_timedelta((2 - expiry_ym.dt.weekday) % 7, unit='D')
    return first_wed + pd.Timedelta(days=21)


def format_data(df, ETF_df, VOL_df, SHIBOR_df):
    # Trading Code layout: '510300' + C/P + YYMM + M/A + strike  (e.g. 510300C2203M04400)
    tc = df['Trading Code'].astype(str)
    df['cp_flag']  = tc.str[6]
    df['adj_flag'] = tc.str[11]                       # 'M' standard, 'A' dividend-adjusted
    df['optionid'] = pd.to_numeric(df['Options Symbol'], errors='coerce')

    expiry_ym         = pd.to_datetime('20' + tc.str[7:11], format='%Y%m')
    df['exdate']      = _fourth_wednesday(expiry_ym)
    df['date']        = pd.to_datetime(df['Date'])
    df['TTM']         = (df['exdate'] - df['date']).dt.days / 365
    df['days_to_exp'] = (df['exdate'] - df['date']).dt.days

    # True strike from Options Name (the Strike column is NOT dividend-adjusted
    # for 'A' contracts); fall back to the Strike column when parsing fails.
    name_k = pd.to_numeric(
        df['Options Name'].astype(str).str.extract(r'(\d+(?:\.\d+)?)\s*$')[0], errors='coerce')
    strike = pd.to_numeric(df['Strike'], errors='coerce')
    # Newer Wind exports write the strike in the Chinese Options Name as an
    # integer x1000 with no decimal (e.g. '50ETF...2800' = 2.800), which the
    # trailing-number regex misreads as 2800. The Strike column is correct in
    # every era, so trust the name-parsed strike ONLY when it is on the same
    # scale as Strike (needed for 'A' contracts, whose Strike is dividend-
    # adjusted); otherwise fall back to the Strike column.
    same_scale = name_k.notna() & name_k.between(0.2 * strike, 5 * strike)
    df['K'] = name_k.where(same_scale, strike)

    df['mid']    = pd.to_numeric(df[config.PRICE_COL], errors='coerce')  # no bid/ask in export
    df['volume'] = pd.to_numeric(df['Volume (D)'], errors='coerce')
    df = df.rename(columns={'Delta': 'delta', 'Gamma': 'gamma',
                            'Vega': 'vega', 'Theta': 'theta'})

    # Underlying: Huatai-PB 300ETF daily close; R = simple daily return
    ETF_df['date'] = pd.to_datetime(ETF_df['Date'])
    ETF_df = ETF_df.sort_values('date')
    ETF_df['S'] = pd.to_numeric(ETF_df['Close (D)'], errors='coerce')
    ETF_df['R'] = ETF_df['S'].pct_change()
    ETF_final = ETF_df[['date', 'S', 'R']].copy()

    # Volatility index (China's VIX substitute), kept under the name 'vix'
    # so feature sets and evaluation code are unchanged vs the US model
    VOL_df['date'] = pd.to_datetime(VOL_df['date'])
    VIX_final = VOL_df[['date', config.VOL_SERIES]].rename(
        columns={config.VOL_SERIES: 'vix'})
    VIX_final['vix'] = pd.to_numeric(VIX_final['vix'], errors='coerce')
    VIX_final = VIX_final.dropna()

    # Risk-free rate: SHIBOR 3M (percent -> decimal), for implied-vol inversion
    SHIBOR_df['date'] = pd.to_datetime(SHIBOR_df['Date'])
    SHIBOR_df['r']    = pd.to_numeric(SHIBOR_df['SHIBOR_3M'], errors='coerce') / 100
    SHIBOR_final = SHIBOR_df[['date', 'r']].dropna()

    df        = df[df['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()
    ETF_final = ETF_final[ETF_final['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()
    VIX_final = VIX_final[VIX_final['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()

    merged_df = pd.merge(df, ETF_final, on='date', how='inner')
    merged_df = pd.merge(merged_df, VIX_final, on='date', how='inner')
    merged_df = pd.merge(merged_df, SHIBOR_final, on='date', how='inner')
    return merged_df


# ── II-b. Black-Scholes implied volatility (no IV column in the export) ──────

def _bs_price(S, K, T, r, sig, is_call):
    d1 = (np.log(S / K) + (r + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(is_call, call, call - S + K * np.exp(-r * T))


def compute_implied_vol(df, flag, n_iter=100):
    """Vectorised Newton inversion of the BS price for European options.
    Returns NaN where the quote is below intrinsic value or not converged."""
    S, K = df['S'].values, df['K'].values
    T, r = df['TTM'].values, df['r'].values
    P    = df['mid'].values
    is_call = np.full(len(df), flag == config.Call)

    intrinsic = np.where(is_call,
                         np.maximum(S - K * np.exp(-r * T), 0),
                         np.maximum(K * np.exp(-r * T) - S, 0))
    valid = (P > intrinsic + 1e-8) & (T > 0) & (S > 0) & (K > 0)

    sig = np.full(len(df), 0.3)
    for _ in range(n_iter):
        with np.errstate(all='ignore'):
            price = _bs_price(S, K, T, r, sig, is_call)
            d1    = (np.log(S / K) + (r + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
            vega  = S * norm.pdf(d1) * np.sqrt(T)
            step  = (price - P) / np.maximum(vega, 1e-10)
        sig = np.clip(sig - np.where(valid, step, 0.0), 1e-4, 5.0)

    with np.errstate(all='ignore'):
        price = _bs_price(S, K, T, r, sig, is_call)
    converged = np.abs(price - P) <= np.maximum(1e-4, 1e-3 * P)
    iv = np.where(valid & converged, sig, np.nan)
    print(f"Implied vol: {np.isfinite(iv).sum():,} / {len(iv):,} rows inverted")
    return iv


# ── III. Filter (Hull & White 2017 criteria) ─────────────────────────────────

def filter_data(merged_df, flag):
    # SSE ETF options are European; no exercise-style filter needed.
    if config.EXCLUDE_ADJUSTED:
        merged_df = merged_df[merged_df['adj_flag'] == 'M']

    filtered_df = merged_df[
        (merged_df['cp_flag'] == flag) &
        (merged_df['volume']  >= config.MIN_VOLUME) &
        merged_df['mid'].notna() &
        merged_df['delta'].notna() &
        merged_df['gamma'].notna() &
        merged_df['vega'].notna() &
        merged_df['theta'].notna() &
        (merged_df['TTM'] >= 14 / 365)
    ].copy()

    if flag == config.Call:
        filtered_df = filtered_df[(filtered_df['delta'] >= 0.05) & (filtered_df['delta'] <= 0.95)]
    else:
        filtered_df = filtered_df[(filtered_df['delta'] <= -0.05) & (filtered_df['delta'] >= -0.95)]
    print(f"Remaining rows: {len(filtered_df)}")

    if config.COMPUTE_IV:
        filtered_df['impl_volatility'] = compute_implied_vol(filtered_df, flag)
        filtered_df = filtered_df[filtered_df['impl_volatility'].notna()]

    filtered_df['Moneyness'] = filtered_df['S'] / filtered_df['K']

    # Data-integrity guards: drop rows with missing optionid (would corrupt the
    # groupby-based pairing) and duplicate (optionid, date) quotes.
    filtered_df = filtered_df.dropna(subset=['optionid'])
    filtered_df['optionid'] = filtered_df['optionid'].astype('int64')
    filtered_df = filtered_df.drop_duplicates(subset=['optionid', 'date'], keep='first')
    return filtered_df


# ── IV-a. Pairing — calendar-day method (07_run_dgf) ─────────────────────

def pair_options_random(filtered_df):
    """Pairs each option row with its next observation within the same contract.
    Keeps 1-7 calendar-day gaps to accommodate weekends and holidays."""
    df_clean = filtered_df.sort_values(['optionid', 'date']).copy()

    grouped               = df_clean.groupby('optionid')
    df_clean['date_next'] = grouped['date'].shift(-1)
    df_clean['mid_next']  = grouped['mid'].shift(-1)
    df_clean['S_next']    = grouped['S'].shift(-1)

    df_clean['day_gap'] = (df_clean['date_next'] - df_clean['date']).dt.days
    df_clean['delta_V'] = df_clean['mid_next'] - df_clean['mid']
    df_clean['delta_S'] = df_clean['S_next']   - df_clean['S']

    df_final = df_clean[df_clean['day_gap'].between(1, 7)].copy()
    df_final = df_final.dropna(subset=['delta_V', 'delta_S'])
    df_final = df_final.sort_values('date')

    print(f"Paired dataset shape: {df_final.shape}")
    print(df_final[['date', 'delta_V', 'delta_S', 'day_gap']].head())
    return df_final


# ── IV-b. Pairing — strict market-day method (08_run_ndgf) ───────────────────
# Preferable model
def pair_options_ndg(filtered_df):
    """Pairs each option quote with the observation exactly k trading days ahead
    (k = 1 daily, 5 weekly, 21 monthly; set via config.HEDGE_FREQ). The trading
    calendar is derived from the dates observed in the data, so weekends and
    holidays are handled implicitly. For k = 1 this is identical to pairing
    strictly consecutive trading days.

    NOTE (frequency analysis): for k > 1 this pairs each quote with the price
    k days LATER (a single k-day hold); that k-day P&L is gamma/theta dominated.
    For rebalancing-frequency GAINS use 05_evaluate.report_frequency (or
    13_freq_stale_delta.py), which rebalances every k days but marks to market
    daily. See project log."""
    k = config.HEDGE_STEPS[config.HEDGE_FREQ]
    df_clean = filtered_df.sort_values(['optionid', 'date']).copy()

    # Map each date to the date exactly k trading days ahead (NaT near sample end)
    unique_dates = sorted(df_clean['date'].unique())
    ahead = {d: (unique_dates[i + k] if i + k < len(unique_dates) else pd.NaT)
             for i, d in enumerate(unique_dates)}
    df_clean['date_target'] = df_clean['date'].map(ahead)

    # Self-merge: keep only quotes with an observation exactly k trading days later
    nxt = df_clean[['optionid', 'date', 'mid', 'S']].rename(
        columns={'date': 'date_target', 'mid': 'mid_next', 'S': 'S_next'})
    df_final = pd.merge(df_clean, nxt, on=['optionid', 'date_target'], how='inner')

    df_final['delta_V'] = df_final['mid_next'] - df_final['mid']
    df_final['delta_S'] = df_final['S_next']   - df_final['S']
    df_final = df_final.dropna(subset=['delta_V', 'delta_S']).sort_values('date')

    print(f"Paired dataset shape: {df_final.shape}  "
          f"({config.HEDGE_FREQ} hedging, k = {k} trading day(s))")
    print(df_final[['date', 'delta_V', 'delta_S']].head())
    return df_final


# ── V. Split and normalise ────────────────────────────────────────────────────

def split_and_normalize(df_final, features):
    test_cutoff = pd.Timestamp(config.TEST_CUTOFF)

    # Use the date that corresponds to the end of delta_V / delta_S.
    target_col = (
        "date_target" if "date_target" in df_final.columns
        else "date_next"
    )

    start_date = pd.to_datetime(df_final["date"])
    target_date = pd.to_datetime(df_final[target_col])

    # Keep training labels entirely before the test period.
    train_mask = (start_date < test_cutoff) & (target_date < test_cutoff)

    # Test pairs start in the test period and end there or later.
    test_mask = (start_date >= test_cutoff) & (target_date >= test_cutoff)

    df_train_val = df_final.loc[train_mask].copy()
    df_test = df_final.loc[test_mask].copy()

    X_trainval_raw = df_train_val[features].values
    dV_trainval    = df_train_val['delta_V'].values
    dS_trainval    = df_train_val['delta_S'].values

    X_test_raw = df_test[features].values
    dV_test    = df_test['delta_V'].values
    dS_test    = df_test['delta_S'].values

    X_train_raw, X_val_raw, dV_train, dV_val, dS_train, dS_val = train_test_split(
        X_trainval_raw, dV_trainval, dS_trainval,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )

    X_mean  = X_train_raw.mean(axis=0)
    X_std   = X_train_raw.std(axis=0)
    X_train = (X_train_raw - X_mean) / X_std
    X_val   = (X_val_raw   - X_mean) / X_std
    X_test  = (X_test_raw  - X_mean) / X_std

    print(f"Final Counts -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (X_train, X_val, X_test,
            dV_train, dV_val, dV_test,
            dS_train, dS_val, dS_test,
            df_test, X_mean, X_std)


# ── Top-level builder ────────────────────────────────────────────────────────

def build_pipeline(pairing='random'):
    """Full data pipeline returning normalised numpy arrays ready for PyTorch.

    Args:
        pairing: 'random' — calendar-day gaps (07_run_dgf)
                 'ndg'    — strict consecutive market-day (08_run_ndgf)

    Returns:
        (X_train, X_val, X_test, dV_train, dV_val, dV_test,
         dS_train, dS_val, dS_test, df_test, X_mean, X_std),  features
    """
    if getattr(config, 'UNDERLYING', 'ETF') == 'INDEX':
        import sys as _sys, os as _os
        _sys.path.insert(
        0,
        _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'supplementary')
        )
        )
        import index_loader
        merged_df = index_loader.build_merged(config)
    else:
        df_raw, ETF_df, VOL_df, SHIBOR_df = load_raw_data()
        merged_df   = format_data(df_raw, ETF_df, VOL_df, SHIBOR_df)
    filtered_df = filter_data(merged_df, config.FLAG)

    k = config.HEDGE_STEPS[config.HEDGE_FREQ]
    if pairing == 'random' and k == 1:
        df_final = pair_options_random(filtered_df)
    elif pairing in ('random', 'ndg'):
        if pairing == 'random' and k > 1:
            print(f"[{config.HEDGE_FREQ} hedging] calendar-gap pairing applies to "
                  f"daily only; using exact {k}-trading-day pairing instead.")
        df_final = pair_options_ndg(filtered_df)
    else:
        raise ValueError(f"Unknown pairing '{pairing}'. Use 'random' or 'ndg'.")

    features = config.get_feature_sets(config.FLAG)[config.MODEL_NAME]
    arrays   = split_and_normalize(df_final, features)
    return arrays, features
