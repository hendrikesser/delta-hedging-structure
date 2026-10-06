import importlib as _il
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

config = _il.import_module('01_config')


# ── I. Load raw data ─────────────────────────────────────────────────────────

def load_raw_data():
    df     = pd.concat([pd.read_csv(p) for p in config.PATHS_OPTIONS], ignore_index=True)
    SPX_df = pd.read_csv(config.PATH_SPX)
    VIX_df = pd.read_csv(config.PATH_VIX)
    print(f"Shape SPX Options data: {df.shape}")
    print(f"Shape SPX Index data:   {SPX_df.shape}")
    print(f"Shape VIX data:         {VIX_df.shape}")
    return df, SPX_df, VIX_df


# ── II. Format / rename columns and merge ────────────────────────────────────

def format_data(df, SPX_df, VIX_df):
    df['K']           = df['strike_price'] / 1000          # OptionMetrics stores strikes x1000
    df['mid']         = (df['best_bid'] + df['best_offer']) / 2   # option price = closing bid/ask midpoint
    df['date']        = pd.to_datetime(df['date'])
    df['exdate']      = pd.to_datetime(df['exdate'])
    df['TTM']         = (df['exdate'] - df['date']).dt.days / 365  # time to maturity in years
    df['days_to_exp'] = (df['exdate'] - df['date']).dt.days

    if 'caldt'  in SPX_df.columns: SPX_df['date'] = pd.to_datetime(SPX_df['caldt'])
    if 'spindx' in SPX_df.columns: SPX_df['S']    = SPX_df['spindx']
    if 'sprtrn' in SPX_df.columns: SPX_df['R']    = SPX_df['sprtrn']
    SPX_final = SPX_df[['date', 'S', 'R']].copy()

    if 'Date' in VIX_df.columns:
        VIX_df['date'] = pd.to_datetime(VIX_df['Date'])
        VIX_df = VIX_df.drop(columns=['Date'])
    VIX_df['date'] = pd.to_datetime(VIX_df['date'])

    df        = df[df['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()
    SPX_final = SPX_final[SPX_final['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()
    VIX_df    = VIX_df[VIX_df['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()

    Option_SPX_df = pd.merge(df, SPX_final, on='date', how='inner')
    merged_df     = pd.merge(Option_SPX_df, VIX_df, on='date', how='inner')
    return merged_df


# ── III. Filter (Hull & White 2017 criteria) ─────────────────────────────────

def filter_data(merged_df, flag):
    filtered_df = merged_df[
        (merged_df['cp_flag']        == flag) &
        (merged_df['exercise_style'] == 'E') &
        (merged_df['volume']         >= config.MIN_VOLUME) &
        merged_df['best_bid'].notna() &
        merged_df['best_offer'].notna() &
        merged_df['impl_volatility'].notna() &
        merged_df['delta'].notna() &
        merged_df['gamma'].notna() &
        merged_df['vega'].notna() &
        merged_df['theta'].notna() &
        (merged_df['TTM'] >= 14 / 365)
    ].copy()

    if getattr(config, 'AM_ONLY', False):
        filtered_df = filtered_df[filtered_df['am_settlement'] == 1]
        print(f"AM_ONLY: {len(filtered_df):,} rows after dropping PM-settled (weeklys)")

    if flag == config.Call:
        filtered_df = filtered_df[(filtered_df['delta'] >= 0.05) & (filtered_df['delta'] <= 0.95)]
        print(f"Remaining rows: {len(filtered_df)}")
        print("Option data in paper: ~1 million rows after filtering")
    else:
        filtered_df = filtered_df[(filtered_df['delta'] <= -0.05) & (filtered_df['delta'] >= -0.95)]
        print(f"Remaining rows: {len(filtered_df)}")
        print("Option data in paper: ~1.4 million rows after filtering")

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
    test_cutoff  = pd.Timestamp(config.TEST_CUTOFF)
    df_train_val = df_final[df_final['date'] <  test_cutoff].copy()
    df_test      = df_final[df_final['date'] >= test_cutoff].copy()

    print(f"9-Year Train+Val: {df_train_val['date'].min().year} - {df_train_val['date'].max().year}")
    print(f"1-Year Test:      {df_test['date'].min().year}")
    print(f"Features:         {features}")

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

    # Chen & Li (2023) 
    print(f"Final Counts -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    if config.FLAG == config.Call:
        print("Paper targets  -> Train: 612770,       Val: 153193,        Test: 207283")
    else:
        print("Paper targets  -> Train: 878237,       Val: 219560,        Test: 299538")

    return (X_train, X_val, X_test,
            dV_train, dV_val, dV_test,
            dS_train, dS_val, dS_test,
            df_test, X_mean, X_std)


# ── Top-level builder ────────────────────────────────────────────────────────

def build_pipeline(pairing='ndg'):
    """Full data pipeline returning normalised numpy arrays ready for PyTorch.

    Args:
        pairing: 'ndg'    — strict consecutive market-day (08_run_ndgf); the
                            default, and the method behind every reported result
                 'random' — calendar-day gaps, 1-7 days (07_run_dgf)

    Returns:
        (X_train, X_val, X_test, dV_train, dV_val, dV_test,
         dS_train, dS_val, dS_test, df_test, X_mean, X_std),  features
    """
    df_raw, SPX_df, VIX_df = load_raw_data()
    merged_df   = format_data(df_raw, SPX_df, VIX_df)
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
