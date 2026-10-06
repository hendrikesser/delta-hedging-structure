"""
index_loader.py -- build the merged option table for the CSI 300 INDEX options
(cash-settled CFFEX contracts, e.g. IO2006-C-3150), in the SAME column format
that each model folder's `format_data()` produces for the ETF options.

Why this exists: Wind exports Greeks for the 300 ETF options but leaves
gamma/vega/theta blank for the 300 INDEX options, so we self-compute them with
`greeks.py` (validated against Wind on the ETF: delta/gamma/vega corr 0.98-0.998).
The rest of the pipeline (filter -> pairing -> split/rolling -> model/eval) is
unchanged; it just receives this frame instead of the ETF one.

Usage inside a folder's 02_data_pipeline.build_pipeline():
    if getattr(config, 'UNDERLYING', 'ETF') == 'INDEX':
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'supplementary')))
        import index_loader
        merged_df = index_loader.build_merged(config)
    else:
        df_raw, ETF_df, VOL_df, SHIBOR_df = load_raw_data()
        merged_df = format_data(df_raw, ETF_df, VOL_df, SHIBOR_df)

Requires these extra config fields (see the config edits):
    PATH_INDEX_OPTIONS_GLOB, PATH_INDEX_UNDERLYING
and reuses the existing PATH_VOL, PATH_SHIBOR, VOL_SERIES, PRICE_COL,
START_YEAR, END_YEAR.
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'supplementary'))
import greeks   # noqa: E402  (self-computed IV + Greeks, models-root module)


def _third_friday(expiry_ym):
    """CFFEX index options expire on the 3rd Friday of the expiry month."""
    first_fri = expiry_ym + pd.to_timedelta((4 - expiry_ym.dt.weekday) % 7, unit='D')
    return first_fri + pd.Timedelta(days=14)


def build_merged(config):
    # ── 1. Load the index-option quotes (Wind .csv = xlsx content) ────────────
    files = sorted(glob.glob(config.PATH_INDEX_OPTIONS_GLOB))
    if not files:
        raise FileNotFoundError(f"No index-option files at {config.PATH_INDEX_OPTIONS_GLOB}")
    dfs = []
    for f in files:
        d = pd.read_excel(f, engine='openpyxl')
        d = d[pd.to_datetime(d['Date'], errors='coerce').notna()]
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    # ── 2. Parse the trading code  IO<YYMM>-<C/P>-<strike> ────────────────────
    tc = df['Trading Code'].astype(str).str.split('-', expand=True)
    df['cp_flag']  = tc[1]                              # 'C' / 'P'
    df['adj_flag'] = 'M'                                # index options have no adjusted series
    df['optionid'] = pd.factorize(df['Trading Code'].astype(str))[0]  # stable int id per contract
    df['date']     = pd.to_datetime(df['Date'])
    expiry_ym         = pd.to_datetime('20' + tc[0].str[2:6], format='%Y%m')
    df['exdate']      = _third_friday(expiry_ym)
    df['TTM']         = (df['exdate'] - df['date']).dt.days / 365
    df['days_to_exp'] = (df['exdate'] - df['date']).dt.days
    df['K']      = pd.to_numeric(tc[2], errors='coerce').fillna(pd.to_numeric(df['Strike'], errors='coerce'))
    df['mid']    = pd.to_numeric(df[config.PRICE_COL], errors='coerce')
    df['volume'] = pd.to_numeric(df['Volume (D)'], errors='coerce')

    # ── 3. Underlying = CSI 300 index level (000300.SH); R = daily return ─────
    U = pd.read_excel(config.PATH_INDEX_UNDERLYING)
    dcol = 'Trading Date' if 'Trading Date' in U.columns else 'Date'
    ccol = 'Closing Price' if 'Closing Price' in U.columns else 'Close (D)'
    U['date'] = pd.to_datetime(U[dcol], errors='coerce')
    U['S'] = pd.to_numeric(U[ccol].astype(str).str.replace(',', ''), errors='coerce')
    U = U[U['date'].notna()].sort_values('date')
    U['R'] = U['S'].pct_change()
    U_final = U[['date', 'S', 'R']].copy()

    # ── 4. Volatility index (same file/series as the ETF pipeline) ────────────
    VOL = pd.read_excel(config.PATH_VOL)
    VOL.columns = ['date', 'vol_300_index_options', 'vol_50etf_option',
                   'vol_500etf', 'vol_300etf_option']
    VOL = VOL[pd.to_datetime(VOL['date'], errors='coerce').notna()].copy()
    VOL['date'] = pd.to_datetime(VOL['date'])
    VIX_final = VOL[['date', config.VOL_SERIES]].rename(columns={config.VOL_SERIES: 'vix'})
    VIX_final['vix'] = pd.to_numeric(VIX_final['vix'], errors='coerce')
    VIX_final = VIX_final.dropna()

    # ── 5. Risk-free rate: SHIBOR 3M (percent -> decimal) ─────────────────────
    SH = pd.read_excel(config.PATH_SHIBOR)
    SH['date'] = pd.to_datetime(SH['Date'])
    SH['r'] = pd.to_numeric(SH['SHIBOR_3M'], errors='coerce') / 100
    SH_final = SH[['date', 'r']].dropna()

    # ── 6. Restrict years, merge ──────────────────────────────────────────────
    df      = df[df['date'].dt.year.between(config.START_YEAR, config.END_YEAR)].copy()
    U_final = U_final[U_final['date'].dt.year.between(config.START_YEAR, config.END_YEAR)]
    m = df.merge(U_final, on='date', how='inner') \
          .merge(VIX_final, on='date', how='inner') \
          .merge(SH_final, on='date', how='inner')

    # ── 7. Self-computed Greeks (Wind leaves them blank for index options) ────
    m['is_call'] = (m['cp_flag'] == config.Call)
    m = m.dropna(subset=['S', 'K', 'TTM', 'mid', 'r'])
    m = m[(m['TTM'] > 0) & (m['mid'] > 0)]
    m = greeks.add_iv_and_greeks(m)   # adds iv, delta, gamma, vega, theta
    print(f"[index_loader] {len(m):,} index-option rows, "
          f"{m['date'].min().date()} to {m['date'].max().date()}; "
          f"Greeks self-computed ({np.isfinite(m['delta']).mean()*100:.1f}% valid)")
    return m
