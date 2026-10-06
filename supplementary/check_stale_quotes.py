"""check_stale_quotes.py -- verify the traded-quote (volume >= 1) filter.

For each Chinese instrument it reports, over the full raw option history:
  - the share of contract-days with zero volume (the rows the filter drops), and
  - how often the Wind 'Close (D)' is UNCHANGED from that contract's previous
    trading day, split by zero-volume vs traded days.

A stale (carried-forward) close shows up as ~100% unchanged on zero-volume days
and near-0% on traded days -- confirming the export marks no-trade days with the
previous close, which is why untraded quotes are screened out.

Data location is read from DATA_DIR (falls back to the author's path).
Run:  python3 check_stale_quotes.py
"""
import os, glob
import numpy as np, pandas as pd

DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Chinese_data'))

INSTRUMENTS = [
    ('CSI 300 ETF',   os.path.join(DATA_DIR, 'CSI_ETF_Options', 'CSI300_*.csv')),
    ('CSI 300 index', os.path.join(DATA_DIR, 'CSI300_Index',    'CSI300_*.csv')),
    ('SSE 50 ETF',    os.path.join(DATA_DIR, 'SSE50',           'SSE50_*.xlsx')),
]
COLS = ['Date', 'Trading Code', 'Close (D)', 'Volume (D)']


def scan(glob_pat):
    files = sorted(glob.glob(glob_pat))
    if not files:
        return None
    df = pd.concat([pd.read_excel(f, engine='openpyxl', usecols=COLS) for f in files],
                   ignore_index=True)                       # .csv files are xlsx content
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])                          # drop footer/junk rows
    df['vol']   = pd.to_numeric(df['Volume (D)'], errors='coerce')
    df['close'] = pd.to_numeric(df['Close (D)'], errors='coerce')
    df = df.dropna(subset=['vol', 'close']).sort_values(['Trading Code', 'Date'])
    df['prev'] = df.groupby('Trading Code', sort=False)['close'].shift(1)
    d = df.dropna(subset=['prev']).copy()
    d['unchanged'] = np.isclose(d['close'], d['prev'])
    zv, pv = d[d['vol'] == 0], d[d['vol'] > 0]
    return dict(rows=len(d),
                zero_share=100 * len(zv) / len(d),
                stale_zerovol=100 * zv['unchanged'].mean() if len(zv) else float('nan'),
                stale_traded=100 * pv['unchanged'].mean())


def main():
    print(f"DATA_DIR = {DATA_DIR}\n")
    print(f"{'instrument':<15} {'rows':>10} {'zero-vol %':>11} {'stale@0vol %':>13} {'stale@traded %':>15}")
    for name, pat in INSTRUMENTS:
        r = scan(pat)
        if r is None:
            print(f"{name:<15}  (no files found at {pat})"); continue
        print(f"{name:<15} {r['rows']:>10,} {r['zero_share']:>11.1f} "
              f"{r['stale_zerovol']:>13.1f} {r['stale_traded']:>15.1f}")


if __name__ == '__main__':
    main()
