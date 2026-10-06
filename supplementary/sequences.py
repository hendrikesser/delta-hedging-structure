"""sequences.py -- turn per-contract paired rows into ordered GRU input windows.

For each paired sample at (optionid, date) we want that contract's last SEQ_LEN
feature vectors, up to and including the current date, as a (SEQ_LEN, F) window.
Windows are LEFT-padded with zeros when fewer than SEQ_LEN prior observations
exist, so the real data always ends at the last timestep (whose GRU hidden state
we read). Normalise the feature matrix BEFORE calling this, so the zero-pads sit
at the feature mean and are neutral.

The returned array is aligned to the ROW ORDER of `df`, so it drops straight into
split_and_normalize / bucket_freq_table alongside X / delta_V / delta_S without
any reordering. Each window is self-contained, so a later shuffle-split of the
samples does not corrupt the histories.
"""
import numpy as np
import pandas as pd


def build_sequences(df, features, seq_len):
    """Return an (len(df), seq_len, len(features)) array of trailing windows.

    df       : the paired dataframe (must have 'optionid' and 'date' columns)
    features : list of feature column names (already normalised)
    seq_len  : window length T
    """
    F = len(features)
    N = len(df)
    out = np.zeros((N, seq_len, F), dtype=np.float32)

    # positional index 0..N-1 in df's current order
    pos = np.arange(N)
    feats = df[features].to_numpy(dtype=np.float32)
    oid = df['optionid'].to_numpy()
    date = pd.to_datetime(df['date']).to_numpy()

    order = np.lexsort((date, oid))         # sort by optionid, then date
    oid_sorted = oid[order]
    # boundaries between contracts in the sorted order
    grp_starts = np.r_[0, np.flatnonzero(oid_sorted[1:] != oid_sorted[:-1]) + 1, len(order)]

    for a, b in zip(grp_starts[:-1], grp_starts[1:]):
        g = order[a:b]                       # positions of this contract, in time order
        arr = feats[g]                       # (n_i, F), chronological
        n = len(g)
        for j in range(n):
            lo = max(0, j - seq_len + 1)
            win = arr[lo:j + 1]              # (<=T, F)
            out[g[j], seq_len - len(win):, :] = win   # left-pad
    return out


if __name__ == '__main__':
    # tiny self-test: one contract, 3 obs, T=2 -> windows [ [0,r0], [r0,r1], [r1,r2] ]
    df = pd.DataFrame({
        'optionid': [7, 7, 7, 9, 9],
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03',
                                '2020-01-01', '2020-01-02']),
        'a': [1., 2., 3., 10., 20.],
        'b': [0.1, 0.2, 0.3, 1.0, 2.0],
    })
    s = build_sequences(df, ['a', 'b'], seq_len=2)
    print('shape', s.shape)                 # (5, 2, 2)
    print('row0 (contract 7, t0, left-pad):\n', s[0])   # [[0,0],[1,.1]]
    print('row2 (contract 7, t2):\n', s[2])             # [[2,.2],[3,.3]]
    print('row3 (contract 9, t0, left-pad):\n', s[3])   # [[0,0],[10,1]]
