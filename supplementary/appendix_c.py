"""appendix_c.py -- Tables C1/C2 and the China Diebold-Mariano test from the
SAME trained models that produce Table 2 / Appendix B.

Input: per-row test predictions written by bucket_freq_table.py with DUMP=1
(supplementary/dumps/{UNDERLYING}_{direct|residual}_{C|P}_{spec}.csv.gz).
For each instrument/network/side the best spec (highest daily gain = the
Table 2 cell) is selected automatically.

  C1: gain by BS-difficulty tercile (test days ranked by daily BS MSE), CSI 300 ETF
  DM: Diebold-Mariano, daily log(MSE_NN / MSE_BS) vs zero, Newey-West lags T^(1/3)
  C2: turnover NN/BS and net-of-cost gain at 0/10/20 bps (cost = c*|d delta|*S,
      opening trade charged at first observation) -- same as 10_transaction_costs.py

Run:  python3 appendix_c.py
"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dumps')
INSTR = [('ETF', 'CSI 300 ETF'), ('INDEX', 'CSI 300 Index'), ('ETF50', 'SSE 50 ETF')]


def load(u, net, flag):
    best = None
    for f in glob.glob(os.path.join(D, f'{u}_{net}_{flag}_*.csv.gz')):
        d = pd.read_csv(f, parse_dates=['date']).sort_values(['optionid', 'date'])
        d['em'] = d['delta_V'] - d['dm'] * d['delta_S']
        d['eb'] = d['delta_V'] - d['dbs'] * d['delta_S']
        g = 1 - (d['em'] ** 2).sum() / (d['eb'] ** 2).sum()
        spec = os.path.basename(f).split('_')[-1].replace('.csv.gz', '')
        if best is None or g > best[0]:
            best = (g, spec, d)
    return best


def nw_p(x):
    x = np.asarray(x, float); T = len(x); L = int(np.floor(T ** (1 / 3)))
    e = x - x.mean(); v = np.sum(e ** 2) / T
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * np.sum(e[l:] * e[:-l]) / T
    dm = x.mean() / np.sqrt(v / T)
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


def c1_dm():
    print("==== C1 (CSI 300 ETF, best spec per cell) + DM typical-day test ====")
    print(f"{'row':<26}{'low':>8}{'mid':>8}{'high':>8}{'pooled':>9}   DM(lnNN/BS vs 0)")
    for net in ('direct', 'residual'):
        for flag, side in (('C', 'calls'), ('P', 'puts')):
            r = load('ETF', net, flag)
            if r is None:
                print(f"  missing dump ETF/{net}/{flag}"); continue
            g, spec, d = r
            day = d.assign(sm=d['em'] ** 2, sb=d['eb'] ** 2).groupby('date').agg(sm=('sm', 'sum'), sb=('sb', 'sum'), n=('sm', 'size'))
            q = (day['sb'] / day['n']).rank(pct=True)
            lab = pd.cut(q, [0, 1 / 3, 2 / 3, 1.0], labels=['low', 'mid', 'high'])
            t = [1 - day[lab == k]['sm'].sum() / day[lab == k]['sb'].sum() for k in ('low', 'mid', 'high')]
            dm, p = nw_p(np.log((day['sm'] / day['n']) / (day['sb'] / day['n'])))
            print(f"{net + ' ' + side + ' (' + spec + ')':<26}{t[0]:>+8.2f}{t[1]:>+8.2f}{t[2]:>+8.2f}{g:>+9.4f}   DM={dm:+.3f} p={p:.4f}")


def c2():
    print("\n==== C2 (best spec per cell, daily) ====")
    print(f"{'row':<40}{'turn':>6}{'gross':>9}{'10bps':>9}{'20bps':>9}")
    for net in ('residual', 'direct'):
        for flag, side in (('C', 'calls'), ('P', 'puts')):
            for u, name in INSTR:
                r = load(u, net, flag)
                if r is None:
                    print(f"  missing dump {u}/{net}/{flag}"); continue
                g0, spec, d = r
                gr = d.groupby('optionid')
                tb = (d['dbs'] - gr['dbs'].shift(1)).abs().fillna(d['dbs'].abs())
                tn = (d['dm'] - gr['dm'].shift(1)).abs().fillna(d['dm'].abs())
                out = []
                for bp in (0, 10, 20):
                    c = bp / 1e4
                    nb = d['eb'] - c * tb * d['S']; nn = d['em'] - c * tn * d['S']
                    out.append(1 - (nn ** 2).sum() / (nb ** 2).sum())
                print(f"{name + ', ' + side + ' (' + net + ', ' + spec + ')':<40}{tn.sum() / tb.sum():>6.2f}"
                      f"{out[0]:>+9.4f}{out[1]:>+9.4f}{out[2]:>+9.4f}")


if __name__ == '__main__':
    c1_dm()
    c2()
