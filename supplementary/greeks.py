"""
 Black-Scholes implied volatility and Greeks, self-computed.
Validated against Wind's Greeks on the CSI 300 ETF options
(delta/gamma/vega corr 0.98-0.998; theta differs by a day-count convention).

INDEX
  [1] bs_price          Black-Scholes price of a European call / put (vectorised)
  [2] implied_vol       Newton inversion of the BS price -> implied volatility
  [3] bs_greeks         delta, gamma, vega, theta from BS at a given sigma
  [4] add_iv_and_greeks Convenience: take a DataFrame, return it with iv + Greek columns
  [5] __main__          tiny round-trip self-test

CONVENTIONS
  - No dividend yield (matches the existing pipeline; add a q term if ever needed).
  - vega  : per 1.00 change in volatility  (Wind matches; divide by 100 for per 1%).
  - theta : per YEAR                        (divide by 365 for per calendar day).
  - is_call: boolean (True = call, False = put); scalars or numpy arrays both work.
"""
import numpy as np
from scipy.stats import norm


# [1] ---- Black-Scholes price ------------------------------------------------
def bs_price(S, K, T, r, sigma, is_call):
    """European BS price. All inputs broadcast; is_call selects call vs put."""
    S, K, T, r, sigma = map(np.asarray, (S, K, T, r, sigma))
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(is_call, call, call - S + K * np.exp(-r * T))   # put via parity


# [2] ---- Implied volatility (Newton inversion) ------------------------------
def implied_vol(S, K, T, r, price, is_call, n_iter=100, sigma0=0.3):
    """Vectorised Newton inversion of the BS price for sigma.
    Returns NaN where the quote is below intrinsic value or does not converge."""
    S, K, T, r, price = map(np.asarray, (S, K, T, r, price))
    is_call = np.broadcast_to(is_call, S.shape)
    intrinsic = np.where(is_call,
                         np.maximum(S - K * np.exp(-r * T), 0),
                         np.maximum(K * np.exp(-r * T) - S, 0))
    valid = (price > intrinsic + 1e-8) & (T > 0) & (S > 0) & (K > 0)
    sigma = np.full(np.shape(S), sigma0, dtype=float)
    for _ in range(n_iter):
        with np.errstate(all='ignore'):
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            step = (bs_price(S, K, T, r, sigma, is_call) - price) / np.maximum(vega, 1e-10)
        sigma = np.clip(sigma - np.where(valid, step, 0.0), 1e-4, 5.0)
    converged = np.abs(bs_price(S, K, T, r, sigma, is_call) - price) <= np.maximum(1e-4, 1e-3 * price)
    return np.where(valid & converged, sigma, np.nan)


# [3] ---- Greeks -------------------------------------------------------------
def bs_greeks(S, K, T, r, sigma, is_call):
    """Return (delta, gamma, vega, theta) from BS, evaluated at sigma.
    vega is per 1.00 vol; theta is per year (see CONVENTIONS)."""
    S, K, T, r, sigma = map(np.asarray, (S, K, T, r, sigma))
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)
    delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)
    theta = np.where(is_call,
                     -S * pdf * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
                     -S * pdf * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return delta, gamma, vega, theta


# [4] ---- Convenience wrapper on a DataFrame ---------------------------------
def add_iv_and_greeks(df, s='S', k='K', t='TTM', r='r', price='mid',
                      is_call='is_call', prefix='', n_iter=100):
    """Add iv, {prefix}delta/gamma/vega/theta columns to a copy of df.
    Column names are configurable so it works on ETF or index-option frames."""
    out = df.copy()
    S, K, T, rr = out[s].values, out[k].values, out[t].values, out[r].values
    P, cp = out[price].values, out[is_call].values.astype(bool)
    iv = implied_vol(S, K, T, rr, P, cp, n_iter=n_iter)
    d, g, v, th = bs_greeks(S, K, T, rr, iv, cp)
    out['iv'] = iv
    out[prefix + 'delta'], out[prefix + 'gamma'] = d, g
    out[prefix + 'vega'],  out[prefix + 'theta'] = v, th
    return out


# [5] ---- self-test ----------------------------------------------------------
if __name__ == '__main__':
    S, K, T, r, sig = 100.0, 100.0, 1.0, 0.03, 0.20
    for cp in (True, False):
        p = bs_price(S, K, T, r, sig, cp)
        rec = implied_vol(S, K, T, r, p, cp)
        d, g, v, th = bs_greeks(S, K, T, r, sig, cp)
        print(f"{'call' if cp else 'put ':4}: price={float(p):.4f}  IV(recovered)={float(rec):.4f}  "
              f"delta={float(d):+.4f} gamma={float(g):.4f} vega={float(v):.4f} theta/yr={float(th):+.4f}")
