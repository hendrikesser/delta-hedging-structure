"""cvar.py -- one-sided (tail) risk metrics for delta hedging.

The gain ratio scores hedges by MSE, which is symmetric: it penalises over- and
under-hedging equally and is driven by the whole error distribution. 

We use Expected Shortfall / CVaR of the per-period hedging error
    e = dV - delta * dS
under the short-option convention: a hedger who is short the option and long
`delta` units of the underlying has period P&L  d(Pi) = delta*dS - dV = -e, so a
LOSS is +e. CVaR_alpha is then the mean loss in the worst (1 - alpha) tail:

    VaR_alpha  = alpha-quantile of the losses e
    CVaR_alpha = mean( e | e >= VaR_alpha )

To compare a model hedge with the Black-Scholes hedge on the same footing as the
gain ratio, we report a CVaR gain ratio
    G_cvar = 1 - CVaR_alpha(model) / CVaR_alpha(BS)
positive when the model shrinks the tail loss relative to BS.

`symmetric=True` instead scores the magnitude tail (mean of the largest |e|),
a position-agnostic "worst hedging slippage" measure.
"""
import numpy as np


def var_cvar(loss, alpha=0.95, symmetric=False):
    """Return (VaR_alpha, CVaR_alpha) for a 1-D array of losses (larger = worse).

    symmetric=True scores |loss| (magnitude tail) instead of the signed upper
    tail, giving a direction-agnostic worst-slippage measure.
    """
    loss = np.asarray(loss, dtype=float)
    loss = loss[np.isfinite(loss)]
    if loss.size == 0:
        return np.nan, np.nan
    if symmetric:
        loss = np.abs(loss)
    var = np.quantile(loss, alpha)
    tail = loss[loss >= var]
    cvar = tail.mean() if tail.size else var
    return var, cvar


def cvar_gain(e_model, e_bs, alpha=0.95, symmetric=False):
    """CVaR gain ratio = 1 - CVaR_alpha(model) / CVaR_alpha(BS).

    e_model, e_bs are the signed per-period hedging errors (dV - delta*dS) of the
    model and of the Black-Scholes hedge on the SAME test trades.
    """
    _, c_m = var_cvar(e_model, alpha, symmetric)
    _, c_b = var_cvar(e_bs, alpha, symmetric)
    if not np.isfinite(c_b) or c_b == 0:
        return np.nan
    return 1.0 - c_m / c_b


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # BS error: heavy-tailed; model error: same but tails shrunk 30%
    e_bs = rng.standard_t(3, size=200_000) * 0.01
    e_nn = e_bs.copy()
    tail = np.abs(e_bs) > np.quantile(np.abs(e_bs), 0.90)
    e_nn[tail] *= 0.7
    for a in (0.95, 0.99):
        v, c = var_cvar(e_bs, a)
        print(f"BS   alpha={a}: VaR={v:.4f} CVaR={c:.4f}")
        print(f"     CVaR gain (short-option) = {cvar_gain(e_nn, e_bs, a):+.4f}"
              f"   |e|-tail = {cvar_gain(e_nn, e_bs, a, symmetric=True):+.4f}")
    # sanity: MSE gain for reference
    print(f"MSE gain = {1 - np.mean(e_nn**2)/np.mean(e_bs**2):+.4f}")
