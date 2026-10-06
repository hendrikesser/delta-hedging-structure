# How Much Structure Does Delta Hedging Need?

Replication code for the working paper *How Much Structure Does Delta Hedging Need?
Parametric, Neural and Recurrent Corrections in the US and Chinese Option Markets*
(Esser, 2026).

The paper evaluates a hierarchy of corrections to the Black–Scholes hedging delta
across two markets, asking how much of the gain from machine learning comes from
flexibility and how much from retaining parametric structure. This repository
contains the full pipeline behind it: data filtering, the four model families,
the evaluation framework and the robustness analyses.

## Models

| Family | Idea | Folder |
|---|---|---|
| Hull & White (2017) | Parametric minimum-variance delta: a vega-weighted quadratic in the BS delta, rolling OLS | `01_Model_HW` |
| Chen & Li (2023) | Direct feed-forward network: learn the hedge ratio itself | `02_Model_Chen_Li` |
| Qiao & Wan (2024) | Residual network: learn only a correction on top of the BS delta | `03_Model_Qiao_Wan` |
| Residual GRU (this paper) | Residual network with a GRU memory channel over the contract's recent path | `04_Model_GRU` |

Each family has a **US** folder (S&P 500 options, OptionMetrics, 2010–2019) and a
**China** folder. The China side covers three instruments through one config switch,
`UNDERLYING`:

- `'ETF'` — CSI 300 ETF options (510300; Wind supplies the Greeks)
- `'INDEX'` — CSI 300 index options (CFFEX `IO`, cash-settled; Wind supplies no
  gamma/vega/theta, so Greeks are self-computed via `supplementary/index_loader.py`
  and `supplementary/greeks.py`)
- `'ETF50'` — SSE 50 ETF options (510050; Wind supplies the Greeks)

## Layout

```
01_Model_HW/        {US,CHN}_model_HW/
02_Model_Chen_Li/   {US,CHN}_model_CL/
03_Model_Qiao_Wan/  {US,CHN}_model_QW/
04_Model_GRU/       CHN_model_GRU/ + gru_model.py
supplementary/      shared modules:
      index_loader.py   CSI 300 index-option loader (UNDERLYING='INDEX')
      greeks.py         self-computed IV (Newton) and Greeks
      cvar.py           one-sided CVaR / expected-shortfall gain
      sequences.py      per-contract GRU input windows
```

**File numbering.** `01`–`08` are the core pipeline; `09`–`13` are robustness
analyses, one per paper section. `bucket_freq_table.py` is the standardised grid
runner. The Hull–White folders stop at `08`, having no learned network.

```
01_config                        run settings: instrument, call/put, features, split, filters
02_data_pipeline                 load, filter, pair next-day, split, standardise
03_fnn_model / 03_hw_features    network / HW regressors
04_trainer   / 04_rolling_ols    training loop / rolling OLS
05_evaluate                      gain ratio, pooled and bucketed
07_run_dgf / 08_run_ndgf         runners: 07 allows 1-7 calendar-day gaps, 08 pairs strictly
                                 k trading days ahead. 08 is the default and produces every
                                 reported result
bucket_freq_table                feature × frequency × side grid; overall + cvar95/cvar99
09–13                            DM tests, regime split, transaction costs, frequency, ablation
```

The `01_config.py` files double as a record of the methodological choices behind
each result — instrument, option type, rebalancing frequency, rolling window,
evaluation window and filter thresholds are all set there.

## Data — not included

**The underlying data is licensed and is not distributed here.** Reproducing the
results requires your own institutional access to:

- **US:** OptionMetrics (S&P 500 option quotes, Greeks and implied volatilities),
  the S&P 500 index level and returns, and VIX.
- **China:** Wind Financial Terminal (option chains and underlying series), plus
  SHIBOR 3M for the risk-free rate.

Expected layout:

```
data/US_data/        Option_data_SPX_06_10.csv, _11_15, _16_20
                     SPX_daily_close_return.csv, VIX_SPX.csv
data/Chinese_data/   CSI_ETF_Options/, CSI300_Index/, SSE50/,
                     Underlying/, Volatility/, SHIBOR/
```

Point the code at your own copy:

```bash
export DATA_DIR=/path/to/data/Chinese_data
```

Two notes on the Chinese data. Wind's export carries no implied-volatility column,
so IV is recovered by Newton inversion (`compute_implied_vol` in `02_data_pipeline.py`,
and `supplementary/greeks.py` for the index options). The `vix` feature is the
1-month implied volatility of the CSI 300 *index* options, not a published index.

## Running it

```bash
pip install -r requirements.txt
export DATA_DIR=/path/to/data/Chinese_data
cd 03_Model_Qiao_Wan/CHN_model_QW
python 08_run_ndgf.py
```

Each model folder follows the same numbered sequence, so execution order is legible
from the filenames.

## The pipeline (`02_data_pipeline.py`)

1. **Load** the option, underlying, volatility and SHIBOR files. For
   `UNDERLYING='INDEX'`, `index_loader.build_merged` replaces this step.
2. **Format**: flags and expiry parsed from the trading code; strike taken from the
   `Strike` column (Wind's Chinese contract name writes the strike ×1000, so the
   name serves only as a same-scale fallback for adjusted contracts); `mid = Close`;
   `TTM = (exdate − date)/365`; merged with underlying, volatility and rate series.
3. **Filter** on Hull and White (2017) criteria: option type, `volume ≥ 1` (traded
   quotes only — the China analogue of a quote screen, the export being close-only),
   non-missing Greeks and IV, `|Δ_BS| ∈ [0.05, 0.95]`, `TTM ≥ 14/365`; builds
   `Moneyness = S/K`; drops missing and duplicate `(optionid, date)` pairs.
4. **Pair** each quote with the same contract's quote exactly *k* trading days ahead
   (k = 1 for daily). Both legs are traded, so no stale gap is spanned.
5. **Split and standardise**: chronological cut at `TEST_CUTOFF`; earlier data split
   80/20 train/validation; mean and standard deviation fit on train only, then
   applied to validation and test.

Feature sets are selected by `MODEL_NAME`: `DNN2 = [TTM, delta]`, optionally plus
`vix`, `return` or `impl_volatility`. All are known at time *t*.

## The models in detail

**Hull & White MV delta.** `03_hw_features` builds `y = ΔV − Δ_BS·ΔS`, regressed
without intercept on `X_i = (vega/√T)(ΔS/S)Δ^(i−1)` for i = 1, 2, 3.
`04_rolling_ols` fits `(a, b, c)` on a trailing 36-month window and applies them to
the following month only, out of sample:
`Δ_MV = Δ_BS + (vega/(S√T))(a + bΔ + cΔ²)`.

**Chen & Li direct network.** Three hidden layers of 128 units, `Linear → BN → ReLU`,
linear output. Adam at lr 5e-4, batch 1024, early stopping with patience 20,
gradient clipping at 1.0. Loss is the mean squared one-step hedging error,
`E[(ΔV − Δ·ΔS)²]`.

**Qiao & Wan residual network.** The same 3×128 backbone with sigmoid activation
and an unconstrained output returning a correction `f(x)`; the trainer forms
`Δ_total = Δ_BS + f(x)` at lr 1e-4. This differs from Chen and Li in three ways at
once — residual target, activation and output constraint — so
`13_residual_ablation.py` isolates the residual target on its own.

**Residual GRU extension.** A GRU encodes the contract's last `SEQ_LEN` feature
vectors; its final hidden state is concatenated with the time-*t* features, and an
MLP head outputs the residual correction. It consumes a `(B, T, F)` tensor and reads
its own last step as the local vector, so the trainer and evaluation are unchanged;
only data preparation gains `supplementary/sequences.build_sequences`.

## Evaluation

```
Gain ratio   G      = 1 − MSE(model hedging error) / MSE(BS hedging error)
CVaR gain    G_cvar = 1 − CVaR_α(model) / CVaR_α(BS)        (one-sided tail risk)
```

`G > 0` means a better mean-square hedge than Black–Scholes; `G_cvar > 0` means
smaller worst-tail losses. CVaR uses the short-option loss convention at α = 95%
and 99% (`supplementary/cvar.py`). `05_evaluate` computes pooled and per-decile
gains; `bucket_freq_table.py` sweeps feature × frequency × side and reports the
overall gain alongside both CVaR gains. Two aggregations are reported: pooled,
which is dominated by turbulent days, and monthly-averaged.

## No look-ahead leakage

- Standardisation is fit on the training set only and applied to validation and test.
- The test set is a strictly later period, cut chronologically at `TEST_CUTOFF`.
- Every feature is known at time *t*; the target uses the next day's price and is
  never an input.
- Hull–White coefficients are fit on a trailing window and applied only to the
  following month.
- GRU windows contain only values up to *t* — memory, not look-ahead.

## Reproducibility

Results use a single fixed seed (`SEED = 42`, `NUM_RUNS = 1`) so that any run
reproduces exactly, rather than averaging over runs. Seed dispersion is examined
separately through `seed_sweep.py`.

## Results

Result files are not committed here; they accompany the paper, which is not yet
publicly released. The code produces every table and figure in the paper from the
scripts above.

## Related

[`dl-delta-hedging-qiao-wan`](https://github.com/hendrikesser/dl-delta-hedging-qiao-wan)
— a standalone replication of Qiao and Wan (2024), kept separate so that it remains
a faithful reproduction of the published specification.

## Citation

> Esser, H. (2026). *How Much Structure Does Delta Hedging Need? Parametric, Neural
> and Recurrent Corrections in the US and Chinese Option Markets.* Working paper,
> Peking University HSBC Business School.

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The licence covers
this code only, not the underlying market data.
