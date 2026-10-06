"""Main runner — strict market-day pairing with Chen & Li direct FNN.

Usage:
    Edit FLAG and MODEL_NAME in 01_config.py, then:
        python 08_run_ndgf.py
"""
import importlib as _il
import random
import numpy as np
import torch


def _set_seed(seed):
    """Seed python/numpy/torch so a single run is fully reproducible (Yu)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

config              = _il.import_module('01_config')
_dp                 = _il.import_module('02_data_pipeline')
build_pipeline      = _dp.build_pipeline
_fnn                = _il.import_module('03_fnn_model')
build_model         = _fnn.build_model
_trainer            = _il.import_module('04_trainer')
train_model         = _trainer.train_model
_eval               = _il.import_module('05_evaluate')
compute_gain_ratio  = _eval.compute_gain_ratio
compute_bucket_gains = _eval.compute_bucket_gains
run_multi_evaluation = _eval.run_multi_evaluation
_tp                 = _il.import_module('06_tables_plots')
print_summary_table = _tp.print_summary_table


def main():
    _set_seed(config.SEED)                       # reproducible single run
    # ── 1. Data pipeline (strict consecutive market-day pairing) ─────────────
    arrays, features = build_pipeline(pairing='ndg')
    (X_train, X_val, X_test,
     dV_train, dV_val, dV_test,
     dS_train, dS_val, dS_test,
     df_test, X_mean, X_std) = arrays

    # ── 2. Convert to tensors ─────────────────────────────────────────────────
    X_train_t  = torch.tensor(X_train,  dtype=torch.float32)
    X_val_t    = torch.tensor(X_val,    dtype=torch.float32)
    X_test_t   = torch.tensor(X_test,   dtype=torch.float32)
    dV_train_t = torch.tensor(dV_train, dtype=torch.float32).view(-1, 1)
    dV_val_t   = torch.tensor(dV_val,   dtype=torch.float32).view(-1, 1)
    dS_train_t = torch.tensor(dS_train, dtype=torch.float32).view(-1, 1)
    dS_val_t   = torch.tensor(dS_val,   dtype=torch.float32).view(-1, 1)
    print(f"Tensor shapes - X_train: {X_train_t.shape}, dV_train: {dV_train_t.shape}")

    # ── 3. Build and train model ──────────────────────────────────────────────
    model = build_model(n_features=len(features))
    model = train_model(model, X_train_t, dV_train_t, dS_train_t,
                               X_val_t,   dV_val_t,   dS_val_t)

    # ── 4. Single-run evaluation ──────────────────────────────────────────────
    print(f"\nFeatures: {features}  |  Flag: {config.FLAG}")
    delta_bs_test_np = df_test['delta'].values
    gain_ratio, delta_nn = compute_gain_ratio(
        model, X_test_t, dV_test, dS_test, delta_bs_test_np,
        dates_test=df_test['date'].values)
    buckets, bucket_gains_run1 = compute_bucket_gains(
        delta_nn, dV_test, dS_test, delta_bs_test_np, config.FLAG)

    # ── 5. 10-run robustness evaluation ───────────────────────────────────────
    try:                                       # corrected rebalancing-frequency gains (mark-to-market)
        _eval.report_frequency(model, X_test_t, df_test)
    except Exception as _e:
        print("frequency report skipped:", _e)

    overall_gains, bucket_gains = run_multi_evaluation(
        n_features        = len(features),
        X_train_t=X_train_t,   dV_train_t=dV_train_t,   dS_train_t=dS_train_t,
        X_val_t=X_val_t,       dV_val_t=dV_val_t,       dS_val_t=dS_val_t,
        X_test_t=X_test_t,     dV_test_np=dV_test,
        dS_test_np=dS_test,    delta_bs_test_np=delta_bs_test_np,
        first_gain=gain_ratio,  first_bucket_gains=bucket_gains_run1,
        buckets=buckets,        flag=config.FLAG,
        dates_test=df_test['date'].values,
    )

    # ── 6. Summary table ──────────────────────────────────────────────────────
    print_summary_table(overall_gains, bucket_gains, buckets)


if __name__ == '__main__':
    main()
