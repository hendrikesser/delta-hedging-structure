"""03_gru_model.py -- build_model shim for the residual/direct GRU hedger.

Exposes the same build_model(n_features) entry point the other folders use, so
the trainer and bucket grid call it identically. The actual GRU classes live in
the shared models-root module 04_Model_GRU/gru_model.py; this shim just wires in
the folder config (SEQ_LEN, GRU_HIDDEN, ... , RESIDUAL) and, for a direct GRU,
the call/put sign so the delta stays in (0,1) / (-1,0).
"""
import importlib as _il
import os as _os, sys as _sys

config = _il.import_module('01_config')

# gru_model.py sits one level up (04_Model_GRU/), shared like index_loader.
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from gru_model import build_gru_model


def build_model(n_features):
    """Instantiate the GRU hedger per config. Residual by default; for a direct
    GRU the sign is taken from config.FLAG so the output is a valid delta."""
    residual = getattr(config, 'RESIDUAL', True)
    sign = 0
    if not residual:
        sign = 1 if config.FLAG == config.Call else -1
    return build_gru_model(n_features, config=config, residual=residual,
                           sign=sign, verbose=True)
