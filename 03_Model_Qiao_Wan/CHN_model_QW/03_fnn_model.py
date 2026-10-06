import importlib as _il
import torch
import torch.nn as nn

config = _il.import_module('01_config')


class ResidualHedgingNN(nn.Module):
    """Residual FNN for delta hedging (Qiao & Wan 2024).

    Learns the *correction* f(x) to the Black-Scholes delta:
        delta_total = delta_BS + f(x)

    Architecture: [Linear(->128) -> BN -> Sigmoid] x N_LAYERS -> Linear(128->1).
    Qiao & Wan use N_LAYERS = 3 with 10 years of data and 2 with 3-year
    subsamples (relevant for the shorter Chinese sample); set in 01_config.py.
    """

    def __init__(self, n_features):
        super(ResidualHedgingNN, self).__init__()
        n_layers = getattr(config, 'N_LAYERS', 3)
        blocks, d = [], n_features
        for _ in range(n_layers):
            blocks += [nn.Linear(d, 128), nn.BatchNorm1d(128), nn.Sigmoid()]
            d = 128
        self.body   = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc_out(self.body(x))   # unconstrained residual correction


def init_weights_xavier(m):
    """Xavier uniform init for Linear layers; biases set to zero."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def build_model(n_features):
    """Instantiate, initialise, and print a ResidualHedgingNN."""
    model = ResidualHedgingNN(n_features)
    model.apply(init_weights_xavier)
    print(model)
    return model
