import importlib as _il
import torch
import torch.nn as nn

config = _il.import_module('01_config')


class HedgingFNN(nn.Module):
    """3-hidden-layer feedforward network for delta hedging (Chen & Li 2023).

    Architecture: Linear(n_features->128) -> BN -> ReLU  x3
                  -> Linear(128->1) -> sign-constrained output for call/put.
    """

    def __init__(self, n_features):
        super(HedgingFNN, self).__init__()
        self.fc1   = nn.Linear(n_features, 128)
        self.bn1   = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.fc2   = nn.Linear(128, 128)
        self.bn2   = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.fc3   = nn.Linear(128, 128)
        self.bn3   = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.fc_out(x)
        # Constrain output sign: calls >= 0, puts <= 0
        if config.FLAG == config.Call:
            return torch.relu(x)
        else:
            return -torch.relu(-x)


def init_weights_xavier(m):
    """Xavier uniform init for Linear layers; biases set to zero."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def build_model(n_features):
    """Instantiate, initialise, and print a HedgingFNN."""
    model = HedgingFNN(n_features)
    model.apply(init_weights_xavier)
    print(model)
    return model
