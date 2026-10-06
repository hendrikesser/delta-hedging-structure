"""gru_model.py -- residual and direct GRU hedgers (models-root, shared).

Both models consume a sequence tensor `seq` of shape (batch, T, F) and derive the
local time-t feature vector internally as `seq[:, -1, :]`, so the existing trainer
and evaluation code work unchanged: they still call `model(batch_x)`, only now
`batch_x` is 3-D.

Encoder  : nn.GRU over the T-step window -> last hidden state h_T (path summary).
Decoder  : MLP head on [h_T || x_t], where x_t = seq[:, -1, :] (local state). The
           skip on x_t hands the head the time-t features directly, so the GRU can
           in principle never do worse than the feed-forward net (it recovers it
           when the GRU channel is uninformative). Mirrors Zhao's encoder
           (sequential) + decoder (local) split.

Residual : output is the correction f; delta_total = delta_BS + f  (Qiao & Wan).
Direct   : output is delta itself (Chen & Li); optional sign-aware activation.

Config knobs (with defaults): SEQ_LEN=10, GRU_HIDDEN=64, GRU_LAYERS=1,
HEAD_LAYERS=2, HEAD_WIDTH=64.
"""
import torch
import torch.nn as nn


def _head(in_dim, width, n_layers):
    blocks, d = [], in_dim
    for _ in range(max(1, n_layers)):
        blocks += [nn.Linear(d, width), nn.ReLU()]
        d = width
    blocks += [nn.Linear(d, 1)]
    return nn.Sequential(*blocks)


class _GRUHedgerBase(nn.Module):
    def __init__(self, n_features, hidden=64, gru_layers=1,
                 head_width=64, head_layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden,
                          num_layers=gru_layers, batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.head = _head(hidden + n_features, head_width, head_layers)

    def _raw(self, seq):
        # seq: (B, T, F).  GRU returns (output, h_n); h_n: (layers, B, hidden)
        _, h_n = self.gru(seq)
        h_last = h_n[-1]                 # (B, hidden) -- summary of the path
        x_t = seq[:, -1, :]              # (B, F)      -- local state at time t
        z = torch.cat([h_last, x_t], dim=1)
        return self.head(z)              # (B, 1)


class ResidualGRUHedger(_GRUHedgerBase):
    """delta_total = delta_BS + f(history, x_t); output is the unconstrained
    correction f (BS anchor added in the trainer/eval, exactly as the residual FNN)."""
    def forward(self, seq):
        return self._raw(seq)


class DirectGRUHedger(_GRUHedgerBase):
    """Output is the delta itself. sign>0 constrains to a call delta in (0,1) via
    sigmoid; sign<0 to a put delta in (-1,0); sign=0 leaves it unconstrained."""
    def __init__(self, *a, sign=0, **k):
        super().__init__(*a, **k)
        self.sign = sign

    def forward(self, seq):
        out = self._raw(seq)
        if self.sign > 0:
            return torch.sigmoid(out)
        if self.sign < 0:
            return -torch.sigmoid(out)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def build_gru_model(n_features, config=None, residual=True, sign=0, verbose=True):
    g = lambda k, d: getattr(config, k, d) if config is not None else d
    kw = dict(n_features=n_features,
              hidden=g('GRU_HIDDEN', 64),
              gru_layers=g('GRU_LAYERS', 1),
              head_width=g('HEAD_WIDTH', 64),
              head_layers=g('HEAD_LAYERS', 2),
              dropout=g('GRU_DROPOUT', 0.0))
    model = ResidualGRUHedger(**kw) if residual else DirectGRUHedger(sign=sign, **kw)
    model.apply(init_weights)
    if verbose:
        print(model)
    return model


if __name__ == '__main__':
    torch = __import__('torch')
    B, T, F = 32, 10, 5
    seq = torch.randn(B, T, F)
    m = build_gru_model(F, residual=True, verbose=False)
    out = m(seq)
    print('residual out', tuple(out.shape))          # (32, 1)
    md = build_gru_model(F, residual=False, sign=1, verbose=False)
    print('direct call out range', float(md(seq).min()), float(md(seq).max()))  # in (0,1)
