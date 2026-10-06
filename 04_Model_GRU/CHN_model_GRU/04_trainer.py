"""Trainer for the GRU hedger -- dual mode (residual or direct).

Loss is the mean squared one-step hedging error E[(delta_V - delta*delta_S)^2],
same as the FNN trainers. Pass dbs_train / dbs_val to train a RESIDUAL model
(delta_total = delta_BS + f); omit them to train a DIRECT model (model output IS
the delta). Adam, mini-batch SGD, grad-clip 1.0, patience early stopping.
"""
import importlib as _il
import math
import torch
import torch.optim as optim

config = _il.import_module('01_config')
_gru   = _il.import_module('03_gru_model')   # for the re-init guard


def train_model(model,
                X_train_t, dV_train_t, dS_train_t,
                X_val_t,   dV_val_t,   dS_val_t,
                dbs_train_t=None, dbs_val_t=None, _attempt=1):
    """Residual mode when dbs_* are given; direct mode when they are None."""
    residual = dbs_train_t is not None
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_val_loss, epochs_no_improve, best_weights = float('inf'), 0, None

    for epoch in range(config.N_EPOCHS):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        train_losses = []
        for i in range(0, X_train_t.size(0), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            bx, bdV, bdS = X_train_t[idx], dV_train_t[idx], dS_train_t[idx]
            optimizer.zero_grad()
            out = model(bx)
            delta_total = (dbs_train_t[idx] + out) if residual else out
            loss = torch.mean((bdV - delta_total * bdS) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            out_val = model(X_val_t)
            dtot_val = (dbs_val_t + out_val) if residual else out_val
            val_loss = torch.mean((dV_val_t - dtot_val * dS_val_t) ** 2).item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Train {sum(train_losses)/len(train_losses):.6f} | Val {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # NaN/inf collapse guard: re-initialise and retry up to 3 attempts.
    if (best_val_loss != best_val_loss or math.isinf(best_val_loss)) and _attempt < 3:
        print(f"[guard] degenerate run; re-initialising (attempt {_attempt + 1})")
        from gru_model import init_weights
        model.apply(init_weights)
        return train_model(model, X_train_t, dV_train_t, dS_train_t,
                           X_val_t, dV_val_t, dS_val_t,
                           dbs_train_t, dbs_val_t, _attempt=_attempt + 1)
    return model
