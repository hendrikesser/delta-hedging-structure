import importlib as _il
import torch
import torch.optim as optim

config = _il.import_module('01_config')
_fnn   = _il.import_module('03_fnn_model')


def train_model(model,
                X_train_t, dV_train_t, dS_train_t, dbs_train_t,
                X_val_t,   dV_val_t,   dS_val_t,   dbs_val_t, _attempt=1):      # include BS delta 
    """Train a ResidualHedgingNN with Adam, mini-batch SGD, and early stopping.

    Loss: mean squared one-step hedging error  E[(delta_V - delta * delta_S)^2]

    Returns: model with best validation-loss weights restored.
    """
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_val_loss      = float('inf')
    epochs_no_improve  = 0
    best_model_weights = None

    for epoch in range(config.N_EPOCHS):
        model.train()
        permutation  = torch.randperm(X_train_t.size(0))
        train_losses = []

        for i in range(0, X_train_t.size(0), config.BATCH_SIZE):
            indices  = permutation[i:i + config.BATCH_SIZE]
            batch_x   = X_train_t[indices]
            batch_dV  = dV_train_t[indices]
            batch_dS  = dS_train_t[indices]
            batch_dbs = dbs_train_t[indices]        # include BS delta

            optimizer.zero_grad()
            correction    = model(batch_x)                   # f(x): residual correction
            delta_total   = batch_dbs + correction           # delta_BS + f(x)
            hedging_error = batch_dV - (delta_total * batch_dS)
            loss          = torch.mean(hedging_error ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            correction_val = model(X_val_t)
            delta_total_val = dbs_val_t + correction_val
            val_loss = torch.mean((dV_val_t - delta_total_val * dS_val_t) ** 2).item()

        avg_train_loss = sum(train_losses) / len(train_losses)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss      = val_loss
            epochs_no_improve  = 0
            best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                break

    model.load_state_dict(best_model_weights)

    # Dead-network guard: residual collapsed to f(x)=0 means val loss ~ the
    # BS-delta baseline; harmless here (delta_total = delta_BS), so no guard
    # on level -- instead guard against NaN/inf collapse only.
    import math
    if (best_val_loss != best_val_loss or math.isinf(best_val_loss)) and _attempt < 3:
        print(f"[guard] degenerate run; re-initialising (attempt {_attempt + 1})")
        model.apply(_fnn.init_weights_xavier)
        args = (X_train_t, dV_train_t, dS_train_t, dbs_train_t,
                X_val_t, dV_val_t, dS_val_t, dbs_val_t)
        return train_model(model, *args, _attempt=_attempt + 1)
    return model
