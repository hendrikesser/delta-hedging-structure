import importlib as _il
import torch
import torch.optim as optim

config = _il.import_module('01_config')
_fnn   = _il.import_module('03_fnn_model')


def train_model(model, X_train_t, dV_train_t, dS_train_t,
                X_val_t, dV_val_t, dS_val_t, _attempt=1):
    """Train a HedgingFNN with Adam, mini-batch SGD, and early stopping.

    Loss: mean squared one-step hedging error  E[(delta_V - delta * delta_S)^2]

    Dead-network guard: if the best validation loss never improves materially
    over the zero-hedge baseline E[delta_V^2] (sign-constrained output
    collapsed to delta = 0 at initialisation), the run is re-initialised and
    retrained (up to 3 attempts) instead of contaminating the 10-run stats.

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
            batch_x  = X_train_t[indices]
            batch_dV = dV_train_t[indices]
            batch_dS = dS_train_t[indices]

            optimizer.zero_grad()
            delta_pred    = model(batch_x)
            hedging_error = batch_dV - (delta_pred * batch_dS)
            loss          = torch.mean(hedging_error ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((dV_val_t - (model(X_val_t) * dS_val_t)) ** 2).item()

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

    # Dead-network guard (see docstring)
    zero_hedge = torch.mean(dV_val_t ** 2).item()
    if best_val_loss > 0.95 * zero_hedge and _attempt < 3:
        print(f"[guard] collapsed run detected (best val {best_val_loss:.6f} ~ "
              f"zero-hedge {zero_hedge:.6f}); re-initialising (attempt {_attempt + 1})")
        model.apply(_fnn.init_weights_xavier)
        return train_model(model, X_train_t, dV_train_t, dS_train_t,
                           X_val_t, dV_val_t, dS_val_t, _attempt=_attempt + 1)
    return model
