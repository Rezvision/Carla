import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.timing import TrainTiming

# Opacus imports — will raise ImportError with a clear message if not installed
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except ImportError as e:
    raise ImportError(
        "Opacus is required for DP-SGD training. Install it with: pip install opacus"
    ) from e


def train_dpsgd(model, train_dataset, val_dataset, save_path,
                noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-8,
                epochs=30, batch_size=256, lr=1e-3, patience=5,
                max_physical_batch_size=64, seed=42, device=None):
    """
    Train model with DP-SGD via Opacus.

    Returns:
        model           : best model (lowest val MSE)
        epsilon_at_best : ε spent AT the epoch the returned (checkpointed) model was
                          saved — this is the correct ε to report/plot against the
                          model's utility (Ponomareva et al. 2023, §5.2), since the
                          accountant's final-epoch ε does not describe an
                          earlier-checkpointed model. Note: selecting the checkpoint
                          by validation loss is itself a data-dependent choice whose
                          own privacy cost is not captured by the accountant
                          (Papernot & Steinke 2022) — see METHODS.md §4.
        epsilon_final   : ε after the LAST epoch actually trained (informational only;
                          do not plot this against the returned model's utility).
        timing          : TrainTiming(train_seconds, epochs_ran) — wall-clock cost
                          (TIMING_TASK.md part B). DP-SGD is expected to be far slower
                          per epoch than train(): Opacus computes PER-SAMPLE gradients
                          and splits the logical batch into smaller physical batches.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed for reproducibility (DP-SGD already injects its own calibrated noise).
    torch.manual_seed(seed)

    # Opacus requires its own GRU implementation — replace nn.GRU automatically
    # DP-SGD needs PER-SAMPLE gradients. Opacus can't get those from a stock nn.GRU,
    # so ModuleValidator.fix swaps in a DP-compatible GRU implementation.
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # The heart of gradient perturbation (DP-SGD): Opacus clips each sample's gradient
    # to max_grad_norm, then adds Gaussian noise scaled by noise_multiplier.
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    best_val_loss = float("inf")
    epsilon_at_best = None
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Same measurement as train() (sync-bracketed wall clock over the whole loop) so the
    # DP-SGD slowdown factor is a like-for-like ratio, not an artefact of timing method.
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    _t0 = time.perf_counter()
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        train_loss = 0.0

        # Per-sample gradients are memory-hungry; this splits the logical batch into
        # smaller physical batches transparently so it fits in memory.
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_loader:
            for (batch,) in memory_safe_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_loss /= len(val_dataset)

        # Ask the privacy accountant how much budget ε has been spent so far (for this δ).
        epsilon = privacy_engine.get_epsilon(delta)
        print(
            f"Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  "
            f"val={val_loss:.6f}  ε={epsilon:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epsilon_at_best = epsilon
            epochs_no_improve = 0
            # Opacus wraps the model; `._module` is the underlying network we actually save.
            torch.save(model._module.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val={best_val_loss:.6f})")
                break

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    timing = TrainTiming(train_seconds=time.perf_counter() - _t0, epochs_ran=epochs_ran)
    print(f"Trained {epochs_ran} epoch(s) in {timing.train_seconds:.1f}s "
          f"({timing.sec_per_epoch:.2f}s/epoch, DP-SGD/Opacus per-sample gradients)")

    epsilon_final = privacy_engine.get_epsilon(delta)
    model._module.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model._module, epsilon_at_best, epsilon_final, timing
