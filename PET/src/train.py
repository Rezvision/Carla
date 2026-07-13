import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Standard (non-private) training loop. Produces the BASELINE model, and is also
# the model used for the input- and output-perturbation experiments.
def train(model, train_dataset, val_dataset, save_path,
          epochs=30, batch_size=256, lr=1e-3, patience=5, device=None, seed=42):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed the RNG (weight init + batch shuffling) so the run is reproducible.
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Autoencoder target is the input itself, so loss = MSE(reconstruction, input).
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False)
        for (batch,) in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Keep the best model by validation loss; stop early if it hasn't improved for
        # `patience` epochs (avoids overfitting and wasted compute).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val={best_val_loss:.6f})")
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model
