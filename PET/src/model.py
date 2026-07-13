import torch
import torch.nn as nn

# GRU autoencoder. The encoder compresses a 50-step window into a single hidden
# vector; the decoder rebuilds the window from it. Trained only on normal driving,
# so genuine anomalies reconstruct badly → high error = likely intrusion.
class GRUAutoencoder(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encode: run the GRU over the window and keep only the final hidden state h
        # (the compressed 'summary' of the whole sequence).
        _, h = self.encoder(x)

        # Decode: take the last layer's summary, repeat it at every timestep, and
        # reconstruct a full sequence the same shape as the input.
        context = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(context, h)
        return self.output_layer(out)


# MLP autoencoder for tabular (non-sequential) data.
# Input shape: (batch, input_size) — no time dimension.
# Named output_layer for the final linear so apply_output_noise_last_layer works here too.
class MLPAutoencoder(nn.Module):
    def __init__(self, input_size=32, hidden_sizes=(16, 8)):
        super().__init__()
        h0, h1 = hidden_sizes
        self.encoder = nn.Sequential(
            nn.Linear(input_size, h0),
            nn.ReLU(),
            nn.Linear(h0, h1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h1, h0),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(h0, input_size)

    def forward(self, x):
        return self.output_layer(self.decoder(self.encoder(x)))