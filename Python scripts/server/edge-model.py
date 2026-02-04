# edge_model.py - GRU/LSTM Autoencoder for Edge Devices (Raspberry Pi)
"""
Lightweight autoencoder designed for Raspberry Pi deployment.
Uses GRU (default) or LSTM for sequence modeling with minimal memory footprint.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import io


class GRUEncoder(nn.Module):
    """GRU-based encoder for sequence compression"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Latent projection
        self.latent_proj = nn.Linear(
            hidden_dim * self.num_directions, 
            latent_dim
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            latent: Latent representation (batch, latent_dim)
            hidden: Final hidden state for decoder initialization
        """
        # Project input
        x = self.input_proj(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # GRU encoding
        output, hidden = self.gru(x)
        
        # Use final hidden state for latent representation
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]
        
        # Project to latent space
        latent = self.latent_proj(hidden_cat)
        
        return latent, hidden


class GRUDecoder(nn.Module):
    """GRU-based decoder for sequence reconstruction"""
    
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        seq_length: int = 20
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # Latent to hidden projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        latent: torch.Tensor, 
        seq_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            latent: Latent representation (batch, latent_dim)
            seq_length: Target sequence length (optional)
        Returns:
            Reconstructed sequence (batch, seq_len, output_dim)
        """
        batch_size = latent.size(0)
        seq_len = seq_length or self.seq_length
        
        # Initialize hidden state from latent
        hidden = self.latent_proj(latent)
        hidden = hidden.view(self.num_layers, batch_size, self.hidden_dim)
        
        # Create decoder input (repeat latent for each time step)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input = self.layer_norm(
            nn.functional.pad(decoder_input, (0, self.hidden_dim - latent.size(1)))
        )
        
        # Alternative: use zeros as input and rely on hidden state
        # decoder_input = torch.zeros(batch_size, seq_len, self.hidden_dim, device=latent.device)
        
        # GRU decoding
        output, _ = self.gru(decoder_input, hidden)
        output = self.dropout(output)
        
        # Project to output dimension
        reconstructed = self.output_proj(output)
        
        return reconstructed


class LSTMEncoder(nn.Module):
    """LSTM-based encoder (alternative to GRU)"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.latent_proj = nn.Linear(
            hidden_dim * self.num_directions * 2,  # h and c states
            latent_dim
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        x = self.input_proj(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        output, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
            cell_cat = torch.cat([cell[-2], cell[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]
            cell_cat = cell[-1]
        
        combined = torch.cat([hidden_cat, cell_cat], dim=1)
        latent = self.latent_proj(combined)
        
        return latent, (hidden, cell)


class LSTMDecoder(nn.Module):
    """LSTM-based decoder"""
    
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        seq_length: int = 20
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        self.latent_proj_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_proj_c = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent: torch.Tensor, seq_length: Optional[int] = None) -> torch.Tensor:
        batch_size = latent.size(0)
        seq_len = seq_length or self.seq_length
        
        hidden = self.latent_proj_h(latent).view(self.num_layers, batch_size, self.hidden_dim)
        cell = self.latent_proj_c(latent).view(self.num_layers, batch_size, self.hidden_dim)
        
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_dim, device=latent.device)
        
        output, _ = self.lstm(decoder_input, (hidden, cell))
        output = self.dropout(output)
        reconstructed = self.output_proj(output)
        
        return reconstructed


class EdgeAutoencoder(nn.Module):
    """
    Complete autoencoder for edge devices.
    Supports both GRU and LSTM architectures.
    Optimized for Raspberry Pi deployment.
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        seq_length: int = 20,
        model_type: str = "gru",
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.model_type = model_type
        
        # Select encoder/decoder based on model type
        if model_type.lower() == "gru":
            self.encoder = GRUEncoder(
                input_dim, hidden_dim, latent_dim, 
                num_layers, dropout, bidirectional
            )
            self.decoder = GRUDecoder(
                input_dim, hidden_dim, latent_dim,
                num_layers, dropout, seq_length
            )
        elif model_type.lower() == "lstm":
            self.encoder = LSTMEncoder(
                input_dim, hidden_dim, latent_dim,
                num_layers, dropout, bidirectional
            )
            self.decoder = LSTMDecoder(
                input_dim, hidden_dim, latent_dim,
                num_layers, dropout, seq_length
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Anomaly detection threshold (learned or set)
        self.register_buffer('threshold', torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        Returns:
            reconstructed: Reconstructed sequence
            latent: Latent representation
        """
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent, x.size(1))
        return reconstructed, latent
    
    def compute_reconstruction_error(
        self, 
        x: torch.Tensor, 
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute reconstruction error (MSE).
        
        Args:
            x: Input tensor
            reduction: "mean", "sum", or "none"
        Returns:
            Reconstruction error
        """
        reconstructed, _ = self.forward(x)
        
        if reduction == "none":
            # Per-sample, per-timestep error
            error = torch.mean((x - reconstructed) ** 2, dim=-1)
        elif reduction == "mean":
            error = torch.mean((x - reconstructed) ** 2)
        else:
            error = torch.sum((x - reconstructed) ** 2)
            
        return error
    
    def detect_anomaly(
        self, 
        x: torch.Tensor, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies based on reconstruction error.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            threshold: Detection threshold (uses learned if None)
        Returns:
            is_anomaly: Boolean tensor (batch,)
            scores: Anomaly scores (batch,)
        """
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            
            # Compute per-sample reconstruction error
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            
            # Use provided threshold or learned threshold
            thresh = threshold if threshold is not None else self.threshold.item()
            
            is_anomaly = error > thresh
            
        return is_anomaly, error
    
    def update_threshold(self, normal_data: torch.Tensor, percentile: float = 95.0):
        """
        Update anomaly threshold based on normal data.
        
        Args:
            normal_data: Tensor of normal samples
            percentile: Percentile for threshold (e.g., 95 = 5% false positive)
        """
        with torch.no_grad():
            errors = self.compute_reconstruction_error(normal_data, reduction="none")
            errors = torch.mean(errors, dim=1)  # Per-sample error
            
            threshold = torch.quantile(errors, percentile / 100.0)
            self.threshold = threshold
            
        return threshold.item()
    
    def get_model_size(self) -> int:
        """Get model size in bytes"""
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.tell()
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_edge_model(config: Optional[Dict] = None) -> EdgeAutoencoder:
    """
    Factory function to create edge model from config.
    
    Args:
        config: Configuration dictionary (uses defaults if None)
    Returns:
        EdgeAutoencoder instance
    """
    from config import edge_model_config
    
    if config is None:
        config = {
            'input_dim': edge_model_config.INPUT_DIM,
            'hidden_dim': edge_model_config.HIDDEN_DIM,
            'latent_dim': edge_model_config.LATENT_DIM,
            'num_layers': edge_model_config.NUM_LAYERS,
            'dropout': edge_model_config.DROPOUT,
            'seq_length': edge_model_config.SEQUENCE_LENGTH,
            'model_type': edge_model_config.MODEL_TYPE,
            'bidirectional': edge_model_config.BIDIRECTIONAL,
        }
    
    model = EdgeAutoencoder(**config)
    
    print(f"Created {config['model_type'].upper()} Autoencoder:")
    print(f"  Parameters: {model.get_num_parameters():,}")
    print(f"  Model size: {model.get_model_size() / 1024:.2f} KB")
    
    return model


# TFLite conversion utilities for deployment
def convert_to_tflite(model: EdgeAutoencoder, sample_input: torch.Tensor) -> bytes:
    """
    Convert PyTorch model to TFLite for efficient Pi deployment.
    
    Args:
        model: Trained EdgeAutoencoder
        sample_input: Sample input for tracing
    Returns:
        TFLite model bytes
    """
    try:
        import tensorflow as tf
        
        # Export to ONNX first
        import torch.onnx
        import onnx
        from onnx_tf.backend import prepare
        
        # Trace model
        model.eval()
        onnx_path = "/tmp/edge_model.onnx"
        
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            input_names=['input'],
            output_names=['reconstructed', 'latent'],
            dynamic_axes={'input': {0: 'batch', 1: 'seq_len'}}
        )
        
        # Convert ONNX to TF
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph("/tmp/edge_model_tf")
        
        # Convert TF to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model("/tmp/edge_model_tf")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        return tflite_model
        
    except ImportError as e:
        print(f"TFLite conversion requires additional packages: {e}")
        return None


if __name__ == "__main__":
    # Test edge model
    print("Testing Edge Autoencoder...")
    
    # Create model
    model = create_edge_model()
    
    # Test forward pass
    batch_size = 4
    seq_length = 20
    input_dim = 22
    
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    reconstructed, latent = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Reconstruction error
    error = model.compute_reconstruction_error(x)
    print(f"Reconstruction error: {error.item():.4f}")
    
    # Anomaly detection
    is_anomaly, scores = model.detect_anomaly(x)
    print(f"Anomaly detected: {is_anomaly.tolist()}")
    print(f"Anomaly scores: {scores.tolist()}")
    
    print("\n✓ Edge model test passed!")
