# server_model.py - Transformer Autoencoder for Central Server
"""
Transformer-based autoencoder for the central aggregation server.
More powerful architecture leveraging server's greater compute resources.
Captures complex temporal patterns across federated edge data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Dict
import io


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with pre-norm architecture"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm self attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed forward
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + ff_out
        
        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        x = x + self.dropout(self_attn_out)
        
        # Cross attention
        x_norm = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(x_norm, memory, memory, attn_mask=memory_mask)
        x = x + self.dropout(cross_attn_out)
        
        # Feed forward
        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        x = x + ff_out
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with latent bottleneck"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        latent_dim: int,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Latent projection (global average pooling + projection)
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Attention mask (optional)
        Returns:
            latent: Latent representation (batch, latent_dim)
            memory: Encoder outputs for decoder (batch, seq_len, d_model)
        """
        # Project input to d_model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Final normalization
        memory = self.final_norm(x)
        
        # Global average pooling for latent representation
        pooled = torch.mean(memory, dim=1)
        latent = self.latent_proj(pooled)
        
        return latent, memory


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence reconstruction"""
    
    def __init__(
        self,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        latent_dim: int,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Latent to initial decoder state
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(max_seq_len, d_model)
        
        # Positional encoding for decoder
        self.pos_decoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        latent: torch.Tensor,
        memory: torch.Tensor,
        seq_length: int,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            latent: Latent representation (batch, latent_dim)
            memory: Encoder memory (batch, src_seq_len, d_model)
            seq_length: Target sequence length
            tgt_mask: Target attention mask (optional)
        Returns:
            Reconstructed sequence (batch, seq_length, output_dim)
        """
        batch_size = latent.size(0)
        
        # Create query embeddings
        positions = torch.arange(seq_length, device=latent.device)
        queries = self.query_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add latent information to queries
        latent_expanded = self.latent_proj(latent).unsqueeze(1)
        x = queries + latent_expanded
        
        # Add positional encoding
        x = self.pos_decoder(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to output dimension
        output = self.output_proj(x)
        
        return output


class ServerTransformerAutoencoder(nn.Module):
    """
    Complete Transformer Autoencoder for central server.
    More powerful than edge model, designed for pattern aggregation.
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 128,
        latent_dim: int = 16,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        
        # Encoder
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            latent_dim=latent_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            output_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            latent_dim=latent_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Anomaly threshold
        self.register_buffer('threshold', torch.tensor(0.5))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self, 
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            src_mask: Source attention mask (optional)
        Returns:
            reconstructed: Reconstructed sequence
            latent: Latent representation
        """
        seq_length = x.size(1)
        
        # Encode
        latent, memory = self.encoder(x, src_mask)
        
        # Decode
        reconstructed = self.decoder(latent, memory, seq_length)
        
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation only"""
        latent, _ = self.encoder(x)
        return latent
    
    def decode(
        self, 
        latent: torch.Tensor, 
        memory: torch.Tensor, 
        seq_length: int
    ) -> torch.Tensor:
        """Decode from latent representation"""
        return self.decoder(latent, memory, seq_length)
    
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Compute reconstruction error (MSE)"""
        reconstructed, _ = self.forward(x)
        
        if reduction == "none":
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
        """Detect anomalies based on reconstruction error"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            
            thresh = threshold if threshold is not None else self.threshold.item()
            is_anomaly = error > thresh
            
        return is_anomaly, error
    
    def compute_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for interpretability.
        Useful for understanding which time steps contribute to anomaly detection.
        """
        attention_weights = {}
        
        # Hook to capture attention weights
        def get_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights[name] = output[1]
            return hook
        
        # Register hooks on attention layers
        hooks = []
        for i, layer in enumerate(self.encoder.encoder_layers):
            h = layer.self_attn.register_forward_hook(get_attention_hook(f'encoder_layer_{i}'))
            hooks.append(h)
        
        # Forward pass
        with torch.no_grad():
            _, _ = self.forward(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
            
        return attention_weights
    
    def update_threshold(self, normal_data: torch.Tensor, percentile: float = 95.0):
        """Update anomaly threshold based on normal data"""
        with torch.no_grad():
            errors = self.compute_reconstruction_error(normal_data, reduction="none")
            errors = torch.mean(errors, dim=1)
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


class HybridServerModel(nn.Module):
    """
    Hybrid model that combines Transformer with edge model knowledge.
    Aggregates patterns from multiple edge devices.
    """
    
    def __init__(
        self,
        transformer_config: Dict,
        num_edge_devices: int = 3
    ):
        super().__init__()
        
        self.num_edge_devices = num_edge_devices
        
        # Main transformer autoencoder
        self.transformer = ServerTransformerAutoencoder(**transformer_config)
        
        # Edge model aggregation layer
        # Combines latent representations from edge models
        edge_latent_dim = 8  # From edge model config
        self.edge_aggregator = nn.Sequential(
            nn.Linear(edge_latent_dim * num_edge_devices, transformer_config['latent_dim']),
            nn.GELU(),
            nn.LayerNorm(transformer_config['latent_dim'])
        )
        
        # Fusion layer to combine transformer and edge knowledge
        self.fusion = nn.Sequential(
            nn.Linear(transformer_config['latent_dim'] * 2, transformer_config['latent_dim']),
            nn.GELU(),
            nn.Linear(transformer_config['latent_dim'], transformer_config['latent_dim'])
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_latents: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional edge model integration.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            edge_latents: Concatenated edge latent representations (batch, edge_latent_dim * num_edges)
        """
        # Get transformer representations
        reconstructed, latent = self.transformer(x)
        
        # If edge latents provided, fuse them
        if edge_latents is not None:
            edge_aggregated = self.edge_aggregator(edge_latents)
            combined = torch.cat([latent, edge_aggregated], dim=-1)
            latent = self.fusion(combined)
        
        return reconstructed, latent


def create_server_model(config: Optional[Dict] = None) -> ServerTransformerAutoencoder:
    """
    Factory function to create server model from config.
    """
    from config import server_model_config
    
    if config is None:
        config = {
            'input_dim': server_model_config.INPUT_DIM,
            'd_model': server_model_config.D_MODEL,
            'nhead': server_model_config.NHEAD,
            'num_encoder_layers': server_model_config.NUM_ENCODER_LAYERS,
            'num_decoder_layers': server_model_config.NUM_DECODER_LAYERS,
            'dim_feedforward': server_model_config.DIM_FEEDFORWARD,
            'latent_dim': server_model_config.LATENT_DIM,
            'max_seq_len': server_model_config.MAX_SEQUENCE_LENGTH,
            'dropout': server_model_config.DROPOUT,
        }
    
    model = ServerTransformerAutoencoder(**config)
    
    print(f"Created Server Transformer Autoencoder:")
    print(f"  D_model: {config['d_model']}")
    print(f"  Attention heads: {config['nhead']}")
    print(f"  Encoder layers: {config['num_encoder_layers']}")
    print(f"  Decoder layers: {config['num_decoder_layers']}")
    print(f"  Parameters: {model.get_num_parameters():,}")
    print(f"  Model size: {model.get_model_size() / 1024:.2f} KB")
    
    return model


if __name__ == "__main__":
    print("Testing Server Transformer Autoencoder...")
    
    # Create model
    model = create_server_model()
    
    # Test forward pass
    batch_size = 8
    seq_length = 50
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
    print(f"Anomaly detected: {is_anomaly.sum().item()}/{batch_size}")
    
    # Attention weights
    attn_weights = model.compute_attention_weights(x)
    print(f"Attention weight tensors: {list(attn_weights.keys())}")
    
    print("\n✓ Server model test passed!")
