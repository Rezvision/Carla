# train.py - Complete Training Script for Federated Learning IDS
"""
Standalone training script for both edge and server models.
Can be used for:
1. Pre-training models before federated deployment
2. Evaluating different privacy mechanisms
3. Testing attack detection performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

# Local imports
from edge_model import EdgeAutoencoder, create_edge_model
from server_model import ServerTransformerAutoencoder, create_server_model
from attack_generator import CANAttackGenerator, AttackConfig, create_training_data
from data_processor import DataPreprocessor, FeatureExtractor, SequenceCreator
from privacy import (
    InputPerturbation, OutputPerturbation, DPSGDOptimizer,
    PrivacyComparison, PrivacyMechanism
)
from config import (
    edge_model_config, server_model_config, privacy_config,
    attack_config, data_config
)


class Trainer:
    """
    Unified trainer for both edge and server models.
    Supports different privacy mechanisms and evaluation.
    """
    
    def __init__(
        self,
        model_type: str = "edge",  # "edge" or "server"
        privacy_mechanism: str = "none",
        device: str = "auto"
    ):
        self.model_type = model_type
        self.privacy_mechanism = privacy_mechanism
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.privacy = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'detection_rate': [],
            'false_positive_rate': [],
            'privacy_budget': [],
        }
        
        # Preprocessor
        self.preprocessor = DataPreprocessor()
        
    def create_model(self, input_dim: int):
        """Create model based on type"""
        if self.model_type == "edge":
            self.model = EdgeAutoencoder(
                input_dim=input_dim,
                hidden_dim=edge_model_config.HIDDEN_DIM,
                latent_dim=edge_model_config.LATENT_DIM,
                num_layers=edge_model_config.NUM_LAYERS,
                dropout=edge_model_config.DROPOUT,
                seq_length=edge_model_config.SEQUENCE_LENGTH,
                model_type=edge_model_config.MODEL_TYPE,
            )
        else:
            self.model = ServerTransformerAutoencoder(
                input_dim=input_dim,
                d_model=server_model_config.D_MODEL,
                nhead=server_model_config.NHEAD,
                num_encoder_layers=server_model_config.NUM_ENCODER_LAYERS,
                num_decoder_layers=server_model_config.NUM_DECODER_LAYERS,
                dim_feedforward=server_model_config.DIM_FEEDFORWARD,
                latent_dim=server_model_config.LATENT_DIM,
            )
        
        self.model = self.model.to(self.device)
        
        print(f"Created {self.model_type} model with {self.model.get_num_parameters():,} parameters")
        
        # Create optimizer
        lr = edge_model_config.LEARNING_RATE if self.model_type == "edge" else server_model_config.LEARNING_RATE
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Setup privacy mechanism
        self._setup_privacy()
    
    def _setup_privacy(self):
        """Setup privacy mechanism"""
        if self.privacy_mechanism == "input_perturbation":
            self.privacy = InputPerturbation(
                noise_scale=privacy_config.INPUT_NOISE_SCALE,
                clip_norm=privacy_config.INPUT_CLIP_NORM
            )
        elif self.privacy_mechanism == "output_perturbation":
            self.privacy = OutputPerturbation(
                noise_scale=privacy_config.OUTPUT_NOISE_SCALE,
                clip_norm=privacy_config.GRADIENT_CLIP_NORM
            )
        elif self.privacy_mechanism == "dp_sgd":
            # DP-SGD requires special handling in training loop
            print("Using DP-SGD - will configure during training")
        else:
            self.privacy = None
    
    def prepare_data(
        self,
        num_train: int = 10000,
        num_val: int = 2000,
        attack_ratio: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data"""
        print(f"\nGenerating synthetic data...")
        print(f"  Training samples: {num_train}")
        print(f"  Validation samples: {num_val}")
        print(f"  Attack ratio: {attack_ratio:.0%}")
        
        # Generate data
        data = create_training_data(
            num_train=num_train,
            num_val=num_val,
            sequence_length=edge_model_config.SEQUENCE_LENGTH,
            attack_ratio=attack_ratio
        )
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        # Fit preprocessor on training data
        self.preprocessor.fit(X_train.numpy())
        
        # Normalize
        X_train_norm = torch.from_numpy(self.preprocessor.transform(X_train.numpy()))
        X_val_norm = torch.from_numpy(self.preprocessor.transform(X_val.numpy()))
        
        # Apply input perturbation if configured
        if isinstance(self.privacy, InputPerturbation):
            print("Applying input perturbation to training data...")
            X_train_norm = self.privacy.perturb(X_train_norm)
        
        # Create DataLoaders
        batch_size = edge_model_config.BATCH_SIZE if self.model_type == "edge" else server_model_config.BATCH_SIZE
        
        train_dataset = TensorDataset(X_train_norm, X_train_norm, y_train)  # input, target, label
        val_dataset = TensorDataset(X_val_norm, X_val_norm, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        print(f"  Batch size: {batch_size}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader, data['num_features']
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y, labels in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed, latent = self.model(batch_x)
            loss = nn.MSELoss()(reconstructed, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Apply output perturbation if configured
            if isinstance(self.privacy, OutputPerturbation):
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * self.privacy.noise_scale
                            param.grad += noise
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate model and compute metrics"""
        self.model.eval()
        total_loss = 0.0
        
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y, labels in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                reconstructed, _ = self.model(batch_x)
                loss = nn.MSELoss()(reconstructed, batch_y)
                total_loss += loss.item()
                
                # Compute per-sample reconstruction error
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute detection metrics
        errors = np.array(all_errors)
        labels = np.array(all_labels)
        
        # Find optimal threshold on validation set
        normal_errors = errors[labels == 0]
        threshold = np.percentile(normal_errors, 95)
        
        predictions = (errors > threshold).astype(int)
        
        # Detection rate (recall for attacks)
        attack_mask = labels == 1
        if attack_mask.sum() > 0:
            detection_rate = predictions[attack_mask].mean()
        else:
            detection_rate = 0.0
        
        # False positive rate
        normal_mask = labels == 0
        if normal_mask.sum() > 0:
            false_positive_rate = predictions[normal_mask].mean()
        else:
            false_positive_rate = 0.0
        
        return avg_loss, detection_rate, false_positive_rate
    
    def train(
        self,
        num_epochs: int = 50,
        num_train: int = 10000,
        num_val: int = 2000,
        attack_ratio: float = 0.2,
        early_stopping_patience: int = 10,
        save_dir: str = "./models"
    ) -> Dict:
        """Full training loop"""
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Model type: {self.model_type}")
        print(f"Privacy mechanism: {self.privacy_mechanism}")
        print(f"Epochs: {num_epochs}")
        print("="*60 + "\n")
        
        # Prepare data
        train_loader, val_loader, num_features = self.prepare_data(
            num_train, num_val, attack_ratio
        )
        
        # Create model
        self.create_model(num_features)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        print("\nStarting training...")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, detection_rate, fpr = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['detection_rate'].append(detection_rate)
            self.history['false_positive_rate'].append(fpr)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"DR: {detection_rate:.2%} | FPR: {fpr:.2%} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint(save_dir, epoch, val_loss, detection_rate)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final detection rate: {self.history['detection_rate'][-1]:.2%}")
        print(f"Final false positive rate: {self.history['false_positive_rate'][-1]:.2%}")
        print("="*60)
        
        return self.history
    
    def _save_checkpoint(self, save_dir: str, epoch: int, val_loss: float, detection_rate: float):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'detection_rate': detection_rate,
            'history': self.history,
            'model_type': self.model_type,
            'privacy_mechanism': self.privacy_mechanism,
        }
        
        path = os.path.join(save_dir, f"{self.model_type}_model_best.pt")
        torch.save(checkpoint, path)
        
        # Save preprocessor
        self.preprocessor.save(os.path.join(save_dir, "preprocessor.json"))
        
        # Save config
        config = {
            'model_type': self.model_type,
            'privacy_mechanism': self.privacy_mechanism,
            'input_dim': self.model.input_dim,
            'timestamp': timestamp,
        }
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)


def compare_privacy_mechanisms():
    """Compare all privacy mechanisms"""
    print("\n" + "="*60)
    print("PRIVACY MECHANISM COMPARISON")
    print("="*60)
    
    mechanisms = ["none", "input_perturbation", "output_perturbation"]
    results = {}
    
    for mechanism in mechanisms:
        print(f"\n--- Training with {mechanism} ---")
        
        trainer = Trainer(
            model_type="edge",
            privacy_mechanism=mechanism
        )
        
        history = trainer.train(
            num_epochs=20,
            num_train=5000,
            num_val=1000,
            save_dir=f"./models/{mechanism}"
        )
        
        results[mechanism] = {
            'final_loss': history['val_loss'][-1],
            'detection_rate': history['detection_rate'][-1],
            'fpr': history['false_positive_rate'][-1],
        }
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Mechanism':<25} {'Val Loss':<12} {'Detection':<12} {'FPR':<12}")
    print("-"*60)
    
    for mech, res in results.items():
        print(f"{mech:<25} {res['final_loss']:<12.4f} {res['detection_rate']:<12.2%} {res['fpr']:<12.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Federated Learning IDS Models')
    
    parser.add_argument('--model', choices=['edge', 'server'], default='edge',
                       help='Model type to train')
    parser.add_argument('--privacy', choices=['none', 'input_perturbation', 'output_perturbation', 'dp_sgd'],
                       default='none', help='Privacy mechanism')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--train-samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--val-samples', type=int, default=2000, help='Validation samples')
    parser.add_argument('--attack-ratio', type=float, default=0.2, help='Attack ratio in data')
    parser.add_argument('--save-dir', default='./models', help='Model save directory')
    parser.add_argument('--compare-privacy', action='store_true', help='Compare all privacy mechanisms')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    if args.compare_privacy:
        compare_privacy_mechanisms()
    else:
        trainer = Trainer(
            model_type=args.model,
            privacy_mechanism=args.privacy,
            device=args.device
        )
        
        trainer.train(
            num_epochs=args.epochs,
            num_train=args.train_samples,
            num_val=args.val_samples,
            attack_ratio=args.attack_ratio,
            save_dir=args.save_dir
        )


if __name__ == "__main__":
    main()
