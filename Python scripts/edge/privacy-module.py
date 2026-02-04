# privacy.py - Differential Privacy and Gradient Protection Mechanisms
"""
Implements three privacy mechanisms for federated learning:
1. Input Perturbation - Add noise to training data
2. Output Perturbation - Add noise to model updates/gradients
3. DP-SGD - Differentially private stochastic gradient descent

Includes analysis of privacy-utility tradeoffs and gradient leakage resistance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import math
from collections import OrderedDict
import copy


class PrivacyMechanism(Enum):
    NONE = "none"
    INPUT_PERTURBATION = "input_perturbation"
    OUTPUT_PERTURBATION = "output_perturbation"
    DP_SGD = "dp_sgd"


@dataclass
class PrivacyMetrics:
    """Tracks privacy budget and metrics"""
    epsilon_spent: float = 0.0
    delta: float = 1e-5
    noise_scale: float = 0.0
    num_queries: int = 0
    gradient_norm_before: float = 0.0
    gradient_norm_after: float = 0.0
    

class PrivacyAccountant:
    """
    Tracks privacy budget across training.
    Uses Rényi Differential Privacy (RDP) accounting for tighter bounds.
    """
    
    def __init__(self, epsilon_budget: float, delta: float = 1e-5):
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.epsilon_spent = 0.0
        self.history = []
        
    def add_noise_event(self, noise_multiplier: float, sample_rate: float, steps: int = 1):
        """Record a privacy-spending event"""
        # Compute epsilon for this event using RDP
        epsilon = self._compute_rdp_epsilon(noise_multiplier, sample_rate, steps)
        self.epsilon_spent += epsilon
        self.history.append({
            'noise_multiplier': noise_multiplier,
            'sample_rate': sample_rate,
            'steps': steps,
            'epsilon': epsilon
        })
        return epsilon
    
    def _compute_rdp_epsilon(self, noise_multiplier: float, sample_rate: float, steps: int) -> float:
        """Compute epsilon using RDP accountant (simplified)"""
        if noise_multiplier == 0:
            return float('inf')
        
        # Simplified RDP computation
        # For exact computation, use Google's dp-accounting library
        rdp_order = 2.0  # Rényi order
        rdp = (sample_rate ** 2) * rdp_order / (2 * noise_multiplier ** 2)
        epsilon = rdp * steps + math.log(1 / self.delta) / (rdp_order - 1)
        return epsilon
    
    def remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.epsilon_budget - self.epsilon_spent)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        return self.epsilon_spent >= self.epsilon_budget


class InputPerturbation:
    """
    Input Perturbation: Add calibrated noise to training data.
    
    Pros:
    - Simple to implement
    - Protects raw data directly
    - No modification to training algorithm
    
    Cons:
    - Reduces data quality
    - May need more data for same accuracy
    - Noise accumulates with feature dimensions
    
    Gradient Leakage Resistance: MEDIUM
    - Attackers can't recover exact inputs
    - But statistical properties may leak
    """
    
    def __init__(
        self,
        noise_scale: float = 0.1,
        clip_norm: float = 1.0,
        mechanism: str = "gaussian"  # "gaussian" or "laplace"
    ):
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.mechanism = mechanism
        self.metrics = PrivacyMetrics()
        
    def perturb(self, data: torch.Tensor) -> torch.Tensor:
        """Add noise to input data"""
        # Clip data to bounded range
        data_clipped = self._clip_data(data)
        
        # Add noise
        if self.mechanism == "gaussian":
            noise = torch.randn_like(data_clipped) * self.noise_scale
        else:  # laplace
            noise = torch.distributions.Laplace(0, self.noise_scale).sample(data_clipped.shape)
            noise = noise.to(data.device)
        
        perturbed = data_clipped + noise
        
        # Update metrics
        self.metrics.noise_scale = self.noise_scale
        self.metrics.num_queries += 1
        
        return perturbed
    
    def _clip_data(self, data: torch.Tensor) -> torch.Tensor:
        """Clip data to bounded L2 norm"""
        norms = torch.norm(data, dim=-1, keepdim=True)
        scale = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
        return data * scale
    
    def compute_epsilon(self, sensitivity: float = 1.0, delta: float = 1e-5) -> float:
        """Compute privacy guarantee (epsilon)"""
        if self.mechanism == "gaussian":
            # Gaussian mechanism epsilon
            epsilon = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / self.noise_scale
        else:
            # Laplace mechanism epsilon
            epsilon = sensitivity / self.noise_scale
        return epsilon


class OutputPerturbation:
    """
    Output Perturbation: Add noise to model updates/gradients before sharing.
    
    Pros:
    - Doesn't degrade training data quality
    - Directly protects shared information
    - Well-studied theoretical guarantees
    
    Cons:
    - May slow convergence
    - Requires careful noise calibration
    - Need to clip gradients first
    
    Gradient Leakage Resistance: HIGH
    - Directly protects against gradient inversion attacks
    - Noise masks individual contributions
    """
    
    def __init__(
        self,
        noise_scale: float = 0.01,
        clip_norm: float = 1.0,
        mechanism: str = "gaussian"
    ):
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.mechanism = mechanism
        self.metrics = PrivacyMetrics()
        
    def perturb_gradients(
        self, 
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add noise to gradient dictionary"""
        perturbed = {}
        
        # First clip gradients
        clipped_grads = self._clip_gradients(gradients)
        
        # Then add noise
        for name, grad in clipped_grads.items():
            if self.mechanism == "gaussian":
                noise = torch.randn_like(grad) * self.noise_scale
            else:
                noise = torch.distributions.Laplace(0, self.noise_scale).sample(grad.shape)
                noise = noise.to(grad.device)
            
            perturbed[name] = grad + noise
        
        self.metrics.num_queries += 1
        return perturbed
    
    def perturb_model_update(
        self, 
        model_update: OrderedDict
    ) -> OrderedDict:
        """Add noise to model update (weight differences)"""
        perturbed = OrderedDict()
        
        total_norm_before = 0.0
        total_norm_after = 0.0
        
        for name, param in model_update.items():
            # Clip
            norm = torch.norm(param)
            total_norm_before += norm.item() ** 2
            
            scale = min(1.0, self.clip_norm / (norm.item() + 1e-8))
            clipped = param * scale
            
            # Add noise
            if self.mechanism == "gaussian":
                noise = torch.randn_like(clipped) * self.noise_scale * self.clip_norm
            else:
                noise = torch.distributions.Laplace(0, self.noise_scale * self.clip_norm).sample(clipped.shape)
                noise = noise.to(clipped.device)
            
            perturbed[name] = clipped + noise
            total_norm_after += torch.norm(perturbed[name]).item() ** 2
        
        self.metrics.gradient_norm_before = math.sqrt(total_norm_before)
        self.metrics.gradient_norm_after = math.sqrt(total_norm_after)
        self.metrics.num_queries += 1
        
        return perturbed
    
    def _clip_gradients(
        self, 
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Clip gradients by global norm"""
        # Compute global norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        total_norm = math.sqrt(total_norm)
        
        self.metrics.gradient_norm_before = total_norm
        
        # Clip
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-8))
        clipped = {name: grad * clip_coef for name, grad in gradients.items()}
        
        return clipped
    
    def compute_epsilon(
        self, 
        num_samples: int, 
        batch_size: int, 
        epochs: int,
        delta: float = 1e-5
    ) -> float:
        """Compute overall privacy guarantee"""
        sample_rate = batch_size / num_samples
        steps = epochs * (num_samples // batch_size)
        
        # Using RDP composition
        rdp_order = 2.0
        rdp_per_step = (sample_rate ** 2) * rdp_order / (2 * self.noise_scale ** 2)
        total_rdp = rdp_per_step * steps
        
        epsilon = total_rdp + math.log(1 / delta) / (rdp_order - 1)
        return epsilon


class DPSGDOptimizer:
    """
    Differentially Private SGD: Full DP training procedure.
    
    Pros:
    - Strongest privacy guarantees
    - Formal composition theorems
    - Per-sample gradient clipping
    
    Cons:
    - Slowest training
    - Requires per-sample gradients
    - Most complex to implement
    
    Gradient Leakage Resistance: HIGHEST
    - Per-sample clipping prevents any single sample from dominating
    - Noise is calibrated to sensitivity
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        batch_size: int = 32,
        num_samples: int = 1000
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        self.accountant = PrivacyAccountant(epsilon_budget=10.0)
        self.metrics = PrivacyMetrics()
        
    def compute_per_sample_gradients(
        self, 
        loss_fn: Callable,
        data: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for each sample individually.
        This is the key for DP-SGD - we need per-sample gradients to clip.
        """
        batch_size = data.size(0)
        per_sample_grads = {name: [] for name, _ in self.model.named_parameters()}
        
        # Compute gradient for each sample
        for i in range(batch_size):
            self.model.zero_grad()
            
            sample_data = data[i:i+1]
            sample_target = targets[i:i+1]
            
            output = self.model(sample_data)
            if isinstance(output, tuple):
                output = output[0]  # Get reconstruction
            
            loss = loss_fn(output, sample_target)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    per_sample_grads[name].append(param.grad.clone())
        
        # Stack gradients
        stacked_grads = {
            name: torch.stack(grads) 
            for name, grads in per_sample_grads.items()
        }
        
        return stacked_grads
    
    def clip_gradients(
        self, 
        per_sample_grads: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Clip per-sample gradients by L2 norm"""
        batch_size = list(per_sample_grads.values())[0].size(0)
        
        # Compute per-sample gradient norms
        norms = torch.zeros(batch_size, device=list(per_sample_grads.values())[0].device)
        for grad in per_sample_grads.values():
            norms += torch.norm(grad.view(batch_size, -1), dim=1) ** 2
        norms = torch.sqrt(norms)
        
        # Clip factors
        clip_factors = torch.clamp(self.max_grad_norm / (norms + 1e-8), max=1.0)
        
        # Apply clipping
        clipped = {}
        for name, grad in per_sample_grads.items():
            clipped[name] = grad * clip_factors.view(-1, *([1] * (grad.dim() - 1)))
        
        return clipped
    
    def add_noise(
        self, 
        clipped_grads: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise calibrated to sensitivity"""
        noisy_grads = {}
        
        noise_std = self.noise_multiplier * self.max_grad_norm
        
        for name, grad in clipped_grads.items():
            # Average over batch
            avg_grad = torch.mean(grad, dim=0)
            
            # Add noise
            noise = torch.randn_like(avg_grad) * noise_std / grad.size(0)
            noisy_grads[name] = avg_grad + noise
        
        return noisy_grads
    
    def step(
        self, 
        loss_fn: Callable,
        data: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Perform one DP-SGD step"""
        # 1. Compute per-sample gradients
        per_sample_grads = self.compute_per_sample_gradients(loss_fn, data, targets)
        
        # 2. Clip per-sample gradients
        clipped_grads = self.clip_gradients(per_sample_grads)
        
        # 3. Add noise and average
        noisy_grads = self.add_noise(clipped_grads)
        
        # 4. Apply gradients
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in noisy_grads:
                    param -= self.learning_rate * noisy_grads[name]
        
        # 5. Account for privacy spent
        sample_rate = self.batch_size / self.num_samples
        self.accountant.add_noise_event(self.noise_multiplier, sample_rate, steps=1)
        
        # Compute loss for monitoring
        with torch.no_grad():
            output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = loss_fn(output, targets)
        
        return loss.item()
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get epsilon and delta spent so far"""
        return self.accountant.epsilon_spent, self.accountant.delta


class PrivacyComparison:
    """
    Compare privacy mechanisms for analysis.
    Evaluates:
    1. Privacy guarantee (epsilon)
    2. Utility (model accuracy)
    3. Gradient leakage resistance
    4. Computational overhead
    """
    
    def __init__(self):
        self.results = {
            'input_perturbation': {},
            'output_perturbation': {},
            'dp_sgd': {}
        }
    
    def evaluate_gradient_leakage_resistance(
        self,
        mechanism: str,
        original_gradients: Dict[str, torch.Tensor],
        protected_gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Measure resistance to gradient leakage attacks.
        Uses cosine similarity and reconstruction error metrics.
        """
        metrics = {}
        
        # Cosine similarity (lower = better protection)
        cos_sims = []
        for name in original_gradients:
            orig = original_gradients[name].flatten()
            prot = protected_gradients[name].flatten()
            
            cos_sim = torch.nn.functional.cosine_similarity(
                orig.unsqueeze(0), prot.unsqueeze(0)
            ).item()
            cos_sims.append(cos_sim)
        
        metrics['avg_cosine_similarity'] = np.mean(cos_sims)
        
        # Relative error
        rel_errors = []
        for name in original_gradients:
            orig_norm = torch.norm(original_gradients[name]).item()
            diff_norm = torch.norm(original_gradients[name] - protected_gradients[name]).item()
            rel_error = diff_norm / (orig_norm + 1e-8)
            rel_errors.append(rel_error)
        
        metrics['avg_relative_error'] = np.mean(rel_errors)
        
        # Gradient leakage resistance score (0-1, higher = better)
        # Based on: high relative error + low cosine similarity = good
        metrics['leakage_resistance_score'] = (
            0.5 * (1 - metrics['avg_cosine_similarity']) +
            0.5 * min(1.0, metrics['avg_relative_error'])
        )
        
        self.results[mechanism]['leakage_resistance'] = metrics
        return metrics
    
    def generate_comparison_report(self) -> str:
        """Generate human-readable comparison report"""
        report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRIVACY MECHANISM COMPARISON REPORT                        ║
╠══════════════════════════════════════════════════════════════════════════════╣

1. INPUT PERTURBATION
   ├─ Privacy: MEDIUM (ε depends on noise scale and data dimension)
   ├─ Utility: HIGH (training unmodified)
   ├─ Gradient Leakage Resistance: MEDIUM
   │  └─ Protects raw data but statistical patterns may leak
   ├─ Computational Overhead: LOW
   └─ Best For: Scenarios where data privacy is primary concern

2. OUTPUT PERTURBATION
   ├─ Privacy: HIGH (directly protects shared model updates)
   ├─ Utility: MEDIUM-HIGH (some convergence slowdown)
   ├─ Gradient Leakage Resistance: HIGH
   │  └─ Directly masks gradients, strong against inversion attacks
   ├─ Computational Overhead: LOW
   └─ Best For: Federated learning with untrusted aggregator

3. DP-SGD (Differentially Private SGD)
   ├─ Privacy: HIGHEST (formal DP guarantees with composition)
   ├─ Utility: MEDIUM (requires more epochs, careful tuning)
   ├─ Gradient Leakage Resistance: HIGHEST
   │  └─ Per-sample clipping + noise provides strongest protection
   ├─ Computational Overhead: HIGH (per-sample gradient computation)
   └─ Best For: Maximum privacy requirements, adversarial settings

╠══════════════════════════════════════════════════════════════════════════════╣
║                           RECOMMENDATION                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣

For Vehicle IDS Federated Learning:

→ PRIMARY: OUTPUT PERTURBATION
  - Good balance of privacy and utility
  - Protects against gradient leakage from malicious aggregator
  - Low overhead suitable for edge devices

→ ENHANCED: OUTPUT PERTURBATION + GRADIENT CLIPPING
  - Add per-layer gradient clipping before noise
  - Provides stronger guarantees with minimal overhead

→ MAXIMUM SECURITY: DP-SGD
  - Use when adversarial attacks are expected
  - Trade-off: longer training time, lower accuracy

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report


def create_privacy_mechanism(
    mechanism_type: PrivacyMechanism,
    config: Dict
) -> object:
    """Factory function to create privacy mechanism"""
    
    if mechanism_type == PrivacyMechanism.INPUT_PERTURBATION:
        return InputPerturbation(
            noise_scale=config.get('noise_scale', 0.1),
            clip_norm=config.get('clip_norm', 1.0)
        )
    elif mechanism_type == PrivacyMechanism.OUTPUT_PERTURBATION:
        return OutputPerturbation(
            noise_scale=config.get('noise_scale', 0.01),
            clip_norm=config.get('clip_norm', 1.0)
        )
    elif mechanism_type == PrivacyMechanism.DP_SGD:
        return DPSGDOptimizer(
            model=config['model'],
            learning_rate=config.get('learning_rate', 0.001),
            noise_multiplier=config.get('noise_multiplier', 1.1),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            batch_size=config.get('batch_size', 32),
            num_samples=config.get('num_samples', 1000)
        )
    else:
        return None


if __name__ == "__main__":
    print("Testing Privacy Mechanisms...")
    
    # Create sample data
    data = torch.randn(32, 20, 22)
    
    # Test Input Perturbation
    print("\n1. Input Perturbation:")
    ip = InputPerturbation(noise_scale=0.1)
    perturbed_data = ip.perturb(data)
    print(f"   Original mean: {data.mean():.4f}")
    print(f"   Perturbed mean: {perturbed_data.mean():.4f}")
    print(f"   Epsilon: {ip.compute_epsilon():.4f}")
    
    # Test Output Perturbation
    print("\n2. Output Perturbation:")
    op = OutputPerturbation(noise_scale=0.01)
    gradients = {'layer1': torch.randn(32, 64), 'layer2': torch.randn(64, 22)}
    perturbed_grads = op.perturb_gradients(gradients)
    print(f"   Original grad norm: {torch.norm(gradients['layer1']):.4f}")
    print(f"   Perturbed grad norm: {torch.norm(perturbed_grads['layer1']):.4f}")
    
    # Print comparison report
    comparison = PrivacyComparison()
    print(comparison.generate_comparison_report())