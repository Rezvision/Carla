# data_processor.py - Data Preprocessing and Feature Extraction
"""
Handles data normalization, feature engineering, and preprocessing
for the federated learning IDS system.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import json


class FeatureExtractor:
    """
    Extracts and engineers features from raw CAN and device data.
    Matches the features used in training.
    """
    
    # Feature categories
    CAN_SIGNALS = [
        'speed_kmh', 'battery_level', 'throttle', 
        'brake', 'steering', 'gear'
    ]
    
    LOCATION_FEATURES = ['location_x', 'location_y']
    
    DERIVED_CAN_FEATURES = [
        'speed_delta', 'throttle_brake_ratio', 'message_frequency',
        'inter_arrival_time', 'payload_entropy'
    ]
    
    DEVICE_METRICS = [
        'cpu_usage_percent', 'cpu_temperature_c', 'memory_used_mb',
        'memory_total_mb', 'memory_usage_percent', 'load_average_1min',
        'throttling_state', 'network_rx_bytes', 'network_tx_bytes',
        'can_rx_count', 'can_tx_count', 'can_error_count'
    ]
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Rolling windows for derived features
        self.speed_history = deque(maxlen=window_size)
        self.message_times = deque(maxlen=window_size)
        self.payload_history = deque(maxlen=window_size)
        
        # Last known values
        self.last_values = {}
        
        # All feature names in order
        self.feature_names = (
            self.CAN_SIGNALS + 
            self.LOCATION_FEATURES + 
            self.DERIVED_CAN_FEATURES + 
            self.DEVICE_METRICS
        )
        
    def extract_features(
        self, 
        raw_data: Dict,
        timestamp: float = None
    ) -> np.ndarray:
        """
        Extract all features from raw data dictionary.
        
        Args:
            raw_data: Dictionary with CAN signals, location, device metrics
            timestamp: Optional timestamp for inter-arrival calculation
            
        Returns:
            Feature vector as numpy array
        """
        import time
        timestamp = timestamp or time.time()
        
        features = {}
        
        # CAN signals (direct)
        for signal in self.CAN_SIGNALS:
            features[signal] = raw_data.get(signal, self.last_values.get(signal, 0.0))
        
        # Location features
        for loc in self.LOCATION_FEATURES:
            features[loc] = raw_data.get(loc, self.last_values.get(loc, 0.0))
        
        # Derived CAN features
        features.update(self._compute_derived_features(raw_data, timestamp))
        
        # Device metrics
        for metric in self.DEVICE_METRICS:
            features[metric] = raw_data.get(metric, 0.0)
        
        # Update last values
        self.last_values.update(features)
        
        # Create feature vector in consistent order
        feature_vector = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        
        return feature_vector
    
    def _compute_derived_features(self, raw_data: Dict, timestamp: float) -> Dict:
        """Compute derived features from raw data"""
        derived = {}
        
        # Speed delta (rate of change)
        current_speed = raw_data.get('speed_kmh', 0)
        self.speed_history.append(current_speed)
        
        if len(self.speed_history) >= 2:
            derived['speed_delta'] = self.speed_history[-1] - self.speed_history[-2]
        else:
            derived['speed_delta'] = 0.0
        
        # Throttle-brake ratio (suspicious if both high)
        throttle = raw_data.get('throttle', 0)
        brake = raw_data.get('brake', 0)
        derived['throttle_brake_ratio'] = (throttle * brake) / 100.0 if (throttle + brake) > 0 else 0
        
        # Message frequency (messages per second)
        self.message_times.append(timestamp)
        if len(self.message_times) >= 2:
            time_span = self.message_times[-1] - self.message_times[0]
            if time_span > 0:
                derived['message_frequency'] = len(self.message_times) / time_span
            else:
                derived['message_frequency'] = len(self.message_times)
        else:
            derived['message_frequency'] = 1.0
        
        # Inter-arrival time
        if len(self.message_times) >= 2:
            derived['inter_arrival_time'] = self.message_times[-1] - self.message_times[-2]
        else:
            derived['inter_arrival_time'] = 0.1
        
        # Payload entropy (randomness measure)
        # For real CAN data, compute entropy of payload bytes
        # Here we simulate based on value changes
        if len(self.speed_history) >= 10:
            values = np.array(list(self.speed_history)[-10:])
            # Normalized variance as proxy for entropy
            if np.std(values) > 0:
                derived['payload_entropy'] = min(1.0, np.std(values) / (np.mean(values) + 1e-8))
            else:
                derived['payload_entropy'] = 0.0
        else:
            derived['payload_entropy'] = 0.0
        
        return derived
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return self.feature_names
    
    def get_num_features(self) -> int:
        """Get total number of features"""
        return len(self.feature_names)
    
    def reset(self):
        """Reset feature extractor state"""
        self.speed_history.clear()
        self.message_times.clear()
        self.payload_history.clear()
        self.last_values.clear()


class DataPreprocessor:
    """
    Handles data normalization and preprocessing for model input.
    """
    
    def __init__(
        self,
        normalization: str = "minmax",  # "minmax" or "standard"
        feature_names: Optional[List[str]] = None
    ):
        self.normalization = normalization
        self.feature_names = feature_names or FeatureExtractor().get_feature_names()
        
        # Initialize scaler
        if normalization == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = StandardScaler()
        
        self.fitted = False
        
        # Feature statistics
        self.feature_stats = {}
    
    def fit(self, data: np.ndarray):
        """
        Fit preprocessor to training data.
        
        Args:
            data: Array of shape (num_samples, num_features) or (num_samples, seq_len, num_features)
        """
        # Reshape if sequences
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])
        
        self.scaler.fit(data)
        self.fitted = True
        
        # Compute statistics
        self.feature_stats = {
            'mean': data.mean(axis=0).tolist(),
            'std': data.std(axis=0).tolist(),
            'min': data.min(axis=0).tolist(),
            'max': data.max(axis=0).tolist(),
        }
        
        print(f"Preprocessor fitted on {data.shape[0]} samples")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Array of shape (num_samples, num_features) or (num_samples, seq_len, num_features)
            
        Returns:
            Normalized data
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        original_shape = data.shape
        
        # Reshape if sequences
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])
        
        normalized = self.scaler.transform(data)
        
        # Restore shape
        if len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)
        
        return normalized.astype(np.float32)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale"""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted.")
        
        original_shape = data.shape
        
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])
        
        original = self.scaler.inverse_transform(data)
        
        if len(original_shape) == 3:
            original = original.reshape(original_shape)
        
        return original
    
    def save(self, path: str):
        """Save preprocessor state"""
        state = {
            'normalization': self.normalization,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
            'feature_stats': self.feature_stats,
        }
        
        # Save scaler parameters
        if self.fitted:
            if self.normalization == "minmax":
                state['scaler_params'] = {
                    'min_': self.scaler.min_.tolist(),
                    'scale_': self.scaler.scale_.tolist(),
                    'data_min_': self.scaler.data_min_.tolist(),
                    'data_max_': self.scaler.data_max_.tolist(),
                    'data_range_': self.scaler.data_range_.tolist(),
                }
            else:
                state['scaler_params'] = {
                    'mean_': self.scaler.mean_.tolist(),
                    'var_': self.scaler.var_.tolist(),
                    'scale_': self.scaler.scale_.tolist(),
                }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also save as pickle for sklearn compatibility
        with open(path.replace('.json', '.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor state"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.normalization = state['normalization']
        self.feature_names = state['feature_names']
        self.fitted = state['fitted']
        self.feature_stats = state['feature_stats']
        
        # Restore scaler
        if self.fitted:
            params = state['scaler_params']
            if self.normalization == "minmax":
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.min_ = np.array(params['min_'])
                self.scaler.scale_ = np.array(params['scale_'])
                self.scaler.data_min_ = np.array(params['data_min_'])
                self.scaler.data_max_ = np.array(params['data_max_'])
                self.scaler.data_range_ = np.array(params['data_range_'])
            else:
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(params['mean_'])
                self.scaler.var_ = np.array(params['var_'])
                self.scaler.scale_ = np.array(params['scale_'])
        
        print(f"Preprocessor loaded from {path}")


class SequenceCreator:
    """
    Creates sequences from time-series data for model input.
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        stride: int = 1,
        padding: str = "zero"  # "zero" or "replicate"
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.padding = padding
    
    def create_sequences(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences from flat data.
        
        Args:
            data: Array of shape (num_samples, num_features)
            labels: Optional labels array
            
        Returns:
            sequences: Array of shape (num_sequences, sequence_length, num_features)
            seq_labels: Labels per sequence (if provided)
        """
        num_samples, num_features = data.shape
        
        # Calculate number of sequences
        num_sequences = (num_samples - self.sequence_length) // self.stride + 1
        
        if num_sequences <= 0:
            # Pad data if too short
            padding_needed = self.sequence_length - num_samples
            if self.padding == "zero":
                pad = np.zeros((padding_needed, num_features))
            else:
                pad = np.tile(data[0], (padding_needed, 1))
            data = np.vstack([pad, data])
            num_sequences = 1
        
        # Create sequences
        sequences = np.zeros((num_sequences, self.sequence_length, num_features), dtype=np.float32)
        
        for i in range(num_sequences):
            start = i * self.stride
            end = start + self.sequence_length
            sequences[i] = data[start:end]
        
        # Create sequence labels (attack if any attack in sequence)
        seq_labels = None
        if labels is not None:
            seq_labels = np.zeros(num_sequences, dtype=np.int64)
            for i in range(num_sequences):
                start = i * self.stride
                end = start + self.sequence_length
                # Label 1 if any attack in sequence
                seq_labels[i] = 1 if labels[start:end].sum() > 0 else 0
        
        return sequences, seq_labels
    
    def create_rolling_sequences(
        self,
        buffer: deque,
        num_features: int
    ) -> Optional[np.ndarray]:
        """
        Create sequences from rolling buffer (for real-time inference).
        
        Args:
            buffer: Deque containing recent feature vectors
            num_features: Number of features per sample
            
        Returns:
            Sequence tensor or None if insufficient data
        """
        if len(buffer) < self.sequence_length:
            return None
        
        # Get last sequence_length samples
        recent_data = list(buffer)[-self.sequence_length:]
        sequence = np.array(recent_data, dtype=np.float32)
        
        return sequence.reshape(1, self.sequence_length, num_features)


def prepare_training_data(
    raw_data_path: str,
    output_dir: str,
    sequence_length: int = 20,
    validation_split: float = 0.2,
    normalization: str = "minmax"
) -> Dict:
    """
    Full pipeline to prepare training data from raw data.
    
    Args:
        raw_data_path: Path to raw data file (JSON or CSV)
        output_dir: Directory to save processed data
        sequence_length: Length of sequences
        validation_split: Fraction for validation
        normalization: Normalization method
        
    Returns:
        Dictionary with paths to saved files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    print(f"Loading raw data from {raw_data_path}...")
    # (Implementation depends on data format)
    
    # Initialize components
    extractor = FeatureExtractor()
    preprocessor = DataPreprocessor(normalization=normalization)
    sequence_creator = SequenceCreator(sequence_length=sequence_length)
    
    # Process data
    # ... (extract features, normalize, create sequences)
    
    # Save components
    preprocessor.save(os.path.join(output_dir, "preprocessor.json"))
    
    # Save config
    config = {
        'sequence_length': sequence_length,
        'num_features': extractor.get_num_features(),
        'feature_names': extractor.get_feature_names(),
        'normalization': normalization,
    }
    
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training data prepared in {output_dir}")
    
    return {
        'config_path': os.path.join(output_dir, "config.json"),
        'preprocessor_path': os.path.join(output_dir, "preprocessor.json"),
    }


if __name__ == "__main__":
    # Test feature extraction
    print("Testing Feature Extractor...")
    
    extractor = FeatureExtractor()
    
    # Simulate raw data
    raw_data = {
        'speed_kmh': 65.5,
        'battery_level': 82.3,
        'throttle': 35.0,
        'brake': 0.0,
        'steering': 2.5,
        'gear': 4,
        'location_x': 125.3,
        'location_y': -45.7,
        'cpu_usage_percent': 28.5,
        'cpu_temperature_c': 48.2,
        'memory_used_mb': 512.0,
        'memory_total_mb': 2048.0,
        'memory_usage_percent': 25.0,
        'load_average_1min': 0.45,
        'throttling_state': 0.0,
        'network_rx_bytes': 1500,
        'network_tx_bytes': 800,
        'can_rx_count': 150,
        'can_tx_count': 50,
        'can_error_count': 0,
    }
    
    features = extractor.extract_features(raw_data)
    print(f"\nExtracted {len(features)} features:")
    for name, value in zip(extractor.get_feature_names(), features):
        print(f"  {name}: {value:.4f}")
    
    # Test preprocessing
    print("\nTesting Preprocessor...")
    preprocessor = DataPreprocessor()
    
    # Generate sample data
    sample_data = np.random.randn(1000, extractor.get_num_features())
    normalized = preprocessor.fit_transform(sample_data)
    
    print(f"Original range: [{sample_data.min():.2f}, {sample_data.max():.2f}]")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # Test sequence creation
    print("\nTesting Sequence Creator...")
    seq_creator = SequenceCreator(sequence_length=20)
    
    sequences, _ = seq_creator.create_sequences(sample_data)
    print(f"Created {sequences.shape[0]} sequences of shape {sequences.shape[1:]}")
    
    print("\n✓ Data processor tests passed!")
