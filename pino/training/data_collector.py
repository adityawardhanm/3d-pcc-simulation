"""
Data Collector for PINO Training
Collects input-output pairs from soft robot simulation for training
"""

import torch
import numpy as np
import json
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single training sample"""
    target_position: np.ndarray      # [3] target x, y, z
    pressures: np.ndarray            # [4] channel pressures
    achieved_position: np.ndarray    # [3] actual achieved position
    config_encoding: np.ndarray      # Configuration encoding
    timestamp: float
    sample_id: str
    
    # Metadata
    num_segments: int
    material_model: str
    material_params: Dict
    
    def to_dict(self) -> Dict:
        return {
            'target_position': self.target_position.tolist(),
            'pressures': self.pressures.tolist(),
            'achieved_position': self.achieved_position.tolist(),
            'config_encoding': self.config_encoding.tolist(),
            'timestamp': self.timestamp,
            'sample_id': self.sample_id,
            'num_segments': self.num_segments,
            'material_model': self.material_model,
            'material_params': self.material_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingSample':
        return cls(
            target_position=np.array(data['target_position']),
            pressures=np.array(data['pressures']),
            achieved_position=np.array(data['achieved_position']),
            config_encoding=np.array(data['config_encoding']),
            timestamp=data['timestamp'],
            sample_id=data['sample_id'],
            num_segments=data['num_segments'],
            material_model=data['material_model'],
            material_params=data['material_params']
        )


class PINODataCollector:
    """
    Collects training data from soft robot simulation
    
    Usage:
        collector = PINODataCollector(save_dir='./training_data')
        collector.set_configuration(config)
        
        # During simulation/operation:
        collector.record_sample(
            target=target_position,
            pressures=applied_pressures,
            achieved=achieved_position
        )
        
        # Save periodically or at end:
        collector.save()
    """
    
    def __init__(self, save_dir: str = './pino_training_data',
                 buffer_size: int = 10000,
                 auto_save_interval: int = 1000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self.auto_save_interval = auto_save_interval
        
        self.samples: List[TrainingSample] = []
        self.current_config: Optional[Dict] = None
        self.config_encoding: Optional[np.ndarray] = None
        
        self.sample_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Thread-safe queue for async collection
        self.sample_queue = queue.Queue(maxsize=buffer_size)
        self.collector_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'samples_per_config': {},
            'position_range': {'min': None, 'max': None},
            'pressure_range': {'min': None, 'max': None}
        }
        
        logger.info(f"PINODataCollector initialized. Session: {self.session_id}")
    
    def set_configuration(self, config: Dict, config_encoder: Optional[Callable] = None):
        """
        Set current robot configuration
        
        Args:
            config: Dict with keys:
                - num_segments: int
                - material_model: str ('neo-hookean', 'mooney-rivlin', 'ogden')
                - material_params: Dict
                - geometry: Dict with segment parameters
            config_encoder: Optional function to encode config to tensor
        """
        self.current_config = config
        
        # Encode configuration
        if config_encoder is not None:
            self.config_encoding = config_encoder(config).numpy()
        else:
            self.config_encoding = self._default_encode_config(config)
        
        config_key = f"{config['num_segments']}seg_{config['material_model']}"
        if config_key not in self.stats['samples_per_config']:
            self.stats['samples_per_config'][config_key] = 0
        
        logger.info(f"Configuration set: {config_key}")
    
    def _default_encode_config(self, config: Dict, max_segments: int = 10) -> np.ndarray:
        """Default configuration encoding (matching UniversalPINO)"""
        # Material model one-hot
        model_enc = np.zeros(3)
        model_idx = {'neo-hookean': 0, 'mooney-rivlin': 1, 'ogden': 2}
        model_enc[model_idx.get(config['material_model'], 0)] = 1.0
        
        # Segment count normalized
        seg_enc = np.array([config['num_segments'] / max_segments])
        
        # Material parameters
        params = config.get('material_params', {})
        if config['material_model'] == 'neo-hookean':
            mat_enc = np.array([
                params.get('mu', 0.5e6) / 1e6,
                params.get('epsilon_pre', 0.15),
                0.0
            ])
        elif config['material_model'] == 'mooney-rivlin':
            mat_enc = np.array([
                params.get('c1', 0.3e6) / 1e6,
                params.get('c2', 0.1e6) / 1e6,
                params.get('epsilon_pre', 0.15)
            ])
        else:  # ogden
            mat_enc = np.array([
                params.get('mu', 0.5e6) / 1e6,
                params.get('alpha', 8.0) / 10.0,
                params.get('epsilon_pre', 0.15)
            ])
        
        # Geometry encoding
        geometry = config.get('geometry', {})
        segments = geometry.get('segments', [])
        
        lengths = np.zeros(max_segments)
        radii = np.zeros(max_segments)
        thicknesses = np.zeros(max_segments)
        
        for i in range(min(config['num_segments'], max_segments)):
            if i < len(segments):
                lengths[i] = segments[i].get('length', 0.05) / 0.1
                radii[i] = segments[i].get('outer_radius', 0.01) / 0.05
                thicknesses[i] = segments[i].get('wall_thickness', 0.002) / 0.01
        
        channel_enc = np.array([
            geometry.get('channel_radius', 0.003) / 0.01,
            geometry.get('septum_thickness', 0.001) / 0.005
        ])
        
        return np.concatenate([
            model_enc, seg_enc, mat_enc,
            lengths, radii, thicknesses, channel_enc
        ])
    
    def record_sample(self, target: np.ndarray, pressures: np.ndarray,
                      achieved: np.ndarray, async_mode: bool = False):
        """
        Record a single training sample
        
        Args:
            target: [3] target position (x, y, z) in meters
            pressures: [4] applied pressures in Pa
            achieved: [3] achieved position in meters
            async_mode: If True, add to queue for async processing
        """
        if self.current_config is None:
            logger.warning("No configuration set! Call set_configuration() first.")
            return
        
        target = np.asarray(target, dtype=np.float32)
        pressures = np.asarray(pressures, dtype=np.float32)
        achieved = np.asarray(achieved, dtype=np.float32)
        
        sample = TrainingSample(
            target_position=target,
            pressures=pressures,
            achieved_position=achieved,
            config_encoding=self.config_encoding.copy(),
            timestamp=datetime.now().timestamp(),
            sample_id=f"{self.session_id}_{self.sample_count:08d}",
            num_segments=self.current_config['num_segments'],
            material_model=self.current_config['material_model'],
            material_params=self.current_config.get('material_params', {})
        )
        
        if async_mode and self.running:
            try:
                self.sample_queue.put_nowait(sample)
            except queue.Full:
                logger.warning("Sample queue full, dropping sample")
        else:
            self._add_sample(sample)
    
    def _add_sample(self, sample: TrainingSample):
        """Add sample to buffer and update stats"""
        self.samples.append(sample)
        self.sample_count += 1
        self.stats['total_samples'] += 1
        
        config_key = f"{sample.num_segments}seg_{sample.material_model}"
        self.stats['samples_per_config'][config_key] = \
            self.stats['samples_per_config'].get(config_key, 0) + 1
        
        # Update ranges
        self._update_stats(sample)
        
        # Auto-save if needed
        if len(self.samples) >= self.auto_save_interval:
            self.save(incremental=True)
    
    def _update_stats(self, sample: TrainingSample):
        """Update statistics with new sample"""
        pos = sample.achieved_position
        press = sample.pressures
        
        if self.stats['position_range']['min'] is None:
            self.stats['position_range']['min'] = pos.copy()
            self.stats['position_range']['max'] = pos.copy()
            self.stats['pressure_range']['min'] = press.copy()
            self.stats['pressure_range']['max'] = press.copy()
        else:
            self.stats['position_range']['min'] = np.minimum(
                self.stats['position_range']['min'], pos
            )
            self.stats['position_range']['max'] = np.maximum(
                self.stats['position_range']['max'], pos
            )
            self.stats['pressure_range']['min'] = np.minimum(
                self.stats['pressure_range']['min'], press
            )
            self.stats['pressure_range']['max'] = np.maximum(
                self.stats['pressure_range']['max'], press
            )
    
    def start_async_collection(self):
        """Start background thread for async sample collection"""
        self.running = True
        self.collector_thread = threading.Thread(target=self._async_collector_loop)
        self.collector_thread.daemon = True
        self.collector_thread.start()
        logger.info("Async collection started")
    
    def stop_async_collection(self):
        """Stop async collection and process remaining samples"""
        self.running = False
        if self.collector_thread is not None:
            self.collector_thread.join(timeout=5.0)
        
        # Process remaining queue items
        while not self.sample_queue.empty():
            try:
                sample = self.sample_queue.get_nowait()
                self._add_sample(sample)
            except queue.Empty:
                break
        
        logger.info("Async collection stopped")
    
    def _async_collector_loop(self):
        """Background loop for processing samples"""
        while self.running:
            try:
                sample = self.sample_queue.get(timeout=0.1)
                self._add_sample(sample)
            except queue.Empty:
                continue
    
    def save(self, incremental: bool = False, format: str = 'hdf5'):
        """
        Save collected samples
        
        Args:
            incremental: If True, append to existing file
            format: 'hdf5', 'json', or 'npz'
        """
        if not self.samples:
            logger.info("No samples to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'hdf5':
            self._save_hdf5(timestamp, incremental)
        elif format == 'json':
            self._save_json(timestamp)
        elif format == 'npz':
            self._save_npz(timestamp)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Clear buffer after save
        num_saved = len(self.samples)
        self.samples = []
        logger.info(f"Saved {num_saved} samples")
    
    def _save_hdf5(self, timestamp: str, incremental: bool):
        """Save to HDF5 format (efficient for large datasets)"""
        filename = self.save_dir / f"pino_data_{self.session_id}.h5"
        mode = 'a' if incremental and filename.exists() else 'w'
        
        with h5py.File(filename, mode) as f:
            # Create or get datasets
            if 'targets' not in f:
                n_samples = len(self.samples)
                config_dim = len(self.samples[0].config_encoding)
                
                f.create_dataset('targets', shape=(n_samples, 3),
                                maxshape=(None, 3), dtype='float32')
                f.create_dataset('pressures', shape=(n_samples, 4),
                                maxshape=(None, 4), dtype='float32')
                f.create_dataset('achieved', shape=(n_samples, 3),
                                maxshape=(None, 3), dtype='float32')
                f.create_dataset('config_encodings', shape=(n_samples, config_dim),
                                maxshape=(None, config_dim), dtype='float32')
                f.create_dataset('timestamps', shape=(n_samples,),
                                maxshape=(None,), dtype='float64')
                
                f.attrs['session_id'] = self.session_id
                f.attrs['config_dim'] = config_dim
                start_idx = 0
            else:
                # Resize datasets
                current_size = f['targets'].shape[0]
                new_size = current_size + len(self.samples)
                
                for key in ['targets', 'pressures', 'achieved', 'config_encodings', 'timestamps']:
                    f[key].resize(new_size, axis=0)
                
                start_idx = current_size
            
            # Write data
            end_idx = start_idx + len(self.samples)
            f['targets'][start_idx:end_idx] = np.array([s.target_position for s in self.samples])
            f['pressures'][start_idx:end_idx] = np.array([s.pressures for s in self.samples])
            f['achieved'][start_idx:end_idx] = np.array([s.achieved_position for s in self.samples])
            f['config_encodings'][start_idx:end_idx] = np.array([s.config_encoding for s in self.samples])
            f['timestamps'][start_idx:end_idx] = np.array([s.timestamp for s in self.samples])
            
            # Update metadata
            f.attrs['total_samples'] = end_idx
        
        logger.info(f"Saved to {filename}")
    
    def _save_json(self, timestamp: str):
        """Save to JSON format (human-readable)"""
        filename = self.save_dir / f"pino_data_{self.session_id}_{timestamp}.json"
        
        data = {
            'session_id': self.session_id,
            'timestamp': timestamp,
            'stats': {
                'total_samples': self.stats['total_samples'],
                'samples_per_config': self.stats['samples_per_config']
            },
            'samples': [s.to_dict() for s in self.samples]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved to {filename}")
    
    def _save_npz(self, timestamp: str):
        """Save to NumPy compressed format"""
        filename = self.save_dir / f"pino_data_{self.session_id}_{timestamp}.npz"
        
        np.savez_compressed(
            filename,
            targets=np.array([s.target_position for s in self.samples]),
            pressures=np.array([s.pressures for s in self.samples]),
            achieved=np.array([s.achieved_position for s in self.samples]),
            config_encodings=np.array([s.config_encoding for s in self.samples]),
            timestamps=np.array([s.timestamp for s in self.samples])
        )
        
        logger.info(f"Saved to {filename}")
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        return {
            'session_id': self.session_id,
            'total_samples': self.stats['total_samples'],
            'buffered_samples': len(self.samples),
            'samples_per_config': self.stats['samples_per_config'],
            'position_range': {
                'min': self.stats['position_range']['min'].tolist() if self.stats['position_range']['min'] is not None else None,
                'max': self.stats['position_range']['max'].tolist() if self.stats['position_range']['max'] is not None else None
            },
            'pressure_range': {
                'min': self.stats['pressure_range']['min'].tolist() if self.stats['pressure_range']['min'] is not None else None,
                'max': self.stats['pressure_range']['max'].tolist() if self.stats['pressure_range']['max'] is not None else None
            }
        }


class PINODataset(torch.utils.data.Dataset):
    """PyTorch Dataset for PINO training data"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        """
        Load training data from file
        
        Args:
            data_path: Path to HDF5, JSON, or NPZ file
            normalize: Whether to normalize inputs
        """
        self.data_path = Path(data_path)
        self.normalize = normalize
        
        # Load data
        if self.data_path.suffix == '.h5':
            self._load_hdf5()
        elif self.data_path.suffix == '.json':
            self._load_json()
        elif self.data_path.suffix == '.npz':
            self._load_npz()
        else:
            raise ValueError(f"Unknown file format: {self.data_path.suffix}")
        
        # Compute normalization statistics
        if normalize:
            self._compute_normalization()
        
        logger.info(f"Loaded {len(self)} samples from {data_path}")
    
    def _load_hdf5(self):
        with h5py.File(self.data_path, 'r') as f:
            self.targets = torch.from_numpy(f['targets'][:]).float()
            self.pressures = torch.from_numpy(f['pressures'][:]).float()
            self.achieved = torch.from_numpy(f['achieved'][:]).float()
            self.config_encodings = torch.from_numpy(f['config_encodings'][:]).float()
    
    def _load_json(self):
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        samples = [TrainingSample.from_dict(s) for s in data['samples']]
        self.targets = torch.tensor([s.target_position for s in samples]).float()
        self.pressures = torch.tensor([s.pressures for s in samples]).float()
        self.achieved = torch.tensor([s.achieved_position for s in samples]).float()
        self.config_encodings = torch.tensor([s.config_encoding for s in samples]).float()
    
    def _load_npz(self):
        data = np.load(self.data_path)
        self.targets = torch.from_numpy(data['targets']).float()
        self.pressures = torch.from_numpy(data['pressures']).float()
        self.achieved = torch.from_numpy(data['achieved']).float()
        self.config_encodings = torch.from_numpy(data['config_encodings']).float()
    
    def _compute_normalization(self):
        """Compute normalization statistics"""
        self.target_mean = self.targets.mean(dim=0)
        self.target_std = self.targets.std(dim=0).clamp(min=1e-6)
        
        self.pressure_mean = self.pressures.mean(dim=0)
        self.pressure_std = self.pressures.std(dim=0).clamp(min=1e-6)
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target = self.targets[idx]
        pressure = self.pressures[idx]
        config_enc = self.config_encodings[idx]
        
        if self.normalize:
            target = (target - self.target_mean) / self.target_std
            pressure = (pressure - self.pressure_mean) / self.pressure_std
        
        return {
            'target': target,
            'pressure': pressure,
            'config_encoding': config_enc,
            'achieved': self.achieved[idx]
        }
    
    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """Get normalization statistics for inference"""
        if self.normalize:
            return {
                'target_mean': self.target_mean,
                'target_std': self.target_std,
                'pressure_mean': self.pressure_mean,
                'pressure_std': self.pressure_std
            }
        return {}


def collect_from_simulation(simulator, collector: PINODataCollector,
                           num_samples: int = 1000,
                           workspace_bounds: Tuple = ((-0.1, 0.1), (-0.1, 0.1), (0, 0.2)),
                           pressure_range: Tuple[float, float] = (0, 100000)):
    """
    Utility function to collect samples from simulation
    
    Args:
        simulator: Soft robot simulator with .solve(pressures) method
        collector: PINODataCollector instance
        num_samples: Number of samples to collect
        workspace_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        pressure_range: (min_pressure, max_pressure) in Pa
    """
    from tqdm import tqdm
    
    for i in tqdm(range(num_samples), desc="Collecting samples"):
        # Random target in workspace
        target = np.array([
            np.random.uniform(*workspace_bounds[0]),
            np.random.uniform(*workspace_bounds[1]),
            np.random.uniform(*workspace_bounds[2])
        ])
        
        # Random pressures
        pressures = np.random.uniform(
            pressure_range[0], pressure_range[1], size=4
        )
        
        # Simulate
        try:
            achieved = simulator.solve(pressures)
            
            # Record sample
            collector.record_sample(
                target=target,
                pressures=pressures,
                achieved=achieved
            )
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            continue
    
    collector.save()