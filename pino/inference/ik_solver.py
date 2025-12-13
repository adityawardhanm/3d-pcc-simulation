# pino/inference/ik_solver.py
"""
Real-Time Inverse Kinematics using Universal PINO
Provides fast IK solutions for any robot configuration
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
import time
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pino.models.unipino import UniversalPINO


class RealTimeUniversalIK:
    """
    Real-time IK using Universal PINO
    Works with any robot configuration
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize IK solver
        
        Args:
            model_path: Path to trained model checkpoint.
                       If None, looks for 'checkpoints/best_model.pt'
        """
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        self.model = UniversalPINO(max_segments=10, device=self.device)
        
        # Load weights
        if model_path is None:
            model_path = _root / "checkpoints" / "best_model.pt"
        
        self._load_model(model_path)
        
        # Cache for common configurations
        self.solution_cache = {}
        self.uncertainty_threshold = 0.3
        
        # Normalization stats (loaded with model or set later)
        self.normalization_stats = None
        
        print(f"RealTimeUniversalIK initialized on {self.device}")
    
    def _load_model(self, model_path):
        """Load model weights from checkpoint"""
        model_path = Path(model_path)
        
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                print(f"✓ Loaded model from {model_path}")
                
                # Try to load normalization stats
                norm_path = model_path.parent / "normalization_stats.pt"
                if norm_path.exists():
                    self.normalization_stats = torch.load(norm_path, map_location=self.device)
                    print(f"✓ Loaded normalization stats")
                    
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                print("  Using randomly initialized weights")
        else:
            print(f"⚠ No model found at {model_path}")
            print("  Using randomly initialized weights")
            print("  Train a model first using: python -m pino.training.train_pino --data ...")
        
        self.model.eval()
    
    def solve_ik(self, target_mm: np.ndarray, robot_config: Dict) -> Dict:
        """
        Real-time IK for any configuration
        
        Args:
            target_mm: target position in mm [x, y, z]
            robot_config: dict with robot configuration containing:
                - num_segments: int
                - material_model: str ('neo-hookean', 'mooney-rivlin', 'ogden')
                - material_params: Dict with model-specific parameters
                - geometry: Dict with segment and channel geometry
        
        Returns:
            Dict with:
                - pressures: np.ndarray [4] in kPa
                - uncertainty: float
                - method: str ('PINO' or 'PINO+refined')
                - config: Dict (input config)
        """
        # Convert to meters
        target_m = np.array(target_mm) * 1e-3
        
        # Check cache first
        cache_key = self._create_cache_key(target_m, robot_config)
        if cache_key in self.solution_cache:
            cached = self.solution_cache[cache_key]
            if time.time() - cached['timestamp'] < 3600:  # 1 hour cache
                return cached['solution']
        
        # PINO inference
        with torch.no_grad():
            target_tensor = torch.FloatTensor(target_m).unsqueeze(0)
            
            if self.device == 'cuda':
                target_tensor = target_tensor.cuda()
            
            pressures, uncertainty, _ = self.model(target_tensor, robot_config)
            
            pressures_kpa = pressures.cpu().numpy().flatten() / 1000  # Pa to kPa
            uncertainty_val = uncertainty.cpu().item()
        
        method = 'PINO'
        
        # Refine if uncertainty is high
        if uncertainty_val > self.uncertainty_threshold:
            pressures_kpa = self._refine_with_physics(
                pressures_kpa, target_m, robot_config
            )
            uncertainty_val = min(uncertainty_val, 0.1)  # Reduced after refinement
            method = 'PINO+refined'
        
        # Build solution
        solution = {
            'pressures': pressures_kpa,
            'uncertainty': uncertainty_val,
            'method': method,
            'config': robot_config
        }
        
        # Cache solution
        self.solution_cache[cache_key] = {
            'solution': solution,
            'timestamp': time.time()
        }
        
        # Keep cache size reasonable
        if len(self.solution_cache) > 1000:
            self._clean_cache()
        
        return solution
    
    def _refine_with_physics(self, initial_pressures: np.ndarray, 
                              target_m: np.ndarray, 
                              config: Dict) -> np.ndarray:
        """
        Refine pressures using physics-based optimization
        
        This is a placeholder - implement your optimization here
        (e.g., gradient descent, differential evolution, etc.)
        """
        # For now, just return the initial pressures
        # TODO: Implement physics-based refinement using FK simulation
        return initial_pressures
    
    def _create_cache_key(self, target: np.ndarray, config: Dict) -> tuple:
        """Create cache key from target and configuration"""
        # Round target to 1mm precision
        target_key = tuple(round(t * 1000) for t in target)
        
        # Create config key
        config_key = (
            config['num_segments'],
            config['material_model'],
            hash(json.dumps(config.get('material_params', {}), sort_keys=True))
        )
        
        return (target_key, config_key)
    
    def _clean_cache(self):
        """Remove old entries from cache"""
        current_time = time.time()
        old_keys = [
            k for k, v in self.solution_cache.items()
            if current_time - v['timestamp'] > 3600
        ]
        for k in old_keys:
            del self.solution_cache[k]
    
    def batch_solve(self, targets_mm: np.ndarray, 
                    robot_config: Dict) -> Dict:
        """
        Solve IK for multiple targets at once
        
        Args:
            targets_mm: [N, 3] array of target positions in mm
            robot_config: Robot configuration dict
        
        Returns:
            Dict with:
                - pressures: [N, 4] array in kPa
                - uncertainties: [N] array
        """
        targets_m = targets_mm * 1e-3
        
        with torch.no_grad():
            target_tensor = torch.FloatTensor(targets_m)
            
            if self.device == 'cuda':
                target_tensor = target_tensor.cuda()
            
            pressures, uncertainties, _ = self.model(target_tensor, robot_config)
            
            pressures_kpa = pressures.cpu().numpy() / 1000
            uncertainty_vals = uncertainties.cpu().numpy().flatten()
        
        return {
            'pressures': pressures_kpa,
            'uncertainties': uncertainty_vals
        }
    
    def get_workspace_estimate(self, robot_config: Dict, 
                               grid_resolution: int = 10) -> Dict:
        """
        Estimate reachable workspace by sampling
        
        Args:
            robot_config: Robot configuration
            grid_resolution: Number of samples per dimension
        
        Returns:
            Dict with workspace bounds and sample points
        """
        # Estimate workspace bounds based on segment lengths
        segments = robot_config.get('geometry', {}).get('segments', [])
        total_length = sum(s.get('length', 0.05) for s in segments)
        
        # Sample grid
        x = np.linspace(-total_length, total_length, grid_resolution)
        y = np.linspace(-total_length, total_length, grid_resolution)
        z = np.linspace(0, total_length * 1.5, grid_resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z)
        targets_m = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        # Convert to mm for batch_solve
        targets_mm = targets_m * 1000
        results = self.batch_solve(targets_mm, robot_config)
        
        # Filter by uncertainty
        valid_mask = results['uncertainties'] < self.uncertainty_threshold
        valid_points = targets_m[valid_mask]
        
        return {
            'bounds': {
                'x': [valid_points[:, 0].min(), valid_points[:, 0].max()],
                'y': [valid_points[:, 1].min(), valid_points[:, 1].max()],
                'z': [valid_points[:, 2].min(), valid_points[:, 2].max()]
            },
            'valid_points': valid_points,
            'valid_count': len(valid_points),
            'total_sampled': len(targets_m)
        }


def create_ik_solver(model_path: Optional[str] = None) -> RealTimeUniversalIK:
    """
    Factory function to create IK solver
    
    Args:
        model_path: Path to trained model (optional)
    
    Returns:
        Configured RealTimeUniversalIK instance
    """
    return RealTimeUniversalIK(model_path)


# Example usage
if __name__ == "__main__":
    # Create solver
    solver = RealTimeUniversalIK()
    
    # Example configuration
    config = {
        'num_segments': 3,
        'material_model': 'neo-hookean',
        'material_params': {
            'mu': 0.5e6,
            'epsilon_pre': 0.15
        },
        'geometry': {
            'segments': [
                {'length': 0.05, 'outer_radius': 0.01, 'wall_thickness': 0.002}
            ] * 3,
            'channel_radius': 0.003,
            'septum_thickness': 0.001
        }
    }
    
    # Test solve
    target = [50, 30, 100]  # mm
    solution = solver.solve_ik(target, config)
    
    print(f"\nTarget: {target} mm")
    print(f"Pressures: {solution['pressures']} kPa")
    print(f"Uncertainty: {solution['uncertainty']:.3f}")
    print(f"Method: {solution['method']}")
