# pino/__init__.py
"""
Universal Physics-Informed Neural Operator (PINO) Package
for Soft Robot Inverse Kinematics
"""

from pathlib import Path
import sys

# Ensure package root is in path
_package_root = Path(__file__).parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

# Import main components
try:
    from .models.unipino import UniversalPINO, PINOLoss
    from .training.data_collector import PINODataCollector, PINODataset, TrainingSample
    from .training.train_pino import PINOTrainer, train_pino
except ImportError as e:
    print(f"Warning: Could not import all PINO components: {e}")

__version__ = "0.1.0"
__all__ = [
    'UniversalPINO',
    'PINOLoss',
    'PINODataCollector',
    'PINODataset',
    'TrainingSample',
    'PINOTrainer',
    'train_pino'
]
