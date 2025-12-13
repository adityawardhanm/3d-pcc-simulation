# pino/training/train_pino.py
"""
Training Script for Universal PINO
Supports multiple material models and segment configurations

Usage:
    python -m pino.training.train_pino --data path/to/data.h5 --output ./output
"""

import sys
from pathlib import Path

# Ensure proper imports work
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import json
import logging
from tqdm import tqdm

# Fixed imports - use absolute imports from package root
from pino.models.unipino import UniversalPINO, PINOLoss
from pino.training.data_collector import PINODataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PINOTrainer:
    """
    Trainer for Universal PINO model
    
    Features:
    - Multi-material training
    - Physics-informed loss
    - Teacher forcing with annealing
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    """
    
    def __init__(self, model: UniversalPINO,
                 train_dataset: PINODataset,
                 val_dataset: Optional[PINODataset] = None,
                 config: Optional[Dict] = None):
        """
        Initialize trainer
        
        Args:
            model: UniversalPINO model
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional, will split from train if not provided)
            config: Training configuration dict
        """
        self.model = model
        self.device = model.device
        
        # Default config
        self.config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 100,
            'val_split': 0.1,
            'patience': 20,
            'min_delta': 1e-6,
            'teacher_forcing_ratio': 0.5,
            'tf_decay_rate': 0.95,
            'tf_min_ratio': 0.1,
            'physics_weight': 0.1,
            'uncertainty_weight': 0.01,
            'gradient_clip': 1.0,
            'save_dir': './pino_checkpoints',
            'log_dir': './pino_logs'
        }
        if config is not None:
            self.config.update(config)
        
        # Split dataset if no validation set
        if val_dataset is None:
            val_size = int(len(train_dataset) * self.config['val_split'])
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device != 'cpu' else False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device != 'cpu' else False
        )
        
        # Loss function
        self.criterion = PINOLoss(
            physics_weight=self.config['physics_weight'],
            uncertainty_weight=self.config['uncertainty_weight']
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Setup directories
        self.save_dir = Path(self.config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(self.log_dir / run_name)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.teacher_forcing_ratio = self.config['teacher_forcing_ratio']
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'learning_rate': []
        }
        
        logger.info(f"Trainer initialized")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Device: {self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_mse = 0
        total_physics = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Get batch data
            targets = batch['target'].to(self.device)
            gt_pressures = batch['pressure'].to(self.device)
            config_enc = batch['config_encoding'].to(self.device)
            
            # Teacher forcing
            use_teacher_forcing = np.random.random() < self.teacher_forcing_ratio
            teacher_pressures = gt_pressures if use_teacher_forcing else None
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(targets, config_enc, teacher_pressures)
            
            # Create dummy configs for physics loss
            configs = self._decode_config_batch(config_enc)
            
            # Compute loss
            losses = self.criterion(predictions, targets, gt_pressures, configs)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            total_physics += losses['physics'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mse': f"{losses['mse'].item():.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'physics': total_physics / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        total_mse = 0
        total_physics = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                targets = batch['target'].to(self.device)
                gt_pressures = batch['pressure'].to(self.device)
                config_enc = batch['config_encoding'].to(self.device)
                
                # Forward pass (no teacher forcing during validation)
                predictions = self.model(targets, config_enc)
                
                # Create configs for loss
                configs = self._decode_config_batch(config_enc)
                
                # Compute loss
                losses = self.criterion(predictions, targets, gt_pressures, configs)
                
                total_loss += losses['total'].item()
                total_mse += losses['mse'].item()
                total_physics += losses['physics'].item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'physics': total_physics / num_batches
        }
    
    def _decode_config_batch(self, config_enc: torch.Tensor) -> List[Dict]:
        """Decode configuration encoding to config dicts"""
        configs = []
        for enc in config_enc:
            enc = enc.cpu().numpy()
            
            # Decode material model
            material_idx = enc[:3].argmax()
            material_model = ['neo-hookean', 'mooney-rivlin', 'ogden'][material_idx]
            
            # Decode segment count
            num_segments = int(round(enc[3] * self.model.max_segments))
            num_segments = max(1, min(num_segments, self.model.max_segments))
            
            # Decode material params
            if material_model == 'neo-hookean':
                material_params = {
                    'mu': enc[4] * 1e6,
                    'epsilon_pre': enc[5]
                }
            elif material_model == 'mooney-rivlin':
                material_params = {
                    'c1': enc[4] * 1e6,
                    'c2': enc[5] * 1e6,
                    'epsilon_pre': enc[6]
                }
            else:
                material_params = {
                    'mu': enc[4] * 1e6,
                    'alpha': enc[5] * 10.0,
                    'epsilon_pre': enc[6]
                }
            
            configs.append({
                'num_segments': num_segments,
                'material_model': material_model,
                'material_params': material_params,
                'max_pressure': 100000
            })
        
        return configs
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Main training loop
        
        Args:
            epochs: Number of epochs (uses config value if not provided)
        
        Returns:
            Training history dict
        """
        if epochs is None:
            epochs = self.config['epochs']
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Teacher forcing decay
            self.teacher_forcing_ratio = max(
                self.config['tf_min_ratio'],
                self.teacher_forcing_ratio * self.config['tf_decay_rate']
            )
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('MSE/train', train_metrics['mse'], epoch)
            self.writer.add_scalar('MSE/val', val_metrics['mse'], epoch)
            self.writer.add_scalar('Physics/train', train_metrics['physics'], epoch)
            self.writer.add_scalar('Physics/val', val_metrics['physics'], epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            self.writer.add_scalar('TeacherForcing', self.teacher_forcing_ratio, epoch)
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"lr={current_lr:.2e}, "
                f"tf_ratio={self.teacher_forcing_ratio:.2f}"
            )
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss - self.config['min_delta']:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt', is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        self.writer.close()
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'teacher_forcing_ratio': self.teacher_forcing_ratio,
            'config': self.config,
            'history': self.history
        }
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            logger.info(f"Saved best model: val_loss={self.best_val_loss:.4f}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.teacher_forcing_ratio = checkpoint['teacher_forcing_ratio']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def train_pino(data_path: str,
               output_dir: str = './pino_output',
               config: Optional[Dict] = None,
               device: str = 'auto') -> Tuple[UniversalPINO, Dict]:
    """
    Convenience function to train PINO model
    
    Args:
        data_path: Path to training data file
        output_dir: Directory for outputs
        config: Training configuration
        device: 'cuda', 'cpu', or 'auto'
    
    Returns:
        Trained model and training history
    """
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = PINODataset(data_path, normalize=True)
    
    # Create model
    model = UniversalPINO(max_segments=10, device=device)
    
    # Update config
    full_config = {
        'save_dir': f'{output_dir}/checkpoints',
        'log_dir': f'{output_dir}/logs'
    }
    if config is not None:
        full_config.update(config)
    
    # Create trainer
    trainer = PINOTrainer(model, dataset, config=full_config)
    
    # Train
    history = trainer.train()
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Save normalization stats
    norm_stats = dataset.get_normalization_stats()
    if norm_stats:
        torch.save(norm_stats, f'{output_dir}/checkpoints/normalization_stats.pt')
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Universal PINO')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to training data (HDF5, NPZ, or JSON)')
    parser.add_argument('--output', type=str, default='./pino_output', 
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }
    
    model, history = train_pino(
        data_path=args.data,
        output_dir=args.output,
        config=config,
        device=args.device
    )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Model saved to: {args.output}/checkpoints/best_model.pt")
    print(f"\nTo use the trained model:")
    print(f"  from pino.inference.ik_solver import RealTimeUniversalIK")
    print(f"  ik = RealTimeUniversalIK('{args.output}/checkpoints/best_model.pt')")
    print(f"{'='*60}")
