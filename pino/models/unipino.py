"""
Universal Physics-Informed Neural Operator (PINO) for Soft Robot Control
Fixed version with all missing components and proper error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer for FNO"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Complex weights for Fourier space
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )
    
    def compl_mul1d(self, input, weights):
        """
        Complex multiplication in Fourier space
        
        Args:
            input: [batch, in_channels, modes] (complex)
            weights: [in_channels, out_channels, modes, 2] (real representation)
        
        Returns:
            [batch, out_channels, modes] (complex)
        """
        # Convert weights to complex
        weights_complex = torch.view_as_complex(weights)  # [in_channels, out_channels, modes]
        
        # Perform batched complex multiplication
        # input: [batch, in_channels, modes]
        # weights: [in_channels, out_channels, modes]
        # output: [batch, out_channels, modes]
        return torch.einsum("bim,iom->bom", input, weights_complex)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch, channels, length] (real)
        
        Returns:
            [batch, channels, length] (real)
        """
        batch_size = x.shape[0]
        
        # Fourier transform (real input -> complex output)
        x_ft = torch.fft.rfft(x)  # [batch, in_channels, length//2 + 1]
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size, 
            self.out_channels, 
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat, 
            device=x.device
        )
        
        # Multiply only the low-frequency modes
        # KEY FIX: Slice BOTH input and output to match self.modes
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes],  # Slice input to first 'modes' frequencies
            self.weights
        )
        
        # Inverse Fourier transform back to spatial domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # [batch, out_channels, length]
        
        return x


class UncertaintyNetwork(nn.Module):
    """Estimates epistemic uncertainty in predictions"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeoHookeanPhysics:
    """Physics calculator for Neo-Hookean material model"""
    
    def compute_physics(self, pressures: torch.Tensor, config: Dict) -> torch.Tensor:
        """Compute physics parameters from pressures"""
        params = config['material_params']
        mu = params.get('mu', 0.5e6)
        epsilon_pre = params.get('epsilon_pre', 0.15)
        
        # Compute stretch ratio from pressure
        # Simplified: λ ≈ 1 + P/(2μ) for small deformations
        P_total = pressures.sum()
        stretch = 1 + P_total / (2 * mu) if mu > 0 else torch.tensor(1.0)
        
        # Compute strain energy density
        # W = μ/2 * (I1 - 3) for incompressible Neo-Hookean
        I1 = stretch**2 + 2/stretch  # Assuming incompressibility
        W = mu / 2 * (I1 - 3)
        
        return torch.stack([stretch, W]) if isinstance(stretch, torch.Tensor) else torch.tensor([float(stretch), float(W)])
    
    def compute_constraint_violation(self, pressures: torch.Tensor, 
                                     expected_physics: torch.Tensor,
                                     config: Dict) -> torch.Tensor:
        """Compute violation of physical constraints"""
        # Pressure must be non-negative
        pressure_violation = F.relu(-pressures).sum()
        
        # Maximum pressure constraint
        max_pressure = config.get('max_pressure', 100000)
        max_violation = F.relu(pressures - max_pressure).sum()
        
        return pressure_violation + max_violation


class MooneyRivlinPhysics:
    """Physics calculator for Mooney-Rivlin material model"""
    
    def compute_physics(self, pressures: torch.Tensor, config: Dict) -> torch.Tensor:
        """Compute physics parameters from pressures"""
        params = config['material_params']
        c1 = params.get('c1', 0.3e6)
        c2 = params.get('c2', 0.1e6)
        
        # Effective modulus
        mu_eff = 2 * (c1 + c2)
        
        P_total = pressures.sum()
        stretch = 1 + P_total / (2 * mu_eff) if mu_eff > 0 else torch.tensor(1.0)
        
        # Mooney-Rivlin strain energy
        I1 = stretch**2 + 2/stretch
        I2 = 2*stretch + 1/stretch**2
        W = c1 * (I1 - 3) + c2 * (I2 - 3)
        
        return torch.stack([stretch, W]) if isinstance(stretch, torch.Tensor) else torch.tensor([float(stretch), float(W)])
    
    def compute_constraint_violation(self, pressures: torch.Tensor,
                                     expected_physics: torch.Tensor,
                                     config: Dict) -> torch.Tensor:
        pressure_violation = F.relu(-pressures).sum()
        max_pressure = config.get('max_pressure', 100000)
        max_violation = F.relu(pressures - max_pressure).sum()
        return pressure_violation + max_violation


class OgdenPhysics:
    """Physics calculator for Ogden material model"""
    
    def compute_physics(self, pressures: torch.Tensor, config: Dict) -> torch.Tensor:
        """Compute physics parameters from pressures"""
        params = config['material_params']
        mu = params.get('mu', 0.5e6)
        alpha = params.get('alpha', 8.0)
        
        P_total = pressures.sum()
        stretch = 1 + P_total / (2 * mu) if mu > 0 else torch.tensor(1.0)
        
        # Ogden strain energy (single term)
        # W = (μ/α) * (λ^α + 2λ^(-α/2) - 3)
        if isinstance(stretch, torch.Tensor):
            W = (mu / alpha) * (stretch**alpha + 2*stretch**(-alpha/2) - 3)
        else:
            W = (mu / alpha) * (stretch**alpha + 2*stretch**(-alpha/2) - 3)
        
        return torch.stack([stretch, W]) if isinstance(stretch, torch.Tensor) else torch.tensor([float(stretch), float(W)])
    
    def compute_constraint_violation(self, pressures: torch.Tensor,
                                     expected_physics: torch.Tensor,
                                     config: Dict) -> torch.Tensor:
        pressure_violation = F.relu(-pressures).sum()
        max_pressure = config.get('max_pressure', 100000)
        max_violation = F.relu(pressures - max_pressure).sum()
        return pressure_violation + max_violation


class ConditionalEncoder(nn.Module):
    """Encodes target + configuration conditionally"""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        self.config_attention = nn.MultiheadAttention(
            latent_dim, num_heads=8, batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_attended, _ = self.config_attention(
            z.unsqueeze(1), z.unsqueeze(1), z.unsqueeze(1)
        )
        z = z + 0.1 * z_attended.squeeze(1)
        return z


class PhysicsEmbeddingNetwork(nn.Module):
    """Embeds physics equations into latent space"""
    def __init__(self, physics_dim: int = 64):
        super().__init__()
        self.physics_dim = physics_dim
        
        self.physics_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, physics_dim)
        )
        
    def forward(self, z: torch.Tensor, config_enc: torch.Tensor) -> torch.Tensor:
        return self.physics_predictor(z)


class UniversalPINO(nn.Module):
    """
    Universal Physics-Informed Neural Operator
    Handles 1-10 segments, multiple material models (Neo-Hookean, Mooney-Rivlin, Ogden)
    """
    def __init__(self, max_segments: int = 10, device: str = 'cpu'):
        super().__init__()
        self.max_segments = max_segments
        self.device = device
        
        # Input dimensions
        config_dim = self._get_config_encoding_dim()
        input_dim = 3 + 4 + config_dim  # target + pressures + config
        
        # Conditional Encoder
        self.conditional_encoder = ConditionalEncoder(input_dim, 512)
        
        # Segment-aware Fourier layers
        self.fno_layers = nn.ModuleDict({
            str(i): self._create_fno_block(i) for i in range(1, max_segments + 1)
        })
        
        # Material-aware decoders
        self.material_decoders = nn.ModuleDict({
            'neo-hookean': self._create_decoder(512, 4),
            'mooney-rivlin': self._create_decoder(512, 4),
            'ogden': self._create_decoder(512, 4)
        })
        
        # Physics embedding network
        self.physics_embedder = PhysicsEmbeddingNetwork()
        
        # Uncertainty estimator
        self.uncertainty_net = UncertaintyNetwork(512)
        
        # Move to device
        self.to(device)
        
    def _get_config_encoding_dim(self) -> int:
        """Calculate configuration encoding dimension"""
        # material_type(3) + num_segments(1) + material_params(3) 
        # + segment_params(max_segments*3) + channel_params(2)
        return 3 + 1 + 3 + (self.max_segments * 3) + 2
    
    def _create_fno_block(self, num_segments: int) -> nn.Module:
        """Create FNO block for specific number of segments"""
        modes = min(12, max(2, num_segments * 2))
        return nn.Sequential(
            SpectralConv1d(512, 512, modes=modes),
            nn.ReLU(),
            SpectralConv1d(512, 512, modes=modes),
            nn.ReLU(),
            SpectralConv1d(512, 512, modes=modes)
        )
    
    def _create_decoder(self, latent_dim: int, output_dim: int) -> nn.Module:
        """Create decoder for specific material model"""
        return nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
    
    def encode_configuration(self, config: Dict) -> torch.Tensor:
        """Encode robot configuration to tensor"""
        device = self.device
        
        # One-hot encode material model
        model_enc = torch.zeros(3, device=device)
        model_idx = {'neo-hookean': 0, 'mooney-rivlin': 1, 'ogden': 2}
        model_enc[model_idx.get(config['material_model'], 0)] = 1.0
        
        # Normalize segment count
        seg_enc = torch.tensor(
            [config['num_segments'] / self.max_segments],
            device=device
        )
        
        # Encode material parameters
        params = config.get('material_params', {})
        if config['material_model'] == 'neo-hookean':
            mat_enc = torch.tensor([
                params.get('mu', 0.5e6) / 1e6,
                params.get('epsilon_pre', 0.15),
                0.0
            ], device=device)
        elif config['material_model'] == 'mooney-rivlin':
            mat_enc = torch.tensor([
                params.get('c1', 0.3e6) / 1e6,
                params.get('c2', 0.1e6) / 1e6,
                params.get('epsilon_pre', 0.15)
            ], device=device)
        else:  # ogden
            mat_enc = torch.tensor([
                params.get('mu', 0.5e6) / 1e6,
                params.get('alpha', 8.0) / 10.0,
                params.get('epsilon_pre', 0.15)
            ], device=device)
        
        # Encode segment geometry
        geometry = config.get('geometry', {})
        segments = geometry.get('segments', [])
        
        lengths = torch.zeros(self.max_segments, device=device)
        radii = torch.zeros(self.max_segments, device=device)
        thicknesses = torch.zeros(self.max_segments, device=device)
        
        for i in range(min(config['num_segments'], self.max_segments)):
            if i < len(segments):
                lengths[i] = segments[i].get('length', 0.05) / 0.1  # Normalize
                radii[i] = segments[i].get('outer_radius', 0.01) / 0.05
                thicknesses[i] = segments[i].get('wall_thickness', 0.002) / 0.01
        
        # Channel parameters
        channel_enc = torch.tensor([
            geometry.get('channel_radius', 0.003) / 0.01,
            geometry.get('septum_thickness', 0.001) / 0.005
        ], device=device)
        
        # Combine all encodings
        config_enc = torch.cat([
            model_enc, seg_enc, mat_enc,
            lengths, radii, thicknesses, channel_enc
        ])
        
        return config_enc
    
    def forward(self, target: torch.Tensor, config: Union[Dict, List[Dict], torch.Tensor],
                teacher_pressures: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            target: [batch, 3] target position
            config: configuration dict, list of dicts, or encoded tensor
            teacher_pressures: optional [batch, 4] for teacher forcing
        
        Returns:
            pressures: [batch, 4]
            uncertainty: [batch, 1]
            physics_embedding: [batch, physics_dim]
        """
        batch_size = target.shape[0]
        
        # Encode configuration if needed
        if isinstance(config, dict):
            config_enc = self.encode_configuration(config).unsqueeze(0).expand(batch_size, -1)
            config_list = [config] * batch_size
        elif isinstance(config, list):
            config_enc = torch.stack([self.encode_configuration(c) for c in config])
            config_list = config
        else:
            config_enc = config
            config_list = None
        
        # Get number of segments and material model
        if config_list is not None:
            num_segments = config_list[0]['num_segments']
            material_model = config_list[0]['material_model']
        else:
            # Decode from tensor (approximate)
            num_segments = int(config_enc[0, 3].item() * self.max_segments)
            num_segments = max(1, min(num_segments, self.max_segments))
            material_idx = config_enc[0, :3].argmax().item()
            material_model = ['neo-hookean', 'mooney-rivlin', 'ogden'][material_idx]
        
        # Prepare input
        if teacher_pressures is not None:
            inputs = torch.cat([target, teacher_pressures, config_enc], dim=1)
        else:
            dummy_pressures = torch.zeros(batch_size, 4, device=target.device)
            inputs = torch.cat([target, dummy_pressures, config_enc], dim=1)
        
        # 1. Conditional encoding
        z = self.conditional_encoder(inputs)
        
        # 2. Apply segment-aware FNO
        z_reshaped = z.unsqueeze(2)  # [batch, latent, 1]
        fno_key = str(min(num_segments, self.max_segments))
        z_reshaped = self.fno_layers[fno_key](z_reshaped)
        z_out = z_reshaped.squeeze(2)
        
        # 3. Physics embedding
        physics_embedding = self.physics_embedder(z_out, config_enc)
        
        # 4. Material-aware decoding
        pressures_normalized = self.material_decoders[material_model](z_out)
        pressures = pressures_normalized * 100000  # Scale to Pa
        
        # 5. Uncertainty estimation
        uncertainty = self.uncertainty_net(z_out)
        
        return pressures, uncertainty, physics_embedding
    
    def predict(self, target: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple prediction interface for inference
        
        Args:
            target: [batch, 3] or [3] target position
            config: robot configuration dict
        
        Returns:
            pressures: [batch, 4] predicted pressures
            uncertainty: [batch, 1] prediction uncertainty
        """
        self.eval()
        with torch.no_grad():
            if target.dim() == 1:
                target = target.unsqueeze(0)
            
            target = target.to(self.device)
            pressures, uncertainty, _ = self.forward(target, config)
            
        return pressures, uncertainty


class MultiMaterialPhysicsLoss(nn.Module):
    """Physics loss that works for all material models"""
    def __init__(self):
        super().__init__()
        
        self.physics_calculators = {
            'neo-hookean': NeoHookeanPhysics(),
            'mooney-rivlin': MooneyRivlinPhysics(),
            'ogden': OgdenPhysics()
        }
        
    def forward(self, predictions: Tuple, ground_truth: Optional[Dict],
                configs: List[Dict]) -> torch.Tensor:
        """Compute physics loss for batch with mixed materials"""
        pressures, physics_embedding = predictions[:2]
        batch_size = pressures.shape[0]
        
        total_physics_loss = torch.tensor(0.0, device=pressures.device)
        total_constraint_loss = torch.tensor(0.0, device=pressures.device)
        
        for i in range(batch_size):
            config = configs[i] if isinstance(configs, list) else configs
            material_model = config['material_model']
            
            calculator = self.physics_calculators[material_model]
            
            expected_physics = calculator.compute_physics(pressures[i], config)
            
            if ground_truth is not None and 'physics_embedding' in ground_truth:
                physics_loss = F.mse_loss(
                    physics_embedding[i],
                    ground_truth['physics_embedding'][i]
                )
                total_physics_loss = total_physics_loss + physics_loss
            
            constraint_loss = calculator.compute_constraint_violation(
                pressures[i], expected_physics, config
            )
            total_constraint_loss = total_constraint_loss + constraint_loss
        
        avg_physics_loss = total_physics_loss / batch_size
        avg_constraint_loss = total_constraint_loss / batch_size
        
        return avg_physics_loss + 0.1 * avg_constraint_loss


class PINOLoss(nn.Module):
    """Combined loss function for PINO training"""
    def __init__(self, physics_weight: float = 0.1, uncertainty_weight: float = 0.01):
        super().__init__()
        self.physics_weight = physics_weight
        self.uncertainty_weight = uncertainty_weight
        self.physics_loss = MultiMaterialPhysicsLoss()
        
    def forward(self, predictions: Tuple, targets: torch.Tensor,
                ground_truth_pressures: torch.Tensor, configs: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            predictions: (pressures, uncertainty, physics_embedding)
            targets: target positions [batch, 3]
            ground_truth_pressures: true pressures [batch, 4]
            configs: list of configuration dicts
        
        Returns:
            Dict with 'total', 'mse', 'physics', 'uncertainty' losses
        """
        pressures, uncertainty, physics_embedding = predictions
        
        # MSE loss on pressures
        mse_loss = F.mse_loss(pressures, ground_truth_pressures)
        
        # Physics-informed loss
        physics_loss = self.physics_loss((pressures, physics_embedding), None, configs)
        
        # Uncertainty-aware loss (penalize high uncertainty when predictions are good)
        prediction_error = (pressures - ground_truth_pressures).pow(2).mean(dim=1, keepdim=True)
        uncertainty_loss = (uncertainty - prediction_error.sqrt()).pow(2).mean()
        
        total_loss = (
            mse_loss + 
            self.physics_weight * physics_loss + 
            self.uncertainty_weight * uncertainty_loss
        )
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'physics': physics_loss,
            'uncertainty': uncertainty_loss
        }