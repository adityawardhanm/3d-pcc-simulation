

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
import os
import copy

# System imports
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]  # parent_directory
print(f"Root path: {root}")
src_path = root / "src" / "python"
sys.path.append(str(src_path))
import fk
import checks
import ik

# Set plotting style for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class RobotConfig:
    """Configuration class for robot parameters"""
    
    def __init__(self, num_segments=2, segment_length=0.1, 
                 outer_radius=0.015, wall_thickness=0.003,
                 channel_radius=0.004, septum_thickness=0.002,
                 epsilon_pre=0.15, material_type='neohookean', mu=50e3):
        
        self.num_segments = num_segments
        self.params = {
            'channel_radius': channel_radius,
            'septum_thickness': septum_thickness,
            'epsilon_pre': epsilon_pre
        }
        
        # Create segments
        self.segments = []
        for i in range(num_segments):
            seg = fk.segment_params(segment_length, outer_radius, wall_thickness)
            seg.I = fk.compute_second_moment(outer_radius, wall_thickness)
            
            # Compute tangent modulus based on material
            if material_type == 'neohookean':
                E_tan = fk.tangent_modulus.neohookean(mu, epsilon_pre)
            elif material_type == 'mooney_rivlin':
                c1 = mu / 4
                c2 = mu / 4
                E_tan = fk.tangent_modulus.mooney_rivlin(c1, c2, epsilon_pre)
            elif material_type == 'ogden':
                alpha = 2.0
                E_tan = fk.tangent_modulus.ogden(mu, alpha, epsilon_pre)
            else:
                E_tan = 3 * mu  # Default to Neo-Hookean approximation
            
            seg.EI = fk.flexural_rigidity(E_tan, seg.I)
            self.segments.append(seg)
        
        self.material_type = material_type
        self.mu = mu


def workspace_analysis_3d(config, n_samples=30, pressure_range=(0, 300e3)):
    """
    3D workspace analysis by sampling pressure space
    """
    print("\n" + "="*60)
    print("3D WORKSPACE ANALYSIS")
    print("="*60)
    
    # Sample pressure space
    pressures_list = []
    positions = []
    
    p_min, p_max = pressure_range
    pressure_samples = np.linspace(p_min, p_max, n_samples)
    
    total_combinations = n_samples ** 4
    print(f"Sampling {n_samples}^4 = {total_combinations:,} pressure combinations...")
    print("This may take a while...")
    
    count = 0
    for pA in pressure_samples[::2]:  # Reduce sampling for speed
        for pB in pressure_samples[::2]:
            for pC in pressure_samples[::2]:
                for pD in pressure_samples[::2]:
                    pressures = np.array([pA, pB, pC, pD])
                    try:
                        pos = ik.forward_kinematics_vector(pressures, config.params, 
                                                          copy.deepcopy(config.segments))
                        positions.append(pos)
                        pressures_list.append(pressures)
                        count += 1
                    except:
                        pass
    
    positions = np.array(positions)
    print(f"Generated {len(positions):,} valid workspace points")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    
    # Main 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(positions[:, 0]*1000, positions[:, 1]*1000, positions[:, 2]*1000,
                         c=positions[:, 2]*1000, cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'3D Workspace ({config.num_segments}-segment robot)')
    plt.colorbar(scatter, ax=ax1, label='Z (mm)', shrink=0.8)
    
    # XY projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(positions[:, 0]*1000, positions[:, 1]*1000, c=positions[:, 2]*1000,
               cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # XZ projection
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(positions[:, 0]*1000, positions[:, 2]*1000, c=positions[:, 1]*1000,
               cmap='plasma', s=1, alpha=0.5)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection')
    ax3.grid(True, alpha=0.3)
    
    # YZ projection
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(positions[:, 1]*1000, positions[:, 2]*1000, c=positions[:, 0]*1000,
               cmap='plasma', s=1, alpha=0.5)
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('YZ Projection')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/workspace_3d_analysis.png", bbox_inches='tight')
    print(f"✓ Saved: workspace_3d_analysis.png")
    
    return positions


def reachability_heatmap(config, z_height=0.15, grid_resolution=50, 
                         pressure_range=(0, 300e3)):
    """
    2D reachability heatmap at a specific height
    """
    print("\n" + "="*60)
    print(f"REACHABILITY HEATMAP AT Z = {z_height*1000} mm")
    print("="*60)
    
    # Estimate workspace bounds
    pressure_bounds = np.array([[pressure_range[0]]*4, [pressure_range[1]]*4]).T
    bounds = ik.estimate_workspace_bounds(config.params, config.segments, pressure_bounds)
    
    max_reach = bounds['max_xy_reach']
    
    # Create grid
    x_range = np.linspace(-max_reach, max_reach, grid_resolution)
    y_range = np.linspace(-max_reach, max_reach, grid_resolution)
    
    reachability = np.zeros((grid_resolution, grid_resolution))
    error_map = np.full((grid_resolution, grid_resolution), np.nan)
    
    print(f"Testing {grid_resolution}x{grid_resolution} = {grid_resolution**2:,} points...")
    
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            target = np.array([x, y, z_height])
            
            # Try to reach this point with IK
            try:
                pressures, error, _ = ik.gradient_descent_ik(
                    target, config.params, copy.deepcopy(config.segments),
                    pressure_bounds, max_iterations=100, tolerance=1e-4, verbose=False
                )
                
                # Check if successfully reached
                if error < 0.005:  # 5mm tolerance
                    reachability[j, i] = 1  # Reachable
                    error_map[j, i] = error * 1000  # mm
                else:
                    reachability[j, i] = 0.5  # Partially reachable
                    error_map[j, i] = error * 1000
            except:
                reachability[j, i] = 0  # Not reachable
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reachability map
    im1 = ax1.contourf(x_range*1000, y_range*1000, reachability, 
                       levels=20, cmap='RdYlGn')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title(f'Reachability at Z = {z_height*1000} mm')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Reachability')
    ax1.grid(True, alpha=0.3)
    
    # Error map
    im2 = ax2.contourf(x_range*1000, y_range*1000, error_map, 
                       levels=20, cmap='viridis_r')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title(f'Position Error (mm) at Z = {z_height*1000} mm')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Error (mm)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reachability_heatmap.png", bbox_inches='tight')
    print(f"✓ Saved: reachability_heatmap.png")
    
    return reachability, error_map


def pressure_curvature_analysis(config, pressure_range=(0, 300e3), n_points=50):
    """
    Analyze relationship between pressure and curvature
    """
    print("\n" + "="*60)
    print("PRESSURE vs CURVATURE ANALYSIS")
    print("="*60)
    
    pressures = np.linspace(pressure_range[0], pressure_range[1], n_points)
    
    # Different pressure configurations
    configs = {
        'A-C Differential (B=D=0)': lambda p: np.array([p, 0, 0, 0]),
        'B-D Differential (A=C=0)': lambda p: np.array([0, p, 0, 0]),
        'Diagonal (A=B, C=D=0)': lambda p: np.array([p, p, 0, 0]),
        'Full Actuation (A=B=C=D)': lambda p: np.array([p, p, p, p])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (config_name, pressure_func) in enumerate(configs.items()):
        curvatures = []
        arc_angles = []
        tip_positions = []
        
        for p in pressures:
            p_vec = pressure_func(p)
            try:
                pos = ik.forward_kinematics_vector(p_vec, config.params,
                                                   copy.deepcopy(config.segments))
                
                # Compute curvature for first segment
                channel_area = fk.channel_area(config.params['channel_radius'])
                centroid_dist = fk.centroid_distance(config.params['channel_radius'],
                                                     config.params['septum_thickness'])
                
                M_ac = fk.directional_moment(p_vec[0], p_vec[2], channel_area, centroid_dist)
                M_bd = fk.directional_moment(p_vec[1], p_vec[3], channel_area, centroid_dist)
                M_res = fk.resultant_moment(M_ac, M_bd)
                
                kappa = fk.curvature(M_res, config.segments[0].EI)
                length_pre = fk.prestrained_length(config.segments[0].length, 
                                                   config.params['epsilon_pre'])
                theta = fk.arc_angle(kappa, length_pre)
                
                curvatures.append(kappa)
                arc_angles.append(np.degrees(theta))
                tip_positions.append(np.linalg.norm(pos[:2]))
            except:
                curvatures.append(np.nan)
                arc_angles.append(np.nan)
                tip_positions.append(np.nan)
        
        ax = axes[idx]
        ax.plot(pressures/1e3, curvatures, 'b-', linewidth=2, label='Curvature')
        ax.set_xlabel('Pressure (kPa)')
        ax.set_ylabel('Curvature (1/m)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_title(config_name)
        ax.grid(True, alpha=0.3)
        
        # Secondary axis for arc angle
        ax2 = ax.twinx()
        ax2.plot(pressures/1e3, arc_angles, 'r--', linewidth=2, label='Arc Angle')
        ax2.set_ylabel('Arc Angle (degrees)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pressure_curvature_analysis.png", bbox_inches='tight')
    print(f"✓ Saved: pressure_curvature_analysis.png")


def material_comparison(segment_length=0.1, outer_radius=0.015,
                       wall_thickness=0.003, epsilon_pre=0.15,
                       pressure_range=(0, 300e3), n_points=50):
    """
    Compare different material models
    """
    print("\n" + "="*60)
    print("MATERIAL MODEL COMPARISON")
    print("="*60)
    
    materials = {
        'Neo-Hookean (μ=50kPa)': ('neohookean', 50e3),
        'Neo-Hookean (μ=100kPa)': ('neohookean', 100e3),
        'Mooney-Rivlin (μ=50kPa)': ('mooney_rivlin', 50e3),
        'Ogden (μ=50kPa)': ('ogden', 50e3)
    }
    
    pressures = np.linspace(pressure_range[0], pressure_range[1], n_points)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    for mat_name, (mat_type, mu) in materials.items():
        tip_x = []
        tip_z = []
        curvatures = []
        
        for p in pressures:
            # Create configuration
            cfg = RobotConfig(num_segments=1, segment_length=segment_length,
                            outer_radius=outer_radius, wall_thickness=wall_thickness,
                            epsilon_pre=epsilon_pre, material_type=mat_type, mu=mu)
            
            p_vec = np.array([p, 0, 0, 0])  # A-C differential
            
            try:
                pos = ik.forward_kinematics_vector(p_vec, cfg.params,
                                                   copy.deepcopy(cfg.segments))
                
                # Compute curvature
                channel_area = fk.channel_area(cfg.params['channel_radius'])
                centroid_dist = fk.centroid_distance(cfg.params['channel_radius'],
                                                     cfg.params['septum_thickness'])
                M_ac = fk.directional_moment(p, 0, channel_area, centroid_dist)
                M_res = fk.resultant_moment(M_ac, 0)
                kappa = fk.curvature(M_res, cfg.segments[0].EI)
                
                tip_x.append(pos[0] * 1000)
                tip_z.append(pos[2] * 1000)
                curvatures.append(kappa)
            except:
                tip_x.append(np.nan)
                tip_z.append(np.nan)
                curvatures.append(np.nan)
        
        ax1.plot(pressures/1e3, curvatures, linewidth=2, label=mat_name, marker='o', markersize=3)
        ax2.plot(tip_x, tip_z, linewidth=2, label=mat_name, marker='s', markersize=3)
        ax3.plot(pressures/1e3, tip_x, linewidth=2, label=mat_name, marker='^', markersize=3)
    
    ax1.set_xlabel('Pressure (kPa)')
    ax1.set_ylabel('Curvature (1/m)')
    ax1.set_title('Pressure vs Curvature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Tip X (mm)')
    ax2.set_ylabel('Tip Z (mm)')
    ax2.set_title('Tip Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    ax3.set_xlabel('Pressure (kPa)')
    ax3.set_ylabel('Tip X Displacement (mm)')
    ax3.set_title('Lateral Deflection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/material_comparison.png", bbox_inches='tight')
    print(f"✓ Saved: material_comparison.png")


def prestrain_sensitivity(segment_length=0.1, n_points=50):
    """
    Analyze effect of pre-strain on robot behavior
    """
    print("\n" + "="*60)
    print("PRE-STRAIN SENSITIVITY ANALYSIS")
    print("="*60)
    
    prestrains = [0.0, 0.05, 0.10, 0.15, 0.20]
    pressures = np.linspace(0, 300e3, n_points)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for eps_pre in prestrains:
        tip_positions = []
        curvatures = []
        
        for p in pressures:
            cfg = RobotConfig(num_segments=1, segment_length=segment_length,
                            epsilon_pre=eps_pre, mu=50e3)
            
            p_vec = np.array([p, 0, 0, 0])
            
            try:
                pos = ik.forward_kinematics_vector(p_vec, cfg.params,
                                                   copy.deepcopy(cfg.segments))
                
                channel_area = fk.channel_area(cfg.params['channel_radius'])
                centroid_dist = fk.centroid_distance(cfg.params['channel_radius'],
                                                     cfg.params['septum_thickness'])
                M_ac = fk.directional_moment(p, 0, channel_area, centroid_dist)
                M_res = fk.resultant_moment(M_ac, 0)
                kappa = fk.curvature(M_res, cfg.segments[0].EI)
                
                tip_positions.append(np.linalg.norm(pos[:2]) * 1000)
                curvatures.append(kappa)
            except:
                tip_positions.append(np.nan)
                curvatures.append(np.nan)
        
        label = f'ε_pre = {eps_pre:.0%}'
        ax1.plot(pressures/1e3, curvatures, linewidth=2, label=label, marker='o', markersize=3)
        ax2.plot(pressures/1e3, tip_positions, linewidth=2, label=label, marker='s', markersize=3)
    
    ax1.set_xlabel('Pressure (kPa)')
    ax1.set_ylabel('Curvature (1/m)')
    ax1.set_title('Effect of Pre-strain on Curvature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Pressure (kPa)')
    ax2.set_ylabel('Lateral Tip Displacement (mm)')
    ax2.set_title('Effect of Pre-strain on Deflection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/prestrain_sensitivity.png", bbox_inches='tight')
    print(f"✓ Saved: prestrain_sensitivity.png")


def wall_thickness_study(n_points=50):
    """
    Study effect of wall thickness on bending characteristics
    """
    print("\n" + "="*60)
    print("WALL THICKNESS STUDY")
    print("="*60)
    
    thicknesses = [0.001, 0.002, 0.003, 0.004, 0.005]  # 1-5mm
    pressures = np.linspace(0, 300e3, n_points)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    for t in thicknesses:
        curvatures = []
        EI_values = []
        tip_deflections = []
        checks_passed = []
        
        for p in pressures:
            cfg = RobotConfig(num_segments=1, wall_thickness=t, mu=50e3)
            
            # Check thin-wall assumption
            thin_wall_ok = checks.test_condition_4(cfg.segments[0].out_radius, t)
            checks_passed.append(thin_wall_ok)
            
            p_vec = np.array([p, 0, 0, 0])
            
            try:
                pos = ik.forward_kinematics_vector(p_vec, cfg.params,
                                                   copy.deepcopy(cfg.segments))
                
                channel_area = fk.channel_area(cfg.params['channel_radius'])
                centroid_dist = fk.centroid_distance(cfg.params['channel_radius'],
                                                     cfg.params['septum_thickness'])
                M_ac = fk.directional_moment(p, 0, channel_area, centroid_dist)
                M_res = fk.resultant_moment(M_ac, 0)
                kappa = fk.curvature(M_res, cfg.segments[0].EI)
                
                curvatures.append(kappa)
                EI_values.append(cfg.segments[0].EI / 1e-6)  # Convert to N⋅mm²
                tip_deflections.append(np.linalg.norm(pos[:2]) * 1000)
            except:
                curvatures.append(np.nan)
                EI_values.append(np.nan)
                tip_deflections.append(np.nan)
        
        label = f't = {t*1000:.1f} mm'
        marker = 'o' if checks_passed[0] else 'x'
        
        ax1.plot(pressures/1e3, curvatures, linewidth=2, label=label, marker=marker, markersize=3)
        ax2.plot(pressures/1e3, tip_deflections, linewidth=2, label=label, marker=marker, markersize=3)
        ax3.axhline(y=EI_values[0], linewidth=2, label=label, linestyle='--')
        
        # Thin-wall check
        ratio = t / cfg.segments[0].out_radius
        ax4.scatter(t*1000, ratio, s=100, marker=marker, 
                   label=f't={t*1000:.1f}mm ({"✓" if thin_wall_ok else "✗"})')
    
    ax1.set_xlabel('Pressure (kPa)')
    ax1.set_ylabel('Curvature (1/m)')
    ax1.set_title('Curvature vs Wall Thickness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Pressure (kPa)')
    ax2.set_ylabel('Tip Deflection (mm)')
    ax2.set_title('Deflection vs Wall Thickness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Pressure (kPa)')
    ax3.set_ylabel('Flexural Rigidity (N⋅mm²)')
    ax3.set_title('Flexural Rigidity (EI)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.axhline(y=0.2, color='r', linestyle='--', linewidth=2, label='Thin-wall limit (t/r=0.2)')
    ax4.set_xlabel('Wall Thickness (mm)')
    ax4.set_ylabel('t/r Ratio')
    ax4.set_title('Thin-Wall Assumption Check')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/wall_thickness_study.png", bbox_inches='tight')
    print(f"✓ Saved: wall_thickness_study.png")


def multi_segment_comparison(pressure=200e3):
    """
    Compare workspace for different number of segments
    """
    print("\n" + "="*60)
    print("MULTI-SEGMENT COMPARISON")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 5))
    
    segment_counts = [1, 2, 3]
    
    for idx, n_seg in enumerate(segment_counts):
        cfg = RobotConfig(num_segments=n_seg, mu=50e3)
        
        # Sample bending plane angles
        phis = np.linspace(0, 2*np.pi, 36)
        
        positions = []
        for phi in phis:
            # Compute pressures for this bending angle
            pA = pressure * np.cos(phi) if np.cos(phi) > 0 else 0
            pB = pressure * np.sin(phi) if np.sin(phi) > 0 else 0
            pC = -pressure * np.cos(phi) if np.cos(phi) < 0 else 0
            pD = -pressure * np.sin(phi) if np.sin(phi) < 0 else 0
            
            p_vec = np.array([pA, pB, pC, pD])
            
            try:
                pos = ik.forward_kinematics_vector(p_vec, cfg.params,
                                                   copy.deepcopy(cfg.segments))
                positions.append(pos)
            except:
                pass
        
        positions = np.array(positions)
        
        # 3D plot
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        ax.plot(positions[:, 0]*1000, positions[:, 1]*1000, positions[:, 2]*1000,
               'b-', linewidth=2, marker='o', markersize=4)
        ax.scatter([0], [0], [0], color='red', s=100, marker='o', label='Base')
        ax.scatter(positions[-1, 0]*1000, positions[-1, 1]*1000, positions[-1, 2]*1000,
                  color='green', s=100, marker='^', label='Tip')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{n_seg}-Segment Robot\n(P = {pressure/1e3:.0f} kPa)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/multi_segment_comparison.png", bbox_inches='tight')
    print(f"✓ Saved: multi_segment_comparison.png")


def validation_checks_summary():
    """
    Create summary table of validation checks for different configurations
    """
    print("\n" + "="*60)
    print("VALIDATION CHECKS SUMMARY")
    print("="*60)
    
    configs_to_test = [
        ('Standard', {'wall_thickness': 0.003, 'outer_radius': 0.015, 'mu': 50e3}),
        ('Thick Wall', {'wall_thickness': 0.005, 'outer_radius': 0.015, 'mu': 50e3}),
        ('Stiff Material', {'wall_thickness': 0.003, 'outer_radius': 0.015, 'mu': 200e3}),
        ('Large Radius', {'wall_thickness': 0.003, 'outer_radius': 0.025, 'mu': 50e3}),
    ]
    
    results = []
    
    for config_name, params in configs_to_test:
        cfg = RobotConfig(**params)
        
        # Test pressure configuration
        pressures = np.array([200e3, 0, 0, 0])
        
        try:
            pos = ik.forward_kinematics_vector(pressures, cfg.params,
                                               copy.deepcopy(cfg.segments))
            
            # Check 1: Incompressibility
            E_tan = fk.tangent_modulus.neohookean(cfg.mu, cfg.params['epsilon_pre'])
            check1 = checks.test_condition_1(cfg.mu, E_tan/3)
            
            # Check 2: Curvature range
            channel_area = fk.channel_area(cfg.params['channel_radius'])
            centroid_dist = fk.centroid_distance(cfg.params['channel_radius'],
                                                cfg.params['septum_thickness'])
            M_ac = fk.directional_moment(200e3, 0, channel_area, centroid_dist)
            M_res = fk.resultant_moment(M_ac, 0)
            kappa = fk.curvature(M_res, cfg.segments[0].EI)
            check2 = checks.test_condition_2(kappa)
            
            # Check 3: Self-collision (simplified - just check tip doesn't fold back)
            check3 = pos[2] > 0
            
            # Check 4: Thin wall
            check4 = checks.test_condition_4(cfg.segments[0].out_radius, 
                                           cfg.segments[0].wall_thickness)
            
            results.append({
                'Configuration': config_name,
                'Check 1\n(Incomp.)': '✓' if check1 else '✗',
                'Check 2\n(Curv.)': '✓' if check2 else '✗',
                'Check 3\n(Geom.)': '✓' if check3 else '✗',
                'Check 4\n(Thin Wall)': '✓' if check4 else '✗',
                'Curvature\n(1/m)': f'{kappa:.2f}',
                't/r': f'{cfg.segments[0].wall_thickness/cfg.segments[0].out_radius:.3f}'
            })
        except Exception as e:
            results.append({
                'Configuration': config_name,
                'Check 1\n(Incomp.)': 'ERR',
                'Check 2\n(Curv.)': 'ERR',
                'Check 3\n(Geom.)': 'ERR',
                'Check 4\n(Thin Wall)': 'ERR',
                'Curvature\n(1/m)': 'N/A',
                't/r': 'N/A'
            })
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [[k for k in results[0].keys()]]
    for r in results:
        table_data.append([str(v) for v in r.values()])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15]*len(results[0]))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(results[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code check results
    for i in range(1, len(table_data)):
        for j in range(1, 5):  # Columns with check results
            cell = table[(i, j)]
            if '✓' in cell.get_text().get_text():
                cell.set_facecolor('#90EE90')
            elif '✗' in cell.get_text().get_text():
                cell.set_facecolor('#FFB6C1')
    
    plt.title('Validation Checks Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{OUTPUT_DIR}/validation_checks_summary.png", bbox_inches='tight')
    print(f"✓ Saved: validation_checks_summary.png")
    
    # Also print to console
    print("\nValidation Results:")
    for r in results:
        print(f"\n{r['Configuration']}:")
        for k, v in r.items():
            if k != 'Configuration':
                print(f"  {k.replace(chr(10), ' ')}: {v}")


def bending_plane_analysis(n_angles=12, pressure=200e3):
    """
    Analyze effect of bending plane angle
    """
    print("\n" + "="*60)
    print("BENDING PLANE ANGLE ANALYSIS")
    print("="*60)
    
    cfg = RobotConfig(num_segments=2, mu=50e3)
    
    phis = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D view
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Top view (XY)
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Side view (XZ)
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Polar plot
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    
    # Trajectory components
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    tip_positions = []
    
    for i, phi in enumerate(phis):
        # Compute pressures for this bending angle
        pA = pressure * np.cos(phi) if np.cos(phi) > 0 else 0
        pB = pressure * np.sin(phi) if np.sin(phi) > 0 else 0
        pC = -pressure * np.cos(phi) if np.cos(phi) < 0 else 0
        pD = -pressure * np.sin(phi) if np.sin(phi) < 0 else 0
        
        p_vec = np.array([pA, pB, pC, pD])
        
        try:
            pos = ik.forward_kinematics_vector(p_vec, cfg.params,
                                               copy.deepcopy(cfg.segments))
            tip_positions.append(pos)
            
            color = plt.cm.hsv(i / n_angles)
            
            # 3D view
            ax1.scatter(pos[0]*1000, pos[1]*1000, pos[2]*1000, 
                       color=color, s=100, marker='o')
            ax1.plot([0, pos[0]*1000], [0, pos[1]*1000], [0, pos[2]*1000],
                    color=color, alpha=0.3, linewidth=1)
            
            # Top view
            ax2.scatter(pos[0]*1000, pos[1]*1000, color=color, s=100, marker='o')
            ax2.plot([0, pos[0]*1000], [0, pos[1]*1000], color=color, alpha=0.3)
            
            # Side view
            ax3.scatter(pos[0]*1000, pos[2]*1000, color=color, s=100, marker='o')
            ax3.plot([0, pos[0]*1000], [0, pos[2]*1000], color=color, alpha=0.3)
            
            # Polar plot
            r = np.linalg.norm(pos[:2])
            ax4.scatter(phi, r*1000, color=color, s=100, marker='o')
            
        except:
            pass
    
    tip_positions = np.array(tip_positions)
    
    # 3D setup
    ax1.scatter([0], [0], [0], color='red', s=200, marker='o', label='Base')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'3D View (P={pressure/1e3:.0f} kPa)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top view setup
    ax2.scatter([0], [0], color='red', s=200, marker='o', label='Base')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View (XY Plane)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Side view setup
    ax3.scatter([0], [0], color='red', s=200, marker='o', label='Base')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('Side View (XZ Plane)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Polar setup
    ax4.set_title(f'Radial Reach vs φ\n(P={pressure/1e3:.0f} kPa)')
    ax4.set_ylabel('Reach (mm)', labelpad=30)
    
    # Component plots
    phi_degrees = np.degrees(phis)
    ax5.plot(phi_degrees, tip_positions[:, 0]*1000, 'b-o', linewidth=2, markersize=6, label='X')
    ax5.plot(phi_degrees, tip_positions[:, 1]*1000, 'r-s', linewidth=2, markersize=6, label='Y')
    ax5.set_xlabel('Bending Plane Angle φ (degrees)')
    ax5.set_ylabel('Position (mm)')
    ax5.set_title('Tip X,Y vs Bending Angle')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6.plot(phi_degrees, tip_positions[:, 2]*1000, 'g-^', linewidth=2, markersize=6)
    ax6.set_xlabel('Bending Plane Angle φ (degrees)')
    ax6.set_ylabel('Z Position (mm)')
    ax6.set_title('Tip Height vs Bending Angle')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bending_plane_analysis.png", bbox_inches='tight')
    print(f"✓ Saved: bending_plane_analysis.png")


def run_all_analyses():
    """
    Run all analyses
    """
    print("\n" + "="*70)
    print(" "*15 + "SOFT ROBOT ANALYSIS SUITE")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    # Create default configuration
    config = RobotConfig(num_segments=2, mu=50e3)
    
    try:
        # Run analyses
        print("\n[1/10] Running 3D workspace analysis...")
        workspace_analysis_3d(config, n_samples=15)
        
        print("\n[2/10] Running reachability heatmap...")
        reachability_heatmap(config, z_height=0.15, grid_resolution=30)
        
        print("\n[3/10] Running pressure-curvature analysis...")
        pressure_curvature_analysis(config)
        
        print("\n[4/10] Running material comparison...")
        material_comparison()
        
        print("\n[5/10] Running pre-strain sensitivity...")
        prestrain_sensitivity()
        
        print("\n[6/10] Running wall thickness study...")
        wall_thickness_study()
        
        print("\n[7/10] Running multi-segment comparison...")
        multi_segment_comparison()
        
        print("\n[8/10] Running validation checks...")
        validation_checks_summary()
        
        print("\n[9/10] Running bending plane analysis...")
        bending_plane_analysis()
        
        print("\n" + "="*70)
        print("✓ ALL ANALYSES COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png'):
                print(f"  - {f}")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_analyses()