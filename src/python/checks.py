# checks.py

import math 
import numpy as np

def test_condition_1(mu, youngs_modulus):
    """
    test_condition_1 : Incompressibility check for material models
    """

    E_approx = 3 * mu
    E_full = 9 * youngs_modulus * mu / (3 * youngs_modulus + mu)
    return np.isclose(E_approx, E_full, rtol=2e-2)

def test_condition_2(curvature):
    """
    test_condition_2 : Order of magnitude check for curvature
    """

    if 0 <= curvature <= 100:
        return True
    else:
        return False
    

def test_condition_3_geometric(segments, thetas, phi, epsilon_pre):
    """
    Geometric self-collision check using actual 3D positions.
    
    Args:
        segments: List of segment objects with out_radius, length
        thetas: Arc angles (radians) for each segment
        phi: Bending plane angle (radians) - same for all segments
        epsilon_pre: Pre-strain
    
    Returns:
        (bool, str): (passed, error_message)
    """
    import numpy as np
    
    # Compute segment end positions using forward kinematics
    positions = [np.array([0.0, 0.0, 0.0])]  # Start at origin
    
    # Build transformation matrices
    T = np.eye(4)
    
    for i, (seg, theta) in enumerate(zip(segments, thetas)):
        kappa = theta / (seg.length * (1 + epsilon_pre))
        
        if abs(kappa) < 1e-6:
            # Straight segment
            T_seg = np.eye(4)
            T_seg[2, 3] = seg.length * (1 + epsilon_pre)
        else:
            # Curved segment
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            T_seg = np.array([
                [cos_phi * cos_theta + sin_phi**2, -sin_phi * cos_phi * (1 - cos_theta), cos_phi * sin_theta, cos_phi * (1 - cos_theta) / kappa],
                [-sin_phi * cos_phi * (1 - cos_theta), sin_phi**2 + cos_phi**2 * cos_theta, sin_phi * sin_theta, sin_phi * (1 - cos_theta) / kappa],
                [-cos_phi * sin_theta, -sin_phi * sin_theta, cos_theta, sin_theta / kappa],
                [0, 0, 0, 1]
            ])
        
        T = T @ T_seg
        positions.append(T[:3, 3])
    
    positions = np.array(positions)
    
    # Check for self-collision: distance between non-adjacent segments
    min_distance = float('inf')
    collision_pair = None
    
    for i in range(len(positions) - 1):
        for j in range(i + 2, len(positions)):  # Skip adjacent segments
            # Distance between segment i and segment j endpoints
            dist = np.linalg.norm(positions[i] - positions[j])
            
            # Minimum safe distance = sum of radii + safety margin
            safe_distance = segments[i].out_radius + segments[j-1].out_radius + 0.005  # 5mm margin
            
            if dist < safe_distance:
                return False, (
                    f"!! Self-collision detected!\n"
                    f"   Segment {i+1} and Segment {j} are too close\n"
                    f"   Distance: {dist*1000:.1f} mm\n"
                    f"   Minimum safe distance: {safe_distance*1000:.1f} mm\n"
                    f"   Position {i+1}: [{positions[i][0]*1000:.1f}, {positions[i][1]*1000:.1f}, {positions[i][2]*1000:.1f}] mm\n"
                    f"   Position {j}: [{positions[j][0]*1000:.1f}, {positions[j][1]*1000:.1f}, {positions[j][2]*1000:.1f}] mm"
                )
            
            if dist < min_distance:
                min_distance = dist
                collision_pair = (i+1, j)
    
    # Also check total arc length vs straight-line distance (heuristic)
    total_arc_length = sum(seg.length * (1 + epsilon_pre) for seg in segments)
    tip_distance = np.linalg.norm(positions[-1])
    
    # If robot bends back more than 80%, warn
    if tip_distance < 0.2 * total_arc_length:
        return False, (
            f"!! Extreme bending detected!\n"
            f"   Robot tip is very close to base\n"
            f"   Arc length: {total_arc_length*1000:.1f} mm\n"
            f"   Tip distance: {tip_distance*1000:.1f} mm\n"
            f"   Ratio: {(tip_distance/total_arc_length)*100:.1f}%\n"
            f"   Risk: Likely self-collision or geometric instability"
        )
    
    return True, ("All geometric self-collision checks passed.")

def test_condition_4(outer_radius, wall_thickness):
    """
    test_condition_4 : thin wall assumption check
    """

    if (wall_thickness / outer_radius) < 0.2:
        return True
    else:
        return False