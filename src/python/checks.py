# checks.py
"""Validation checks for soft robot simulation parameters."""

# Standard imports
import numpy as np


def check_incompressibility(mu, bulk_modulus):
    """Check material incompressibility assumption (E ≈ 3μ).

    For incompressible materials, E = 3μ should hold within 2% error.

    Args:
        mu (float): Shear modulus (Pa).
        bulk_modulus (float): Bulk modulus (Pa).

    Returns:
        bool: True if incompressibility assumption is valid.
    """
    e_approx = 3 * mu
    e_full = 9 * bulk_modulus * mu / (3 * bulk_modulus + mu)
    return np.isclose(e_approx, e_full, rtol=2e-2)


def check_curvature_bounds(kappa):
    """Check if curvature is within physically reasonable bounds.

    Curvature should be between 0 and 100 rad/m for typical soft actuators.

    Args:
        kappa (float): Curvature (1/m).

    Returns:
        bool: True if curvature is within bounds.
    """
    return 0 <= kappa <= 100


def check_self_collision(segments, thetas, phi, epsilon_pre):
    """Check for geometric self-collision using 3D positions.

    Computes segment end positions using forward kinematics and checks
    if any non-adjacent segments are too close to each other.

    Args:
        segments (list): List of segment objects with out_radius, length.
        thetas (list): Arc angles (rad) for each segment.
        phi (float): Bending plane angle (rad).
        epsilon_pre (float): Pre-strain (dimensionless).

    Returns:
        tuple: (passed: bool, message: str).
    """
    # Compute segment end positions using forward kinematics
    positions = [np.array([0.0, 0.0, 0.0])]
    t_cumulative = np.eye(4)

    for seg, theta in zip(segments, thetas):
        kappa = theta / (seg.length * (1 + epsilon_pre))

        if abs(kappa) < 1e-6:
            # Straight segment
            t_seg = np.eye(4)
            t_seg[2, 3] = seg.length * (1 + epsilon_pre)
        else:
            # Curved segment
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            t_seg = np.array([
                [
                    cos_phi**2 + sin_phi**2 * cos_theta,
                    -sin_phi * cos_phi * (1 - cos_theta),
                    cos_phi * sin_theta,
                    cos_phi * (1 - cos_theta) / kappa
                ],
                [
                    -sin_phi * cos_phi * (1 - cos_theta),
                    sin_phi**2 + cos_phi**2 * cos_theta,
                    sin_phi * sin_theta,
                    sin_phi * (1 - cos_theta) / kappa
                ],
                [
                    -cos_phi * sin_theta,
                    -sin_phi * sin_theta,
                    cos_theta,
                    sin_theta / kappa
                ],
                [0, 0, 0, 1]
            ])

        t_cumulative = t_cumulative @ t_seg
        positions.append(t_cumulative[:3, 3].copy())

    positions = np.array(positions)

    # Check for self-collision: distance between non-adjacent segments
    min_distance = float('inf')

    for i in range(len(positions) - 1):
        for j in range(i + 2, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            safe_distance = (
                segments[i].out_radius
                + segments[j - 1].out_radius
                + 0.005  # 5mm safety margin
            )

            if dist < safe_distance:
                return False, (
                    f"!! Self-collision detected!\n"
                    f"   Segment {i + 1} and Segment {j} are too close\n"
                    f"   Distance: {dist * 1000:.1f} mm\n"
                    f"   Minimum safe distance: {safe_distance * 1000:.1f} mm\n"
                    f"   Position {i + 1}: ["
                    f"{positions[i][0] * 1000:.1f}, "
                    f"{positions[i][1] * 1000:.1f}, "
                    f"{positions[i][2] * 1000:.1f}] mm\n"
                    f"   Position {j}: ["
                    f"{positions[j][0] * 1000:.1f}, "
                    f"{positions[j][1] * 1000:.1f}, "
                    f"{positions[j][2] * 1000:.1f}] mm"
                )

            if dist < min_distance:
                min_distance = dist

    # Check for extreme bending (tip too close to base)
    total_arc_length = sum(seg.length * (1 + epsilon_pre) for seg in segments)
    tip_distance = np.linalg.norm(positions[-1])

    if tip_distance < 0.2 * total_arc_length:
        return False, (
            f"!! Extreme bending detected!\n"
            f"   Robot tip is very close to base\n"
            f"   Arc length: {total_arc_length * 1000:.1f} mm\n"
            f"   Tip distance: {tip_distance * 1000:.1f} mm\n"
            f"   Ratio: {(tip_distance / total_arc_length) * 100:.1f}%\n"
            f"   Risk: Likely self-collision or geometric instability"
        )

    return True, "All geometric self-collision checks passed."


def check_thin_wall_assumption(outer_radius, wall_thickness):
    """Check if thin-wall assumption is valid (t/r < 0.2).

    Args:
        outer_radius (float): Outer radius of segment (m).
        wall_thickness (float): Wall thickness of segment (m).

    Returns:
        bool: True if thin-wall assumption is valid.
    """
    return (wall_thickness / outer_radius) < 0.2