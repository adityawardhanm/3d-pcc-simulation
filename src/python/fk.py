# fk.py
"""Forward kinematics for soft robotic actuators using PCC model."""

# Standard imports
import math

class SegmentParams:
    """Parameters defining a soft robotic segment.

    Args:
        length (float): Length of the segment (m).
        out_radius (float): Outer radius of the segment (m).
        wall_thickness (float): Wall thickness of the segment (m).

    Attributes:
        I (float): Second moment of area (m^4).
        EI (float): Flexural rigidity (N.m^2).
        M (float): Bending moment (N.m).
        K (float): Curvature (1/m).
    """

    def __init__(self, length, out_radius, wall_thickness):
        self.length = length
        self.out_radius = out_radius
        self.wall_thickness = wall_thickness
        self.I = None
        self.EI = None
        self.M = None
        self.K = None


class MaterialModel:
    """Parameters defining a material model.

    Args:
        name (str): Name of the material.
        youngs_modulus (float): Young's modulus (Pa).
        density (float): Density (kg/m^3).
    """

    def __init__(self, name, youngs_modulus, density):
        self.name = name
        self.youngs_modulus = youngs_modulus
        self.density = density

    def info(self):
        """Return a string with material information."""
        return (
            f"Material: {self.name}, "
            f"Young's Modulus: {self.youngs_modulus} Pa, "
            f"Density: {self.density} kg/m^3"
        )


class MathModel:
    """Base class for mathematical material models."""

    def __init__(self, name):
        self.name = name


class NeoHookean(MathModel):
    """Neo-Hookean hyperelastic material model.

    Args:
        mu (float): Shear modulus (Pa).
    """

    def __init__(self, mu):
        super().__init__("Neo-Hookean")
        self._mu = mu

    @property
    def mu(self):
        """float: Initial shear modulus (Pa)."""
        return self._mu


class MooneyRivlin(MathModel):
    """Mooney-Rivlin hyperelastic material model.

    Args:
        c1 (float): Mooney-Rivlin constant C1 (Pa).
        c2 (float): Mooney-Rivlin constant C2 (Pa).
    """

    def __init__(self, c1, c2):
        super().__init__("Mooney-Rivlin")
        self.c1 = c1
        self.c2 = c2

    @property
    def mu(self):
        """float: Initial shear modulus μ = 2(C1 + C2) (Pa)."""
        return 2 * (self.c1 + self.c2)


class Ogden(MathModel):
    """Ogden hyperelastic material model.

    Args:
        alpha (float): Ogden exponent (dimensionless).
        mu_input (float): Initial shear modulus (Pa).
    """

    def __init__(self, alpha, mu_input):
        super().__init__("Ogden")
        self.alpha = alpha
        self.mu_input = mu_input

    @property
    def mu(self):
        """float: Initial shear modulus (Pa)."""
        return self.mu_input


class TangentModulus:
    """Static methods to compute tangent modulus for hyperelastic models."""

    @staticmethod
    def neohookean(mu, e):
        """Compute tangent modulus for Neo-Hookean model.

        Args:
            mu (float): Shear modulus (Pa).
            e (float): Axial strain (dimensionless).

        Returns:
            float: Tangent modulus (Pa).
        """
        lam = 1 + e
        return mu * (2 * lam + lam**-2)

    @staticmethod
    def mooney_rivlin(c1, c2, e):
        """Compute tangent modulus for Mooney-Rivlin model.

        Args:
            c1 (float): Mooney-Rivlin constant C1 (Pa).
            c2 (float): Mooney-Rivlin constant C2 (Pa).
            e (float): Axial strain (dimensionless).

        Returns:
            float: Tangent modulus (Pa).
        """
        lam = 1 + e
        term1 = 2 * (-c2 * lam**-2) * (lam**2 - lam**-1)
        term2 = (c2 + c1 * lam**-1) * (2 * lam + lam**-2)
        return 2 * (term1 + term2)

    @staticmethod
    def ogden(mu, alpha, e):
        """Compute tangent modulus for Ogden model.

        Args:
            mu (float): Initial shear modulus (Pa).
            alpha (float): Ogden exponent (dimensionless).
            e (float): Axial strain (dimensionless).

        Returns:
            float: Tangent modulus (Pa).
        """
        lam = 1 + e
        return mu * (alpha * lam**(alpha - 1) + (alpha / 2) * lam**(-alpha / 2 - 1))


def compute_second_moment(outer_radius, wall_thickness):
    """Compute second moment of area for hollow circular cross-section.

    Args:
        outer_radius (float): Outer radius of the segment (m).
        wall_thickness (float): Wall thickness of the segment (m).

    Returns:
        float: Second moment of area (m^4).
    """
    inner_radius = outer_radius - wall_thickness
    return math.pi / 4 * (outer_radius**4 - inner_radius**4)


def flexural_rigidity(e_tan, second_moment):
    """Compute flexural rigidity EI.

    Args:
        e_tan (float): Tangent modulus (Pa).
        second_moment (float): Second moment of area (m^4).

    Returns:
        float: Flexural rigidity (N.m^2).
    """
    return second_moment * e_tan


def channel_area(channel_radius):
    """Compute cross-sectional area of one quarter-circle channel.

    Args:
        channel_radius (float): Radius of the channel (m).

    Returns:
        float: Cross-sectional area of one channel (m^2).
    """
    return (math.pi * channel_radius**2) / 4


def centroid_distance(channel_radius, septum_thickness):
    """Compute distance from neutral axis to channel centroid.

    Args:
        channel_radius (float): Radius of the channel (m).
        septum_thickness (float): Thickness of the septum (m).

    Returns:
        float: Distance from neutral axis to channel centroid (m).
    """
    return (
        channel_radius
        + septum_thickness / 2
        + (4 * channel_radius) / (3 * math.pi)
    )


def directional_moment(p1, p2, area, centroid_dist):
    """Compute bending moment from pressure difference in opposing channels.

    Args:
        p1 (float): Pressure in channel 1 (Pa).
        p2 (float): Pressure in channel 2 (Pa).
        area (float): Cross-sectional area of the channel (m^2).
        centroid_dist (float): Distance to channel centroid (m).

    Returns:
        float: Bending moment (N.m).
    """
    delta_p = p1 - p2
    return delta_p * area * centroid_dist


def resultant_moment(m_ac, m_bd):
    """Compute resultant bending moment from orthogonal moments.

    Args:
        m_ac (float): Bending moment from channels A and C (N.m).
        m_bd (float): Bending moment from channels B and D (N.m).

    Returns:
        float: Resultant bending moment (N.m).
    """
    return math.sqrt(m_ac**2 + m_bd**2)


def bending_plane_angle(m_ac, m_bd):
    """Compute bending plane angle from orthogonal moments.

    Args:
        m_ac (float): Bending moment from channels A and C (N.m).
        m_bd (float): Bending moment from channels B and D (N.m).

    Returns:
        float: Bending plane angle (rad).
    """
    if m_ac == 0 and m_bd == 0:
        return 0.0
    return math.atan2(m_bd, m_ac)


def curvature(m_res, ei):
    """Compute curvature from moment and flexural rigidity.

    Args:
        m_res (float): Resultant bending moment (N.m).
        ei (float): Flexural rigidity (N.m^2).

    Returns:
        float: Curvature (1/m).
    """
    return m_res / ei


def prestrained_length(original_length, epsilon_pre):
    """Compute pre-strained length of the segment.

    Args:
        original_length (float): Original length of the segment (m).
        epsilon_pre (float): Pre-strain (dimensionless).

    Returns:
        float: Pre-strained length (m).
    """
    return original_length * (1 + epsilon_pre)


def arc_angle(kappa, length_pre):
    """Compute arc angle of the bent segment.

    Args:
        kappa (float): Curvature (1/m).
        length_pre (float): Pre-strained length (m).

    Returns:
        float: Arc angle (rad).
    """
    return kappa * length_pre