# fk.py

# IMPORTS
import math
import numpy as np

class segment_params:
    def __init__(self, length, out_radius, wall_thickness):
        self.length = length                    # in meters
        self.out_radius = out_radius            # in meters
        self.wall_thickness = wall_thickness    # in meters
        self.I = None                           # Second moment of area
        self.EI = None                          # Flexural rigidity
        self.M = None                           # Bending moment 
        self.K = None                           # Curvature

class material_model:
    def __init__(self,name, youngs_modulus,density):
        self.name = name
        self.youngs_modulus = youngs_modulus    # in Pa
        self.density = density                  # in kg/m^3
        # self.poisson_ratio = poisson_ratio      # dimensionless
    def info(self):
        return f"Material: {self.name}, Young's Modulus: {self.youngs_modulus} Pa, Density: {self.density} kg/m^3"

class math_model:
    def __init__(self, name):
        self.name = name


class neohookean(math_model):
    def __init__(self, mu):
        super().__init__("Neo-Hookean")
        self._mu = mu  # Store in private variable
    
    @property
    def mu(self):
        return self._mu


class mooney_rivlin(math_model):
    def __init__(self, c1, c2):
        super().__init__("Mooney-Rivlin")
        self.c1 = c1  # in Pa
        self.c2 = c2  # in Pa
    
    @property
    def mu(self):
        return 2 * (self.c1 + self.c2)


class ogden(math_model):
    def __init__(self, alpha, mu_input):
        super().__init__("Ogden")
        self.alpha = alpha  # dimensionless
        self.mu_input = mu_input  # in Pa
    
    @property
    def mu(self):
        return self.mu_input

class tangent_modulus:

    @staticmethod
    def neohookean(mu,e):
        lam = 1 + e
        e_tan = mu * (2*lam + lam**-2)
        return e_tan
    
    @staticmethod
    def mooney_rivlin(c1,c2,e):
        lam = 1 + e
        term1 = 2 * (-c2*lam**-2)*(lam**2-lam**-1)
        term2 = (c2 + c1*lam**-1)*(2*lam + lam**-2)
        e_tan = 2 * (term1 + term2)
        return e_tan
    
    @staticmethod
    def ogden(mu,alpha,e):
        lam = 1 + e
        e_tan = mu * (alpha * lam**(alpha -1) + (alpha/2) * lam **(-alpha/2-1))
        return e_tan

def compute_second_moment(outer_radius, wall_thickness):
    inner_radius = outer_radius - wall_thickness
    second_moment = math.pi/4*(outer_radius**4-inner_radius**4)
    return second_moment

def flexural_rigidity(e_tan,second_moment):
    EI = second_moment * e_tan
    return EI

def channel_area(channel_radius):
    channel_area = (math.pi*channel_radius**2)/4 # four semi-circular channels
    return channel_area

def centroid_distance(channel_radius,septum_thickness):
    term1 = channel_radius + septum_thickness/2
    term2 = 4*channel_radius/3*math.pi
    centroid_distance = term1 + term2
    return centroid_distance

def directional_moment(channel_1_pressure, channel_2_pressure, channel_area, centroid_distance):
    delta_p = channel_1_pressure - channel_2_pressure
    M = delta_p * channel_area * centroid_distance
    return M

def resultant_moment(M_ac, M_bd):
    M_res = math.sqrt(M_ac**2 + M_bd**2)
    return M_res

def bending_plane_ang(M_ac, M_bd):
    if M_ac == 0 and M_bd == 0:
        phi = 0.0
    else:
        phi = math.atan2(M_bd, M_ac)
    return phi # in degrees

def curvature(M_res, EI):
    curvature = M_res / EI
    return curvature

def prestrained_length(original_length, epsilon_pre):
    length_pre = original_length * (1 + epsilon_pre)
    return length_pre

def arc_angle(curvature,length_pre):
    theta = curvature * length_pre
    return theta # in degrees
