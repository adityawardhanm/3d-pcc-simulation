# tests/test_etan.py
import pytest
import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Direct import from etan module
import src.python.etan as etan

tangent_modulus = etan.tangent_modulus


class TestTangentModulusNeohookean:
    """Test Neo-Hookean tangent modulus calculations"""
    
    def test_zero_strain(self):
        """At zero strain (lam=1), e_tan = mu * (2 + 1) = 3*mu"""
        mu = 1000
        e = 0.0
        e_tan = tangent_modulus.neohookean(mu, e)
        expected = mu * (2 * 1 + 1**-2)  # 3*mu
        assert e_tan == pytest.approx(expected)
        assert e_tan == pytest.approx(3000)
    
    def test_positive_strain(self):
        """Test with positive strain (extension)"""
        mu = 1000
        e = 0.1  # 10% strain
        lam = 1.1
        e_tan = tangent_modulus.neohookean(mu, e)
        expected = mu * (2 * lam + lam**-2)
        assert e_tan == pytest.approx(expected)
    
    def test_negative_strain(self):
        """Test with negative strain (compression)"""
        mu = 1000
        e = -0.1  # -10% strain (compression)
        lam = 0.9
        e_tan = tangent_modulus.neohookean(mu, e)
        expected = mu * (2 * lam + lam**-2)
        assert e_tan == pytest.approx(expected)
    
    def test_large_extension(self):
        """Test with large extension strain"""
        mu = 1000
        e = 1.0  # 100% strain (double length)
        lam = 2.0
        e_tan = tangent_modulus.neohookean(mu, e)
        expected = mu * (2 * 2.0 + 0.5**2)
        assert e_tan == pytest.approx(expected)
    
    def test_various_mu_values(self):
        """Test with different shear modulus values"""
        test_cases = [
            (100, 0.0, 300),
            (500, 0.0, 1500),
            (1000, 0.0, 3000),
            (5000, 0.0, 15000),
        ]
        for mu, e, expected in test_cases:
            e_tan = tangent_modulus.neohookean(mu, e)
            assert e_tan == pytest.approx(expected)
    
    def test_monotonicity_in_extension(self):
        """Tangent modulus should increase with extension for Neo-Hookean"""
        mu = 1000
        strains = [0.0, 0.1, 0.2, 0.5, 1.0]
        e_tan_values = [tangent_modulus.neohookean(mu, e) for e in strains]
        
        # Check that values are increasing
        for i in range(len(e_tan_values) - 1):
            assert e_tan_values[i+1] > e_tan_values[i]


class TestTangentModulusMooneyRivlin:
    """Test Mooney-Rivlin tangent modulus calculations"""
    
    def test_zero_strain(self):
        """At zero strain, check the tangent modulus"""
        mu = 1000
        c1 = 300
        c2 = 200
        e = 0.0
        lam = 1.0
        
        e_tan = tangent_modulus.mooney_rivlin(mu, c1, c2, e)
        
        # Calculate expected value
        term1 = 2 * (-c2 * lam**-2) * (lam**2 - lam**-1)
        term2 = (c2 + c1 * lam**-1) * (2 * lam + lam**-2)
        expected = 2 * (term1 + term2)
        
        assert e_tan == pytest.approx(expected)
    
    def test_positive_strain(self):
        """Test with positive strain"""
        mu = 1000
        c1 = 300
        c2 = 200
        e = 0.2
        lam = 1.2
        
        e_tan = tangent_modulus.mooney_rivlin(mu, c1, c2, e)
        
        term1 = 2 * (-c2 * lam**-2) * (lam**2 - lam**-1)
        term2 = (c2 + c1 * lam**-1) * (2 * lam + lam**-2)
        expected = 2 * (term1 + term2)
        
        assert e_tan == pytest.approx(expected)
    
    def test_negative_strain(self):
        """Test with negative strain (compression)"""
        mu = 1000
        c1 = 300
        c2 = 200
        e = -0.2
        lam = 0.8
        
        e_tan = tangent_modulus.mooney_rivlin(mu, c1, c2, e)
        
        term1 = 2 * (-c2 * lam**-2) * (lam**2 - lam**-1)
        term2 = (c2 + c1 * lam**-1) * (2 * lam + lam**-2)
        expected = 2 * (term1 + term2)
        
        assert e_tan == pytest.approx(expected)
    
    def test_neohookean_limit(self):
        """When c2=0, Mooney-Rivlin should behave differently than Neo-Hookean"""
        mu = 1000
        c1 = 500  # mu = 2(c1 + c2) = 1000 when c2=0
        c2 = 0
        e = 0.1
        
        e_tan_mr = tangent_modulus.mooney_rivlin(mu, c1, c2, e)
        # Note: This won't equal Neo-Hookean e_tan because the formulas are different
        assert e_tan_mr > 0  # Just check it's positive
    
    def test_various_parameter_combinations(self):
        """Test with different parameter combinations"""
        test_cases = [
            (1000, 250, 250, 0.0),
            (1000, 300, 200, 0.1),
            (2000, 500, 500, 0.2),
            (1500, 400, 350, -0.1),
        ]
        for mu, c1, c2, e in test_cases:
            e_tan = tangent_modulus.mooney_rivlin(mu, c1, c2, e)
            assert isinstance(e_tan, (int, float))
            # Tangent modulus should be positive for reasonable strains
            if abs(e) < 0.5:
                assert e_tan > 0


class TestTangentModulusOgden:
    """Test Ogden tangent modulus calculations"""
    
    def test_zero_strain(self):
        """At zero strain (lam=1)"""
        mu = 1000
        alpha = 2.0
        e = 0.0
        lam = 1.0
        
        e_tan = tangent_modulus.ogden(mu, alpha, e)
        expected = mu * (alpha * lam**(alpha - 1) + (alpha/2) * lam**(-alpha/2 - 1))
        
        assert e_tan == pytest.approx(expected)
    
    def test_positive_strain(self):
        """Test with positive strain"""
        mu = 1000
        alpha = 2.0
        e = 0.3
        lam = 1.3
        
        e_tan = tangent_modulus.ogden(mu, alpha, e)
        expected = mu * (alpha * lam**(alpha - 1) + (alpha/2) * lam**(-alpha/2 - 1))
        
        assert e_tan == pytest.approx(expected)
    
    def test_negative_strain(self):
        """Test with negative strain"""
        mu = 1000
        alpha = 2.0
        e = -0.2
        lam = 0.8
        
        e_tan = tangent_modulus.ogden(mu, alpha, e)
        expected = mu * (alpha * lam**(alpha - 1) + (alpha/2) * lam**(-alpha/2 - 1))
        
        assert e_tan == pytest.approx(expected)
    
    def test_various_alpha_values(self):
        """Test with different alpha parameters"""
        mu = 1000
        e = 0.1
        lam = 1.1
        
        test_alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
        for alpha in test_alphas:
            e_tan = tangent_modulus.ogden(mu, alpha, e)
            expected = mu * (alpha * lam**(alpha - 1) + (alpha/2) * lam**(-alpha/2 - 1))
            assert e_tan == pytest.approx(expected)
    
    def test_alpha_equals_two(self):
        """Special case: alpha = 2"""
        mu = 1000
        alpha = 2.0
        e = 0.5
        lam = 1.5
        
        e_tan = tangent_modulus.ogden(mu, alpha, e)
        # alpha * lam^(alpha-1) = 2 * lam^1 = 2 * lam
        # (alpha/2) * lam^(-alpha/2-1) = 1 * lam^(-2)
        expected = mu * (2 * lam + lam**-2)
        
        assert e_tan == pytest.approx(expected)
    
    def test_large_extension(self):
        """Test with large extension"""
        mu = 1000
        alpha = 2.0
        e = 2.0  # 200% strain
        lam = 3.0
        
        e_tan = tangent_modulus.ogden(mu, alpha, e)
        assert isinstance(e_tan, (int, float))
        assert e_tan > 0


class TestTangentModulusComparison:
    """Compare tangent moduli across different models"""
    
    def test_all_models_at_zero_strain(self):
        """Compare all models at zero strain"""
        mu = 1000
        c1 = 300
        c2 = 200
        alpha = 2.0
        e = 0.0
        
        e_tan_neo = tangent_modulus.neohookean(mu, e)
        e_tan_mr = tangent_modulus.mooney_rivlin(mu, c1, c2, e)
        e_tan_ogden = tangent_modulus.ogden(mu, alpha, e)
        
        # All should be positive
        assert e_tan_neo > 0
        assert e_tan_mr > 0
        assert e_tan_ogden > 0
    
    def test_consistency_check(self):
        """Verify tangent moduli are in reasonable ranges"""
        strains = [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
        mu = 1000
        
        for e in strains:
            e_tan_neo = tangent_modulus.neohookean(mu, e)
            
            # Should be positive for reasonable strains
            if e > -0.5:
                assert e_tan_neo > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_very_small_strain(self):
        """Test with very small strain values"""
        mu = 1000
        e = 1e-10
        
        e_tan_neo = tangent_modulus.neohookean(mu, e)
        # Should be very close to 3*mu
        assert e_tan_neo == pytest.approx(3000, rel=1e-6)
    
    def test_zero_mu(self):
        """Test with zero shear modulus"""
        mu = 0
        e = 0.1
        
        e_tan_neo = tangent_modulus.neohookean(mu, e)
        assert e_tan_neo == 0
    
    def test_large_mu(self):
        """Test with very large shear modulus"""
        mu = 1e10
        e = 0.1
        lam = 1.1
        
        e_tan_neo = tangent_modulus.neohookean(mu, e)
        expected = mu * (2 * lam + lam**-2)
        assert e_tan_neo == pytest.approx(expected)
    
    def test_mooney_rivlin_zero_coefficients(self):
        """Test Mooney-Rivlin with zero coefficients"""
        mu = 1000
        c1 = 0
        c2 = 0
        e = 0.1
        
        e_tan = tangent_modulus.mooney_rivlin(mu, c1, c2, e)
        assert isinstance(e_tan, (int, float))


class TestNumericalStability:
    """Test numerical stability and extreme values"""
    
    def test_near_incompressibility_limit(self):
        """Test near the incompressibility limit (lam → 0 not physically meaningful)"""
        mu = 1000
        e = -0.9  # 90% compression (lam = 0.1)
        
        # Should still compute without errors
        e_tan = tangent_modulus.neohookean(mu, e)
        assert isinstance(e_tan, (int, float))
        assert not math.isnan(e_tan)
        assert not math.isinf(e_tan)
    
    def test_float_precision(self):
        """Test with various float precisions"""
        mu = 1000.123456789
        e = 0.123456789
        
        e_tan_neo = tangent_modulus.neohookean(mu, e)
        assert isinstance(e_tan_neo, (int, float))
        assert e_tan_neo > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])