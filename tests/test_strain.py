# tests/test_strain.py
import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.python.strain import (
    math_model, 
    neohookean, 
    mooney_rivlin, 
    ogden, 
    yeoh, 
    yeoh_simplified, 
    gent
)


class TestMathModel:
    """Test the base math_model class"""
    
    def test_base_model_initialization(self):
        model = math_model("Test Model")
        assert model.name == "Test Model"


class TestNeohookean:
    """Test Neo-Hookean material model"""
    
    def test_initialization(self):
        mat = neohookean(mu=1000)
        assert mat.name == "Neo-Hookean"
        assert mat._mu == 1000
    
    def test_mu_property(self):
        mat = neohookean(mu=1000)
        assert mat.mu == 1000
    
    def test_mu_calculation_various_values(self):
        test_cases = [100, 500, 1000, 5000, 10000]
        for mu_val in test_cases:
            mat = neohookean(mu=mu_val)
            assert mat.mu == mu_val


class TestMooneyRivlin:
    """Test Mooney-Rivlin material model"""
    
    def test_initialization(self):
        mat = mooney_rivlin(c1=300, c2=200)
        assert mat.name == "Mooney-Rivlin"
        assert mat.c1 == 300
        assert mat.c2 == 200
    
    def test_mu_calculation(self):
        mat = mooney_rivlin(c1=300, c2=200)
        # mu = 2(c1 + c2) = 2(300 + 200) = 1000
        assert mat.mu == 1000
    
    def test_mu_various_parameters(self):
        test_cases = [
            (100, 50, 300),   # 2(100+50) = 300
            (500, 0, 1000),   # 2(500+0) = 1000 (Neo-Hookean limit)
            (250, 250, 1000), # 2(250+250) = 1000
        ]
        for c1, c2, expected_mu in test_cases:
            mat = mooney_rivlin(c1=c1, c2=c2)
            assert mat.mu == expected_mu


class TestOgden:
    """Test Ogden material model"""
    
    def test_initialization(self):
        mat = ogden(alpha=2.0, mu_input=1000)
        assert mat.name == "Ogden"
        assert mat.alpha == 2.0
        assert mat.mu_input == 1000
    
    def test_mu_property(self):
        mat = ogden(alpha=2.0, mu_input=1000)
        assert mat.mu == 1000  # For single-term Ogden, mu = mu_input
    
    def test_mu_various_values(self):
        test_cases = [
            (1.5, 500, 500),
            (2.0, 1000, 1000),
            (3.0, 2000, 2000),
        ]
        for alpha, mu_input, expected in test_cases:
            mat = ogden(alpha=alpha, mu_input=mu_input)
            assert mat.mu == expected


class TestYeoh:
    """Test Yeoh material model"""
    
    def test_initialization(self):
        mat = yeoh(c1=500, c2=100, c3=50)
        assert mat.name == "Yeoh"
        assert mat.c1 == 500
        assert mat.c2 == 100
        assert mat.c3 == 50
    
    def test_mu_calculation(self):
        mat = yeoh(c1=500, c2=100, c3=50)
        # mu = 2*c1 = 2*500 = 1000
        assert mat.mu == 1000
    
    def test_mu_various_parameters(self):
        test_cases = [
            (250, 100, 50, 500),   # 2*250 = 500
            (500, 200, 100, 1000), # 2*500 = 1000
            (1000, 0, 0, 2000),    # 2*1000 = 2000
        ]
        for c1, c2, c3, expected_mu in test_cases:
            mat = yeoh(c1=c1, c2=c2, c3=c3)
            assert mat.mu == expected_mu


class TestYeohSimplified:
    """Test Yeoh Simplified material model"""
    
    def test_initialization(self):
        mat = yeoh_simplified(c1=500)
        assert mat.name == "Yeoh Simplified"
        assert mat.c1 == 500
    
    def test_mu_calculation(self):
        mat = yeoh_simplified(c1=500)
        # mu = 2*c1 = 2*500 = 1000
        assert mat.mu == 1000
    
    def test_mu_various_values(self):
        test_cases = [100, 250, 500, 1000, 2000]
        for c1 in test_cases:
            mat = yeoh_simplified(c1=c1)
            assert mat.mu == 2 * c1


class TestGent:
    """Test Gent material model"""
    
    def test_initialization(self):
        mat = gent(mu_param=1000, Jm=50)
        assert mat.name == "Gent"
        assert mat._mu_param == 1000
        assert mat.Jm == 50
    
    def test_mu_property(self):
        mat = gent(mu_param=1000, Jm=50)
        assert mat.mu == 1000
    
    def test_mu_various_values(self):
        test_cases = [
            (500, 30, 500),
            (1000, 50, 1000),
            (2000, 100, 2000),
        ]
        for mu_param, Jm, expected in test_cases:
            mat = gent(mu_param=mu_param, Jm=Jm)
            assert mat.mu == expected


class TestMaterialList:
    """Test using multiple materials in a list (real-world usage)"""
    
    def test_heterogeneous_material_list(self):
        materials = [
            neohookean(mu=1000),
            mooney_rivlin(c1=300, c2=200),
            ogden(alpha=2.0, mu_input=1500),
            yeoh(c1=400, c2=100, c3=50),
            yeoh_simplified(c1=600),
            gent(mu_param=800, Jm=40),
        ]
        
        expected_mus = [1000, 1000, 1500, 800, 1200, 800]
        actual_mus = [mat.mu for mat in materials]
        
        assert actual_mus == expected_mus
    
    def test_material_names(self):
        materials = [
            neohookean(mu=1000),
            mooney_rivlin(c1=300, c2=200),
            ogden(alpha=2.0, mu_input=1500),
            yeoh(c1=400, c2=100, c3=50),
            yeoh_simplified(c1=600),
            gent(mu_param=800, Jm=40),
        ]
        
        expected_names = [
            "Neo-Hookean",
            "Mooney-Rivlin",
            "Ogden",
            "Yeoh",
            "Yeoh Simplified",
            "Gent"
        ]
        
        actual_names = [mat.name for mat in materials]
        assert actual_names == expected_names


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_parameters(self):
        mat_neo = neohookean(mu=0)
        assert mat_neo.mu == 0
        
        mat_mr = mooney_rivlin(c1=0, c2=0)
        assert mat_mr.mu == 0
    
    def test_negative_parameters_allowed(self):
        # Some models might allow negative parameters in theory
        mat = neohookean(mu=-100)
        assert mat.mu == -100
    
    def test_float_parameters(self):
        mat = mooney_rivlin(c1=123.456, c2=78.9)
        assert mat.mu == pytest.approx(2 * (123.456 + 78.9))
    
    def test_very_large_parameters(self):
        mat = neohookean(mu=1e10)
        assert mat.mu == 1e10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])