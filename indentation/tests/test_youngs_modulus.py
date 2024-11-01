import unittest
import numpy as np
from scipy.optimize import fmin
from indentation.processing.calculate_parameters import parameter_youngs_modulus

class TestYoungsModulusCalculator(unittest.TestCase):
    def setUp(self):
        # Create sample data that would produce a known Young's modulus
        # Using a simple case where force increases with displacement^(3/2)
        z_values = np.linspace(0, 1, 100)
        alpha_true = 1000  # Known coefficient
        beta_true = 0
        force_values = alpha_true * (z_values**(3/2)) + beta_true
        
        self.test_data = {
            'z': -z_values,  # Negative because the function expects negative z values
            'force': force_values
        }
        
        self.radius = 10  # 10 micrometers
        self.poisson_ratio = 0.5
        self.cutoff = 50  # 50% of radius
        
    def test_basic_calculation(self):
        """Test if the function returns expected type and format"""
        E_mod, keyname = parameter_youngs_modulus(
            data=self.test_data,
            radius=self.radius,
            nu=self.poisson_ratio,
            cutoff=self.cutoff
        )
        
        # Check if return values are of correct type
        self.assertIsInstance(E_mod, float)
        self.assertIsInstance(keyname, str)
        
        # Check if Young's modulus is positive
        self.assertGreater(E_mod, 0)
        
        # Check if default keyname is correct
        self.assertEqual(keyname, "youngs_modulus")

    def test_custom_keyname(self):
        """Test if custom keyname works"""
        custom_key = "custom_modulus"
        _, keyname = parameter_youngs_modulus(
            data=self.test_data,
            radius=self.radius,
            nu=self.poisson_ratio,
            cutoff=self.cutoff,
            keyname=custom_key
        )
        self.assertEqual(keyname, custom_key)

    def test_input_validation(self):
        """Test if function handles invalid inputs appropriately"""
        # Test with empty data
        empty_data = {'z': np.array([]), 'force': np.array([])}
        with self.assertRaises(ValueError):
            parameter_youngs_modulus(
                data=empty_data,
                radius=self.radius,
                nu=self.poisson_ratio,
                cutoff=self.cutoff
            )
        
        # Test with negative radius
        with self.assertRaises(ValueError):
            parameter_youngs_modulus(
                data=self.test_data,
                radius=-1,
                nu=self.poisson_ratio,
                cutoff=self.cutoff
            )
            
        # Test with missing keys
        bad_data = {'x': np.array([1, 2, 3])}
        with self.assertRaises(ValueError):
            parameter_youngs_modulus(
                data=bad_data,
                radius=self.radius,
                nu=self.poisson_ratio,
                cutoff=self.cutoff
            )
            
        # Test with invalid data type
        invalid_data = {'z': [1, 2, 3], 'force': [1, 2, 3]}  # Lists instead of numpy arrays
        with self.assertRaises(ValueError):
            parameter_youngs_modulus(
                data=invalid_data,
                radius=self.radius,
                nu=self.poisson_ratio,
                cutoff=self.cutoff
            )

    def test_different_initial_conditions(self):
        """Test if function works with different initial conditions"""
        x0_test = [0.01, 0.1]
        E_mod1, _ = parameter_youngs_modulus(
            data=self.test_data,
            radius=self.radius,
            nu=self.poisson_ratio,
            cutoff=self.cutoff,
            x0=x0_test
        )
        
        # The result should be relatively close regardless of initial conditions
        x0_different = [0.1, 0.5]
        E_mod2, _ = parameter_youngs_modulus(
            data=self.test_data,
            radius=self.radius,
            nu=self.poisson_ratio,
            cutoff=self.cutoff,
            x0=x0_different
        )
        
        # Check if results are within 5% of each other
        percent_difference = abs(E_mod1 - E_mod2) / E_mod1 * 100
        self.assertLess(percent_difference, 5)

if __name__ == '__main__':
    unittest.main()
