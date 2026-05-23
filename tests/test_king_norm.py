import unittest
import sys
import os
import csv
import inspect

# Ensure the parent directory is in the path so we can import implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import implementations

class TestKingNorm(unittest.TestCase):
    def test_mathematica_values_relative_error(self):
        """
        Test that every implementation matches the Mathematica values 
        up to a relative error of 10^-10.
        """
        # Find all implementation functions dynamically
        impl_funcs = []
        for name, obj in inspect.getmembers(implementations, inspect.isfunction):
            if name.startswith('compute_'):
                impl_funcs.append(obj)

        csv_path = os.path.join(os.path.dirname(__file__), 'mathematica_vals.csv')
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                alpha = float(row['alpha'])
                beta = float(row['beta'])
                expected_N = float(row['N'])
                
                for func in impl_funcs:
                    with self.subTest(func=func.__name__, alpha=alpha, beta=beta):
                        actual_N = float(func(alpha, beta))
                        rel_error = abs(actual_N - expected_N) / abs(expected_N)
                        self.assertLessEqual(
                            rel_error, 1e-10,
                            f"{func.__name__} failed for alpha={alpha}, beta={beta}. "
                            f"Expected: {expected_N}, Actual: {actual_N}, Rel Error: {rel_error}"
                        )

if __name__ == '__main__':
    unittest.main()
