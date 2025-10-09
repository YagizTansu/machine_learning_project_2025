import numpy as np
import cross_validation as cv
from itertools import product

class HyperparameterTuning:
    def __init__(self, model, param_grid, k_folds=5):
        self.model = model
        self.param_grid = param_grid
        self.k_folds = k_folds

    def _generate_param_combinations(self):
        """Generate all parameter combinations from param_grid."""
        # Separate common params from kernel-specific params
        common_params = {}
        kernel_configs = []
        
        for key, value in self.param_grid.items():
            if key == 'kernel':
                kernel_configs = value
            elif isinstance(value, (list, tuple, np.ndarray)):
                common_params[key] = value
            else:
                common_params[key] = [value]
        
        # Generate combinations for common parameters
        common_keys = list(common_params.keys())
        common_values = [common_params[k] for k in common_keys]
        
        all_combinations = []
        
        for common_combo in product(*common_values):
            base_params = dict(zip(common_keys, common_combo))
            
            # Add kernel configurations
            for kernel_config in kernel_configs:
                params = base_params.copy()
                params.update(kernel_config)
                all_combinations.append(params)
        
        return all_combinations

    def grid_search(self, X, y):
        best_params = None
        best_score = 0

        param_combinations = self._generate_param_combinations()
        
        for params in param_combinations:
            print(f"Evaluating parameters: {params}")
            self.model.__init__(**params)  # Re-initialize model with new parameters
            cross_val = cv.CrossValidation(self.model, k_folds=self.k_folds)
            score = cross_val.evaluate(X, y)

            if score > best_score:
                best_score = score
                best_params = params

        print(f"Best parameters: {best_params} with score: {best_score:.4f}")
        return best_params, best_score