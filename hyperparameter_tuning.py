import numpy as np
import cross_validation as cv

class HyperparameterTuning:
    def __init__(self, model, param_grid, k_folds=5):
        self.model = model
        self.param_grid = param_grid
        self.k_folds = k_folds

    def grid_search(self, X, y):
        best_params = None
        best_score = 0

        for params in self.param_grid:
            print(f"Evaluating parameters: {params}")
            self.model.__init__(**params)  # Re-initialize model with new parameters
            cross_val = cv.CrossValidation(self.model, k_folds=self.k_folds)
            score = cross_val.evaluate(X, y)

            if score > best_score:
                best_score = score
                best_params = params

        print(f"Best parameters: {best_params} with score: {best_score:.4f}")
        return best_params, best_score