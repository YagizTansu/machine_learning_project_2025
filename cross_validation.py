import numpy as np

class CrossValidation:
    def __init__(self, model, k_folds=5):
        # Initializes the cross-validation object with model and number of folds
        self.model = model
        self.k_folds = k_folds

    def split_data(self, X, y):
        # Splits the data into k folds for cross-validation
        fold_size = len(X) // self.k_folds
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        folds = []
        for k in range(self.k_folds):
            start = k * fold_size
            end = start + fold_size if k != self.k_folds - 1 else len(X)
            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))
            folds.append((train_indices, test_indices))
        
        return folds

    def evaluate(self, X, y):
        # Evaluates the model using k-fold cross-validation
        folds = self.split_data(X, y)
        accuracies = []

        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            accuracy = np.mean(predictions == y_test)
            accuracies.append(accuracy)

        average_accuracy = np.mean(accuracies)
        print(f"Average Accuracy over {self.k_folds} folds: {average_accuracy:.4f}")
        return average_accuracy