import numpy as np

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate the accuracy of predictions."""
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        """Calculate the precision of predictions."""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    @staticmethod
    def recall(y_true, y_pred):
        """Calculate the recall of predictions."""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred):
        """Calculate the F1 score of predictions."""
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
