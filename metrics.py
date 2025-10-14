import numpy as np

class Metrics:
    @staticmethod
    # Calculates the accuracy of predictions
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    # Calculates the precision of predictions
    def precision(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    @staticmethod
    # Calculates the recall of predictions
    def recall(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    @staticmethod
    # Calculates the F1 score of predictions
    def f1_score(y_true, y_pred):
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    @staticmethod
    # Calculates the confusion matrix
    def confusion_matrix(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[tn, fp], [fn, tp]])