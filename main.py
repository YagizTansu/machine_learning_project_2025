import data_preprocessing as dp
from SVM_classifier import SVMClassifier as SVMBasicClassifier
from SVMClassifierKernel import SVMClassifier as SVMKernelClassifier
from LogisticRegression import LogisticRegression
from LogisticRegressionKernel import LogisticRegression as LogisticRegressionKernel
from metrics import Metrics
from cross_validation import CrossValidation
from hyperparameter_tuning import HyperparameterTuning
import numpy as np

def main():
    """Main function to run the SVM classifier on the wine quality dataset."""
    # Load and combine datasets
    combined_wine = dp.load_and_combine_datasets()
    
    # Split and prepare data
    X_train, X_test, y_train, y_test = dp.split_and_prepare_data(combined_wine)
    X_train_scaled, X_test_scaled = dp.StandardScaler(X_train, X_test)
    
    # Convert to numpy arrays for cross-validation compatibility
    X_train_scaled = np.array(X_train_scaled, dtype=np.float64)
    X_test_scaled = np.array(X_test_scaled, dtype=np.float64)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    svm_param_grid = [
        {'learning_rate': 0.001, 'number_of_iterations': 1000, 'lambda_param': 0.01, 'kernel': 'linear'},
        {'learning_rate': 0.001, 'number_of_iterations': 1000, 'lambda_param': 0.01, 'kernel': 'polynomial', 'degree': 2},
        {'learning_rate': 0.001, 'number_of_iterations': 1000, 'lambda_param': 0.01, 'kernel': 'polynomial', 'degree': 3},
        {'learning_rate': 0.001, 'number_of_iterations': 1000, 'lambda_param': 0.01, 'kernel': 'rbf', 'gamma': 0.1},
        {'learning_rate': 0.001, 'number_of_iterations': 1000, 'lambda_param': 0.01, 'kernel': 'rbf', 'gamma': 1.0},
        {'learning_rate': 0.001, 'number_of_iterations': 1000, 'lambda_param': 0.01, 'kernel': 'rbf', 'gamma': 10.0}
    ]
    

    # Hyperparameter tuning
    svm_tuner = HyperparameterTuning(SVMKernelClassifier(), svm_param_grid, k_folds=5)
    best_params, best_score = svm_tuner.grid_search(X_train_scaled, y_train)

    # Train final model with best parameters
    best_svm_model = SVMKernelClassifier(**best_params)
    best_svm_model.fit(X_train_scaled, y_train)
    predictions = best_svm_model.predict(X_test_scaled)

    best_svm_model_test_accuracy = Metrics.accuracy(y_test, predictions)
    best_svm_model_test_recall = Metrics.recall(y_test, predictions)
    best_svm_model_test_f1 = Metrics.f1_score(y_test, predictions)
    
    logistic_param_grid = [
        {'learning_rate': 0.01, 'number_of_iterations': 1000, 'kernel': 'linear'},
        {'learning_rate': 0.01, 'number_of_iterations': 1000, 'kernel': 'polynomial', 'degree': 2},
        {'learning_rate': 0.01, 'number_of_iterations': 1000, 'kernel': 'polynomial', 'degree': 3},
        {'learning_rate': 0.01, 'number_of_iterations': 1000, 'kernel': 'rbf', 'gamma': 0.1},
        {'learning_rate': 0.01, 'number_of_iterations': 1000, 'kernel': 'rbf', 'gamma': 1.0},
        {'learning_rate': 0.01, 'number_of_iterations': 1000, 'kernel': 'rbf', 'gamma': 10.0}
        ]

    logistic_tuner = HyperparameterTuning(LogisticRegressionKernel(), logistic_param_grid, k_folds=5)
    
    best_logistic_params, best_logistic_score = logistic_tuner.grid_search(X_train_scaled, y_train)
    best_logistic_model = LogisticRegressionKernel(**best_logistic_params)
    best_logistic_model.fit(X_train_scaled, y_train)
    logistic_predictions = best_logistic_model.predict(X_test_scaled)

    logistic_test_accuracy = Metrics.accuracy(y_test, logistic_predictions)
    logistic_test_recall = Metrics.recall(y_test, logistic_predictions)
    logistic_test_f1 = Metrics.f1_score(y_test, logistic_predictions)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE RESULTS")
    print("="*60)
    
    print("\nLogistic Regression:")
    print("-" * 25)
    print(f"  Test Accuracy: {logistic_test_accuracy:.4f}")
    print(f"  Test Recall:   {logistic_test_recall:.4f}")
    print(f"  Test F1 Score: {logistic_test_f1:.4f}")
    
    print("\nSVM Classifier:")
    print("-" * 25)
    print(f"  Test Accuracy: {best_svm_model_test_accuracy:.4f}")
    print(f"  Test Recall:   {best_svm_model_test_recall:.4f}")
    print(f"  Test F1 Score: {best_svm_model_test_f1:.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    better_accuracy = "SVM" if best_svm_model_test_accuracy > logistic_test_accuracy else "Logistic Regression"
    better_recall = "SVM" if best_svm_model_test_recall > logistic_test_recall else "Logistic Regression"
    better_f1 = "SVM" if best_svm_model_test_f1 > logistic_test_f1 else "Logistic Regression"
    
    print(f"Best Accuracy:  {better_accuracy}")
    print(f"Best Recall:    {better_recall}")
    print(f"Best F1 Score:  {better_f1}")
    print("="*60)
    
    
if __name__ == "__main__":
    main()