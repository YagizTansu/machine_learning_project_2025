import data_preprocessing as dp
from SVMClassifierKernel import SVMClassifier as SVMKernelClassifier
from LogisticRegressionKernel import LogisticRegression as LogisticRegressionKernel
from hyperparameter_tuning import HyperparameterTuning
from metrics import Metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load and combine datasets
    combined_wine = dp.load_and_combine_datasets()
    
    dp.explore_dataset(combined_wine)
    
    # Split and prepare data
    X_train, X_test, y_train, y_test = dp.split_and_prepare_data(combined_wine)
    X_train_scaled, X_test_scaled = dp.StandardScaler(X_train, X_test)
    
    # Convert to numpy arrays for cross-validation compatibility
    X_train_scaled = np.array(X_train_scaled, dtype=np.float64)
    X_test_scaled = np.array(X_test_scaled, dtype=np.float64)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
     
    svm_param_grid = {
        'learning_rate': [0.01,0.1],
        'number_of_iterations': [500, 1000],
        'lambda_param': [0.01, 0.05],
        'kernel': [
            {'kernel': 'linear'},
            {'kernel': 'polynomial', 'degree': 2},
            {'kernel': 'polynomial', 'degree': 3},
            {'kernel': 'rbf', 'gamma': 0.01},
            {'kernel': 'rbf', 'gamma': 0.1},
        ]
    }
    
    # Hyperparameter tuning
    svm_tuner = HyperparameterTuning(SVMKernelClassifier(), svm_param_grid, k_folds=5)
    best_params, best_score = svm_tuner.grid_search(X_train_scaled, y_train)

    # Train final model with best parameters
    best_svm_model = SVMKernelClassifier(**best_params)
    best_svm_model.fit(X_train_scaled, y_train)
    predictions = best_svm_model.predict(X_test_scaled)
    
    # Evaluate final model
    best_svm_model_train_accuracy = Metrics.accuracy(y_train, best_svm_model.predict(X_train_scaled))
    best_svm_model_test_accuracy = Metrics.accuracy(y_test, predictions)
    best_svm_model_test_precision = Metrics.precision(y_test, predictions)
    best_svm_model_test_recall = Metrics.recall(y_test, predictions)
    best_svm_model_test_f1 = Metrics.f1_score(y_test, predictions)
    
    logistic_param_grid = {
        'learning_rate': [ 0.01, 0.1],  # 0.001 to 0.1, step 0.01
        'number_of_iterations': [500, 1000],  # 500 to 2000, step 500
        'kernel': [
            {'kernel': 'linear'},
            {'kernel': 'polynomial', 'degree': 2},
            {'kernel': 'polynomial', 'degree': 3},
            {'kernel': 'rbf', 'gamma': 0.01},
            {'kernel': 'rbf', 'gamma': 0.1},
        ]
    }

    logistic_tuner = HyperparameterTuning(LogisticRegressionKernel(), logistic_param_grid, k_folds=5)
    
    best_logistic_params, best_logistic_score = logistic_tuner.grid_search(X_train_scaled, y_train)
    best_logistic_model = LogisticRegressionKernel(**best_logistic_params)
    best_logistic_model.fit(X_train_scaled, y_train)
    logistic_predictions = best_logistic_model.predict(X_test_scaled)

    best_logistic_train_accuracy = Metrics.accuracy(y_train, best_logistic_model.predict(X_train_scaled))
    best_logistic_test_accuracy = Metrics.accuracy(y_test, logistic_predictions)
    best_logistic_test_precision = Metrics.precision(y_test, logistic_predictions)
    best_logistic_test_recall = Metrics.recall(y_test, logistic_predictions)
    best_logistic_test_f1 = Metrics.f1_score(y_test, logistic_predictions)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE RESULTS")
    print("="*60)
    
    print("\nLogistic Regression:")
    print("-" * 25)
    print(f"  Train Accuracy:  {best_logistic_train_accuracy:.4f}")
    print(f"  Test Accuracy:  {best_logistic_test_accuracy:.4f}")
    print(f"  Test Precision: {best_logistic_test_precision:.4f}")
    print(f"  Test Recall:    {best_logistic_test_recall:.4f}")
    print(f"  Test F1 Score:  {best_logistic_test_f1:.4f}")

    print("\nSVM Classifier:")
    print("-" * 25)
    print(f"  Train Accuracy:  {best_svm_model_train_accuracy:.4f}")
    print(f"  Test Accuracy:  {best_svm_model_test_accuracy:.4f}")
    print(f"  Test Precision: {best_svm_model_test_precision:.4f}")
    print(f"  Test Recall:    {best_svm_model_test_recall:.4f}")
    print(f"  Test F1 Score:  {best_svm_model_test_f1:.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    better_accuracy = "SVM" if best_svm_model_test_accuracy > best_logistic_test_accuracy else "Logistic Regression"
    better_precision = "SVM" if best_svm_model_test_precision > best_logistic_test_precision else "Logistic Regression"
    better_recall = "SVM" if best_svm_model_test_recall > best_logistic_test_recall else "Logistic Regression"
    better_f1 = "SVM" if best_svm_model_test_f1 > best_logistic_test_f1 else "Logistic Regression"

    print(f"Best Accuracy:  {better_accuracy}")
    print(f"Best Precision: {better_precision}")
    print(f"Best Recall:    {better_recall}")
    print(f"Best F1 Score:  {better_f1}")
    print("="*60)
    

    # Confusion matrix
    svm_cm = Metrics.confusion_matrix(y_test, predictions)
    logistic_cm = Metrics.confusion_matrix(y_test, logistic_predictions)

    # COMBINED PLOTTING - All visualizations in one figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Row 1: Confusion Matrices
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_title('SVM Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(logistic_cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2)
    ax2.set_title('Logistic Regression Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    # Row 2: Loss Curves
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(best_svm_model.loss_history, label='SVM Loss', color='blue', linewidth=2)
    ax3.set_title('SVM Loss over Iterations', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(best_logistic_model.loss_history, label='Logistic Regression Loss', color='orange', linewidth=2)
    ax4.set_title('Logistic Regression Loss over Iterations', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 3: Accuracy Curves
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(best_svm_model.accuracy_history, label='SVM Accuracy', color='blue', linewidth=2)
    ax5.set_title('SVM Accuracy over Iterations', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel('Accuracy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(best_logistic_model.accuracy_history, label='Logistic Regression Accuracy', color='orange', linewidth=2)
    ax6.set_title('Logistic Regression Accuracy over Iterations', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Iterations')
    ax6.set_ylabel('Accuracy')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Model Performance Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()