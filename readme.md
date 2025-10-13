# Wine Quality Classification Project

This project implements and compares Support Vector Machine (SVM) and Logistic Regression classifiers on the Wine Quality dataset. It includes custom kernel support, hyperparameter tuning, cross-validation, and performance visualization.

## Project Structure

- `data_preprocessing.py`: Functions for loading, combining, exploring, and preprocessing the wine datasets.
- `cross_validation.py`: Implements k-fold cross-validation logic.
- `hyperparameter_tuning.py`: Grid search for hyperparameter optimization.
- `LogisticRegressionKernel.py`: Custom Logistic Regression implementation with kernel support.
- `SVMClassifierKernel.py`: Custom SVM implementation with kernel support.
- `metrics.py`: Functions for calculating accuracy, precision, recall, F1 score, and confusion matrix.
- `data/`: Contains wine quality datasets (`winequality-red.csv`, `winequality-white.csv`, `winequality.names`).
- `result.txt`: Stores experiment results (if used).

## Features

- **Custom SVM and Logistic Regression**: Both models support linear, polynomial, and RBF kernels.
- **Hyperparameter Tuning**: Grid search over learning rate, iterations, regularization, and kernel parameters.
- **Cross-Validation**: 5-fold cross-validation for robust model selection.
- **Performance Metrics**: Accuracy, precision, recall, F1 score, and confusion matrix.
- **Visualization**: Combined plots for confusion matrices, loss curves, and accuracy curves for both models.

## Usage

1. **Install Requirements**
   - Python 3.8+
   - Required packages: `numpy`, `matplotlib`, `seaborn`, `pandas`
   - Install with:
     ```bash
     pip install numpy matplotlib seaborn pandas
     ```

2. **Run the Project**
   ```bash
   python main.py
   ```

3. **Results**
   - Model performance metrics and visualizations will be displayed.
   - Comparison summary will show which model performed better on each metric.

## How It Works

- Loads and combines red and white wine datasets.
- Preprocesses data (scaling, splitting).
- Performs grid search hyperparameter tuning for SVM and Logistic Regression.
- Trains final models with best parameters.
- Evaluates and compares models using multiple metrics.
- Visualizes results in a single figure.