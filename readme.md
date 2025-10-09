# Machine Learning Project 2025: SVM vs Logistic Regression

## Model Definitions

### 1. Logistic Regression
Logistic Regression is a binary classification algorithm that models the probability of a sample belonging to a particular class using the sigmoid function.

**Mathematical Formulation:**
- Linear Model: Z = X·W + b
- Sigmoid Function: σ(Z) = 1 / (1 + e^(-Z))
- Decision Boundary: y_pred = 1 if σ(Z) ≥ 0.5, else 0

**Loss Function:** Binary Cross-Entropy
```
L = -1/m Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Optimization:** Gradient Descent
- ∂L/∂W = 1/m · X^T · (ŷ - y)
- ∂L/∂b = 1/m · Σ(ŷ - y)

**Hyperparameters:**
- learning_rate: Step size for gradient descent
- number_of_iterations: Training iterations
- kernel: Feature transformation ('linear', 'polynomial', 'rbf')
- degree: Polynomial degree (for polynomial kernel)
- gamma: RBF kernel parameter

### 2. Support Vector Machine (SVM)
SVM finds the optimal hyperplane that maximizes the margin between classes.

**Mathematical Formulation:**
- Decision Function: f(x) = W·X - b
- Prediction: y = sign(f(x))

**Loss Function:** Hinge Loss with L2 Regularization
```
L = 1/m Σ max(0, 1 - y_i·f(x_i)) + λ·||W||²
```

**Optimization:** Gradient Descent on Primal Form
- If y_i·f(x_i) ≥ 1: ∂L/∂W = 2λW
- Else: ∂L/∂W = 2λW - y_i·x_i

**Hyperparameters:**
- learning_rate: Step size for gradient descent
- number_of_iterations: Training iterations
- lambda_param: Regularization strength
- kernel: Feature transformation ('linear', 'polynomial', 'rbf')
- degree: Polynomial degree (for polynomial kernel)
- gamma: RBF kernel parameter

## Implementation Details

### Kernel Functions
Both models support three kernel types:

1. **Linear Kernel:** K(x₁, x₂) = x₁·x₂
2. **Polynomial Kernel:** K(x₁, x₂) = (x₁·x₂ + 1)^d
3. **RBF Kernel:** K(x₁, x₂) = exp(-γ·||x₁ - x₂||²)

### Hyperparameter Tuning
- Method: Grid Search with 5-Fold Cross-Validation
- Evaluation Metric: Accuracy
- Models evaluated on: Accuracy, Recall, F1-Score

### Dataset
Wine Quality Dataset (Red + White wines combined)
- Binary classification: Good quality (≥7) vs Poor quality (<7)
- Features: 11 physicochemical properties
- Preprocessing: Standard Scaling

## Performance Metrics
- **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 · (Precision · Recall) / (Precision + Recall)