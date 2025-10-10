import numpy as np
from metrics import Metrics

class LogisticRegression:
    # declaring the hyperparameters: learning_rate, number_of_iterations, kernel, degree, gamma
    def __init__(self, learning_rate=0.01, number_of_iterations=1000, kernel='linear', degree=2, gamma=1.0):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.weights = None
        self.bias = None
        self.X_train = None  # Store training data for kernel computation
        self.loss_history = []  # Track loss over iterations
        self.accuracy_history = []  # Track accuracy over iterations
    
    def kernel_function(self, X1, X2=None):
        """Apply kernel transformation"""
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            # RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
            # Compute squared Euclidean distance matrix
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            squared_dists = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * squared_dists)
        else:
            raise ValueError("Unsupported kernel type. Use 'linear', 'polynomial', or 'rbf'")
    
    def calculate_loss(self, y_hat, y):
        """Calculate binary cross-entropy loss"""
        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss
    
    # Fit the model to the training data
    def fit(self, X, y):
        self.X_train = X  # Store training data
        
        # Apply kernel transformation
        if self.kernel == 'linear':
            X_transformed = X
        else:  # polynomial or rbf kernel
            X_transformed = self.kernel_function(X, X)
        
        self.m, self.n = X_transformed.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        
        self.X = X_transformed
        self.y = y
        
        # Gradient descent
        for i in range(self.number_of_iterations):
            self.update_weights()
            
            # Calculate and store loss every iteration (or every N iterations for efficiency)
            if i % 10 == 0:  # Calculate every 10 iterations to save computation
                Z = np.dot(self.X, self.weights) + self.bias
                y_hat = 1 / (1 + np.exp(-Z))
                loss = self.calculate_loss(y_hat, self.y)
                self.loss_history.append(loss)
                
                # Calculate accuracy
                y_pred = np.where(y_hat <= 0.5, 0, 1)
                accuracy = Metrics.accuracy(self.y, y_pred)
                self.accuracy_history.append(accuracy)
    
    def update_weights(self):
        
        # linear model which is Z = X.W + b and then applying sigmoid function(y^ = 1 / (1 + e^(-Z)))
        Z = np.dot(self.X, self.weights) + self.bias
        y_hat = 1 / (1 + np.exp(-Z))

        # partial derivatives
        dw = (1 / self.m) * np.dot(self.X.T, (y_hat - self.y))
        db = (1 / self.m) * np.sum(y_hat - self.y)
        
        # update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        # Apply kernel transformation
        if self.kernel == 'linear':
            X_transformed = X
        else:  # polynomial or rbf kernel
            X_transformed = self.kernel_function(X, self.X_train)
            
        Z = np.dot(X_transformed, self.weights) + self.bias
        y_hat = 1 / (1 + np.exp(-Z))  # Sigmoid function
        y_hat = np.where(y_hat <= 0.5, 0, 1)  # Convert probabilities to class labels
        return y_hat

