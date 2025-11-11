import numpy as np
from metrics import Metrics

class SVMClassifier():
    
    def __init__(self, learning_rate=0.001, number_of_iterations=1000, lambda_param=0.01, kernel='linear', degree=3, gamma=1.0):
        # Initializes the SVM classifier and its parameters
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.lambda_param = lambda_param
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma

        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.K = None  # For kernel matrix
        self.loss_history = []
        self.accuracy_history = []

    def kernel_function(self, X1, X2=None):
        # Computes the kernel matrix between X1 and X2
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            # Vectorized RBF kernel
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            squared_dists = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * squared_dists)
        else:
            return np.dot(X1, X2.T)

    def calculate_loss(self):
        # Calculates the loss value for current weights
        y_labels = np.where(self.y <= 0, -1, 1)
        
        if self.kernel != 'linear':
            margins = y_labels * (np.dot(self.K, self.w) - self.b)
            reg = self.lambda_param * (self.w @ (self.K @ self.w))
        else:
            margins = y_labels * (np.dot(self.X, self.w) - self.b)
            reg = self.lambda_param * (self.w @ self.w)
        
        hinge_loss = np.maximum(0, 1 - margins)
        loss = np.mean(hinge_loss) + reg
        
        return loss

    def fit(self, X, y):
        # Trains the SVM model on the given data
        self.m, self.n = X.shape
        
        if self.kernel != 'linear':
            self.K = self.kernel_function(X, X)
            self.w = np.zeros(self.m)
        else:
            self.w = np.zeros(self.n)
        
        self.b = 0
        self.X = X
        self.y = y

        for i in range(self.number_of_iterations):
            self.update_weights()
            if i % 10 == 0:
                loss = self.calculate_loss()
                self.loss_history.append(loss)
                
                y_pred = self.predict(X)
                accuracy = Metrics.accuracy(y, y_pred)
                self.accuracy_history.append(accuracy)
        
    def update_weights(self):
        # Updates the model weights using gradient descent
        y_labels = np.where(self.y <= 0, -1, 1)

        if self.kernel != 'linear':
            # Use pre-calculated kernel matrix
            margins = y_labels * (np.dot(self.K, self.w) - self.b)
            
            # Vectorized gradient computation
            misclassified = margins < 1
            dw = 2 * self.lambda_param * (self.K @ self.w)
            dw -= self.K @ (misclassified * y_labels) / self.m
            
            db = np.sum(misclassified * y_labels) / self.m
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        else:
            margins = y_labels * (np.dot(self.X, self.w) - self.b)
            misclassified = margins < 1

            dw = (2 * self.lambda_param * self.w) - (np.dot(self.X.T, misclassified * y_labels) / self.m)
            db = np.sum(misclassified * y_labels) / self.m
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        # Predicts class labels for the input data
        if self.kernel != 'linear':
            # Calculate kernel for test (between X_train and X_test)
            K_test = self.kernel_function(X, self.X)
            output = np.dot(K_test, self.w) - self.b
        else:
            output = np.dot(X, self.w) - self.b
        
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= 0, 0, 1)
        
        return y_hat