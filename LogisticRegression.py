import numpy as np

class LogisticRegression:
    # declaring the hyperparameters: learning_rate, number_of_iterations
    def __init__(self,learning_rate=0.01, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        print("Logistic Regression initialized with learning_rate={}, number_of_iterations={}".format(learning_rate, number_of_iterations))
    
    # Fit the model to the training data
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        
        self.X = X
        self.y = y
        
        # Gradient descent
        for i in range(self.number_of_iterations):
            self.update_weights()
            
    def update_weights(self):
        
        # linear model which is Z = X.W + b and then applying sigmoid function(y^ = 1 / (1 + e^(-Z)))
        Z = np.dot(self.X, self.weights) + self.bias
        y_hat = 1 / (1 + np.exp(-Z))

        #partial derivatives
        dw = (1 / self.m) * np.dot(self.X.T, (y_hat - self.y))
        db = (1 / self.m) * np.sum(y_hat - self.y)
        
        #update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        y_hat = 1 / (1 + np.exp(-Z))  # Sigmoid function
        y_hat = np.where(y_hat <= 0.5, 0, 1)  # Convert probabilities to class labels
        return y_hat

