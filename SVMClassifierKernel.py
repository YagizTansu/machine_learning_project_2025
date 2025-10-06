import numpy as np

class SVMClassifier():
    
    def __init__(self, learning_rate=0.001, number_of_iterations=1000, lambda_param=0.01, kernel='linear', degree=3, gamma=1.0):
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
        self.loss_history = []

        print("SVM Classifier initialized with learning_rate={}, number_of_iterations={}, lambda_param={}, kernel={}".format(
            learning_rate, number_of_iterations, lambda_param, kernel))

    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'polynomial':
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            squared_distance = np.sum((x1 - x2) ** 2)
            return np.exp(-self.gamma * squared_distance)
        else:
            return np.dot(x1, x2)

    def calculate_loss(self):
        y_labels = np.where(self.y <= 0, -1, 1)
        
        if self.kernel == 'rbf':
            margins = []
            for index, x_i in enumerate(self.X):
                kernel_values = np.array([self.kernel_function(x_i, x_j) for x_j in self.X])
                margin = y_labels[index] * (np.dot(kernel_values, self.w) - self.b)
                margins.append(margin)
            margins = np.array(margins)
        else:
            # Linear ve polynomial için düzeltme: w.T @ X.T hesaplanmalı
            margins = y_labels * (np.dot(self.X, self.w) - self.b)
        
        hinge_loss = np.maximum(0, 1 - margins)
        loss = np.mean(hinge_loss) + self.lambda_param * np.dot(self.w, self.w)
        
        return loss

    def fit(self, X, y):
        self.m, self.n = X.shape
        
        if self.kernel == 'rbf':
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
        
    def update_weights(self):
        y_labels = np.where(self.y <= 0, -1, 1)
        
        if self.kernel == 'rbf':
            # Batch gradient descent için tüm gradient'leri topla
            dw_total = np.zeros(self.m)
            db_total = 0
            
            for index, x_i in enumerate(self.X):
                kernel_values = np.array([self.kernel_function(x_i, x_j) for x_j in self.X])
                condition = y_labels[index] * (np.dot(kernel_values, self.w) - self.b)
                
                if condition >= 1:
                    dw_total += 2 * self.lambda_param * self.w
                else:
                    dw_total += 2 * self.lambda_param * self.w - (kernel_values * y_labels[index])
                    db_total += y_labels[index]
            
            # Ortalama gradient ile güncelle
            self.w = self.w - self.learning_rate * (dw_total / self.m)
            self.b = self.b - self.learning_rate * (db_total / self.m)
        else:
            # Linear ve polynomial için düzeltilmiş gradient descent
            dw_total = np.zeros(self.n)
            db_total = 0
            
            for index, x_i in enumerate(self.X):
                condition = y_labels[index] * (np.dot(x_i, self.w) - self.b)
                
                if condition >= 1:
                    dw_total += 2 * self.lambda_param * self.w
                else:
                    dw_total += 2 * self.lambda_param * self.w - (x_i * y_labels[index])
                    db_total += y_labels[index]
            
            self.w = self.w - self.learning_rate * (dw_total / self.m)
            self.b = self.b - self.learning_rate * (db_total / self.m)

    def predict(self, X):
        if self.kernel == 'rbf':
            output = []
            for x in X:
                kernel_values = np.array([self.kernel_function(x, x_train) for x_train in self.X])
                output.append(np.dot(kernel_values, self.w) - self.b)
            output = np.array(output)
        else:
            # Linear ve polynomial için düzeltme
            output = np.dot(X, self.w) - self.b
        
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= 0, 0, 1)
        
        return y_hat