import numpy as np

class SVMClassifier():
    
    # Initialize the SVM classifier with default parameters
    def __init__(self, learning_rate=0.001, number_of_iterations=1000, lambda_param=0.01, kernel='linear', degree=3, gamma=1.0):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.lambda_param = lambda_param
        self.kernel = kernel  # 'linear', 'polynomial', or 'rbf'
        self.degree = degree  # polynomial kernel için derece
        self.gamma = gamma  # RBF kernel için gamma parametresi

        self.w = None
        self.b = None
        self.X = None
        self.y = None

        print("SVM Classifier initialized with learning_rate={}, number_of_iterations={}, lambda_param={}, kernel={}".format(
            learning_rate, number_of_iterations, lambda_param, kernel))

    # Kernel fonksiyonu
    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'polynomial':
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            # RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
            squared_distance = np.sum((x1 - x2) ** 2)
            return np.exp(-self.gamma * squared_distance)
        else:
            return np.dot(x1, x2)  # default olarak linear

    # fitting the dataset to SVM classifier
    def fit(self, X, y):
        self.m, self.n = X.shape
        
        if self.kernel == 'rbf':
            # RBF kernel için training data'yı saklıyoruz
            self.w = np.zeros(self.m)  # RBF için ağırlık vektörü sample sayısı kadar
        else:
            self.w = np.zeros(self.n)
        
        self.b = 0
        
        self.X = X
        self.y = y

        # Gradient descent
        for i in range(self.number_of_iterations):
            self.update_weights()
        
    # Update the weights
    def update_weights(self):
        y_labels = np.where(self.y <= 0, -1, 1)
        
        if self.kernel == 'rbf':
            # RBF kernel için farklı güncelleme
            for index, x_i in enumerate(self.X):
                # Kernel değerlerini hesapla
                kernel_values = np.array([self.kernel_function(x_i, x_j) for x_j in self.X])
                condition = y_labels[index] * (np.dot(kernel_values, self.w) - self.b)
                
                if condition >= 1:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - (kernel_values * y_labels[index])
                    db = y_labels[index]
            
                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db
        else:
            # Linear ve polynomial kernel için mevcut kod
            for index, x_i in enumerate(self.X):
                condition = y_labels[index] * (self.kernel_function(x_i, self.w) - self.b)
                
                if condition >= 1:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - (x_i * y_labels[index])
                    db = y_labels[index]
            
                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db

    # Predict
    def predict(self, X):
        if self.kernel == 'rbf':
            # RBF kernel için prediction
            output = []
            for x in X:
                kernel_values = np.array([self.kernel_function(x, x_train) for x_train in self.X])
                output.append(np.dot(kernel_values, self.w) - self.b)
            output = np.array(output)
        else:
            # Linear ve polynomial kernel için mevcut kod
            output = np.array([self.kernel_function(x, self.w) - self.b for x in X])
        
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= 0, 0, 1)
        
        return y_hat