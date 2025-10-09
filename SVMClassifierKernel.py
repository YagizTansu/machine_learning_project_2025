import numpy as np
from metrics import Metrics

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
        self.K = None  # Kernel matrisi için
        self.loss_history = []
        self.accuracy_history = []

    def kernel_function(self, X1, X2=None):
        """Vectorized kernel computation"""
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
        y_labels = np.where(self.y <= 0, -1, 1)
        
        if self.kernel == 'rbf':
            # Kernel matrisi ile vektörize hesaplama
            margins = y_labels * (np.dot(self.K, self.w) - self.b)
        else:
            margins = y_labels * (np.dot(self.X, self.w) - self.b)
        
        hinge_loss = np.maximum(0, 1 - margins)
        loss = np.mean(hinge_loss) + self.lambda_param * np.dot(self.w, self.w)
        
        return loss

    def fit(self, X, y):
        self.m, self.n = X.shape
        
        # ✅ KERNEL MATRİSİNİ BİR KEZ HESAPLA
        if self.kernel == 'rbf':
            self.K = self.kernel_function(X, X)  # 7000×7000 - sadece bir kez!
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
        y_labels = np.where(self.y <= 0, -1, 1)
        
        if self.kernel == 'rbf':
            # ✅ Önceden hesaplanmış kernel matrisini kullan
            margins = y_labels * (np.dot(self.K, self.w) - self.b)
            
            # Vectorized gradient computation
            misclassified = margins < 1
            dw = 2 * self.lambda_param * self.w
            dw -= np.dot(self.K.T, misclassified * y_labels) / self.m
            
            db = -np.sum(misclassified * y_labels) / self.m
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        else:
            margins = y_labels * (np.dot(self.X, self.w) - self.b)
            
            misclassified = margins < 1
            dw = 2 * self.lambda_param * self.w
            dw -= np.dot(self.X.T, misclassified * y_labels) / self.m
            
            db = -np.sum(misclassified * y_labels) / self.m
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        if self.kernel == 'rbf':
            # Test için kernel hesapla (X_train ile X_test arası)
            K_test = self.kernel_function(X, self.X)
            output = np.dot(K_test, self.w) - self.b
        else:
            output = np.dot(X, self.w) - self.b
        
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= 0, 0, 1)
        
        return y_hat