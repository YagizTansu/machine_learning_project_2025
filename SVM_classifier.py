import numpy as np

class SVMClassifier():
    
    # Initialize the SVM classifier with default parameters
    def __init__(self,learning_rate=0.001, number_of_iterations=1000,lambda_param=0.01):
        self.learning_rate = learning_rate # learning rate for weight updates
        self.number_of_iterations = number_of_iterations # number of iterations for training
        self.lambda_param = lambda_param # regularization parameter

        self.w = None # weights
        self.b = None # bias
        self.X = None # training data
        self.y = None # target labels

        print("SVM Classifier initialized with learning_rate={}, number_of_iterations={}, lambda_param={}".format(learning_rate, number_of_iterations, lambda_param))

    # fitting the dataset to SVM classifier
    def fit(self, X, y):
        # m -> number of samples(number of rows), n -> number of features(number of columns)
        self.m , self.n = X.shape
        self.w = np.zeros(self.n) # weights initialization
        self.b = 0 # bias initialization
        
        self.X = X
        self.y = y

        # Implement Gradient descent algorithm to update weights
        for i in range(self.number_of_iterations):
            self.update_weights()
        

    # Update the weights of the SVM classifier, updating bias and support vectors
    def update_weights(self):
        # label encoding: convert all 0 labels to -1 for SVM algorithm
        y_labels = np.where(self.y <= 0, -1, 1)
        
        # Iterate through each sample and update weights and bias , Gradients -> dw and db
        for index, x_i in enumerate(self.X):
            if (y_labels[index] * (np.dot(x_i, self.w) - self.b)) >= 1: # yi * (wx - b) >= 1
                dw = 2 * self.lambda_param * self.w
                db = 0
            elif (y_labels[index] * (np.dot(x_i, self.w) - self.b)) < 1: # yi * (wx - b) < 1
                dw = 2 * self.lambda_param * self.w - (x_i * y_labels[index])
                db = y_labels[index]
        
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    # Predict the class labels for the input data
    def predict(self, X):
        output = np.dot(X, self.w) - self.b # xw - b
        predicted_labels = np.sign(output) # return -1 or 1 based on the sign of the output
        y_hat = np.where(predicted_labels <= 0, 0, 1) # convert -1 back to 0 for original labels
        
        return y_hat