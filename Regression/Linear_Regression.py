import numpy as np 

class linear_regression:

    # parameter initialization
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations 

    # weight & bias initialization
    def param_init(self):
        self.w = np.zeros(self.X.shape[1])
        self.b = 0

    # fitting process
    def fit(self, X, y):
        self.X = X 
        self.y = y
        self.param_init()

        for i in range(self.n_iterations):

            y_hat = np.matmul(self.X, self.w) + self.b

            dw = 1/X.shape[0] * np.matmul(self.X.T, (y_hat - self.y))
            db = 1/X.shape[0] * np.sum(y_hat - self.y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    # prediction process
    def predict(self, X):
        y_pred = np.matmul(X, self.w) + self.b

        return y_pred
    
# Adapted from Siddhardhan's Logistic Regression code.
# Check: https://www.youtube.com/watch?v=DeUAvYyB0Os&list=PLfFghEzKVmjsF8ixJ-xKVuQayPWRH4Sp6&index=6
