import numpy as np

class Logistic_Regression():

    # parameter initialization
    def __init__(self, learning_rate=0.001, iters_number=1000):
        self.learning_rate = learning_rate
        self.iters_number  = iters_number

    # weight & bias initialization
    def param_init(self):
        self.w = np.zeros(self.X.shape[1])
        self.b = 0 

    # fitting process
    def fit(self, X, y):
        self.X= X
        self.y= y

        self.param_init()
        for i in range(self.iters_number): 
            self.update_param()
    
    # update w and b
    def update_param(self):
        Y_hat= self.sigmoid()
        dw   = (1/self.X.shape[0]) * np.dot(self.X.T, (Y_hat - self.y))
        db   = (1/self.X.shape[0]) * np.sum(Y_hat - self.y)

        self.w -= self.learning_rate * dw 
        self.b -= self.learning_rate * db

    # sigmoid function
    def sigmoid(self):
        return 1 / (1 + np.exp(-(self.forward())))
    
    # forward propagation
    def forward(self):
        return np.matmul(self.X, self.w) + self.b
    
    # prediction process
    def predict(self, X): 
        Y_pred = 1 / (1 + np.exp(-(np.matmul(X, self.w) + self.b)))
        Y_pred = [1 if i > 0.5 else 0 for i in Y_pred]
        return Y_pred
    
    # Modified from Siddhardhan's code.
    # Check: https://www.youtube.com/watch?v=DeUAvYyB0Os&list=PLfFghEzKVmjsF8ixJ-xKVuQayPWRH4Sp6&index=6

            