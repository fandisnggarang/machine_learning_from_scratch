import numpy as np 

class SVr_(): 
    # parameter initialization
    def __init__(self, lambda_param, learning_rate, num_of_iters, epsilon): 
        self.lambda_param  = lambda_param
        self.learning_rate = learning_rate
        self.num_of_iters  = num_of_iters
        self.epsilon       = epsilon 

    # weight and bias initialization
    def param_init(self): 
        self.w = np.zeros(self.X.shape[1])
        self.b = 0

    # calculate hyper_plane with model's w and b
    def hyper_plane(self, x): 
        return np.dot(x, self.w) + self.b

    # update weight and bias
    def update_weight(self):
        for i in range(self.X.shape[0]): 
            margin = (self.hyper_plane(self.X[i])) - self.Y[i]
        
            # grad_l2_reg = gradien of L2 regularization term
            grad_L2_reg = 2 * self.lambda_param * self.w

            if margin > self.epsilon:
                dw = grad_L2_reg + self.X[i]
                db = margin - self.epsilon
            elif margin < -self.epsilon: 
                dw = grad_L2_reg - self.X[i]
                db = margin + self.epsilon
            else:
                dw = grad_L2_reg
                db = 0

            self.w -= self.learning_rate * dw 
            self.b -= self.learning_rate * db 
    
    # fitting process
    def fit(self, X, y):
        self.X = X
        self.Y = y
        self.param_init()
        for i in range(self.num_of_iters):
            self.update_weight()

    # prediction process
    def predict(self, X): 
        y_hat      = np.dot(X, self.w) + self.b
        return y_hat 

# Adapted from Siddhardhan's SVM Classifier code.
# Check: https://www.youtube.com/watch?v=pCQQaeC9WRE&list=PLfFghEzKVmjvzS4DILijsdQk27Ew7xIPu

    