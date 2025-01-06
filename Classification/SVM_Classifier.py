import numpy as np 

class SVM_Classifier():

    # parameter initialization
    def __init__(self, lambda_param=0.0001, learning_rate=0.001, num_of_iters=1000): 
        self.lambda_param  = lambda_param
        self.learning_rate = learning_rate
        self.num_of_iters  = num_of_iters

    # weight and bias initialization
    def param_init(self): 
        self.w = np.zeros(self.X.shape[1])
        self.b = 0

    # calculate hyper_plane with model's w and b
    def hyper_plane(self, x): 
        return np.dot(x, self.w) - self.b

    # update weight and bias
    def update_weight(self):
        for idx, x in enumerate(self.X): 
            condition = self.label[idx] * (self.hyper_plane(x)) >= 1

            # grad_l2_reg = gradien of L2 regularization term
            grad_L2_reg = 2 * self.lambda_param * self.w
            if condition: 
                dw = grad_L2_reg
                db = 0 

            else: 
                dw = grad_L2_reg - np.dot(x, self.label[idx])
                db = self.label[idx]

            self.w -= self.learning_rate * dw 
            self.b -= self.learning_rate * db 
    
    # fitting process
    def fit(self, X, y):
        self.X = X
        self.Y = y

        self.label = np.where(y <=0, -1, 1)

        self.param_init()

        for i in range(self.num_of_iters):
            self.update_weight()

    # prediction process
    def predict(self, X): 
        output      = np.dot(X, self.w) - self.b
        pred_labels = np.sign(output)
        y_hat  = np.where(pred_labels <= -1, 0, 1)

        return y_hat 

# Modified from Siddhardhan's code.
# Check: https://www.youtube.com/watch?v=pCQQaeC9WRE&list=PLfFghEzKVmjvzS4DILijsdQk27Ew7xIPu