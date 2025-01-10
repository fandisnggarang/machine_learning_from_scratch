import numpy as np

# create decision stump with weak model
class DecisionStump:
    # parameter initialization
    def __init__(self): 
        self.feature_idx= None 
        self.threshold  = None 
        self.mean_left  = None 
        self.mean_right = None 

    # prediction process
    def predict(self, X): 
        n_samples  = X.shape[0]
        X_column   = X[:, self.feature_idx]
        predictions= np.zeros(n_samples) 

        predictions[X_column < self.threshold]  = self.mean_left
        predictions[X_column >= self.threshold] = self.mean_right

        # handling potentian nan values
        predictions = np.nan_to_num(predictions, nan=0.0) 
        return predictions 

class Adaboost_Regressor:
    # parameter initialization 
    def __init__(self, n_clf=5): 
        self.n_clf = n_clf 
        self.clfs  = []

    # fitting process
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        # iterate through weak classifiers (decision stump)
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            # find out the best threshold and feature
            for feature_i in range(n_features):
                X_column  = X[:, feature_i]
                thresholds= np.unique(X_column)
                for threshold in thresholds:
                    # initialize predictions array filled with zeros 
                    predictions = np.zeros(n_samples)
                    if np.any(X_column < threshold):
                        mean_left = np.mean(y[X_column < threshold])
                    else:
                        mean_left = 0

                    if np.any(X_column >= threshold):
                        mean_right = np.mean(y[X_column >= threshold])
                    else:
                        mean_right = 0

                    predictions[X_column < threshold]  = mean_left
                    predictions[X_column >= threshold] = mean_right
                    
                    # handle NaN in predictions 
                    predictions = np.nan_to_num(predictions, nan=0.0)

                    # calculate weighted squared error
                    error = np.sum(w * (predictions - y) ** 2)

                    # store the best configuration
                    if error < min_error:
                        clf.feature_idx= feature_i
                        clf.threshold  = threshold
                        clf.mean_left  = mean_left
                        clf.mean_right = mean_right
                        min_error      = error

            # calculate alpha
            EPS = 1e-10
            min_error = max(EPS, min(min_error, 1 - EPS))
            clf.alpha = 0.5 * np.log((1.0 - min_error) / min_error)

            # calculate predictions and update weights
            predictions = clf.predict(X)
            residual    = y - predictions
            w *= np.exp(-clf.alpha * np.abs(residual))

            # handle NaN in weights 
            w = np.nan_to_num(w, nan=1/n_samples)

            # reset weights if they sum to zero
            if np.sum(w) == 0: 
                w = np.full(n_samples, (1/n_samples))

            # normalize weights
            w /= np.sum(w)  

            # save the classifier
            self.clfs.append(clf)

    # prediction process
    def predict(self, X): 
        clf_preds = []
        for clf in self.clfs:
            pred = clf.alpha * clf.predict(X)
            clf_preds.append(pred)
        y_pred = np.sum(clf_preds, axis=0)

        # handle NaN in final predictions
        y_pred = np.nan_to_num(y_pred, nan=0.0)
        return y_pred
    
# Adapted from adaboost code of patrickloeber.
# Check: https://github.com/patrickloeber/MLfromscratch/blob/master/mlfromscratch/adaboost.py



