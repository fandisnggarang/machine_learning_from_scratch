import numpy as np
from DecTree_Regressor import Decision_Tree_Regressor
from collections import Counter

class Random_Forest_Regressor:
    
    # initialize parameter
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    # fitting process with decision tree
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = Decision_Tree_Regressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_samp, y_samp = self.bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    # prediction process
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [np.mean(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    
    # create a bootstrap sample of the dataset randomly with replacement
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True) 
        return X[idxs], y[idxs] 
    
# Modified from Random Forest code of patrickloeber.
# Check: https://github.com/patrickloeber/MLfromscratch/blob/master/mlfromscratch/random_forest.py