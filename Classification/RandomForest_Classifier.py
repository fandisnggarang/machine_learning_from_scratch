import numpy as np
from collections import Counter
from DecTree_Classifier import Decision_Tree_Classifier

class Random_Forest_Classifier:

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
            tree = Decision_Tree_Classifier(
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
        y_pred = [self.most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    
    # create a bootstrap sample of the dataset randomly with replacement
    def bootstrap_sample(self, X, y):
        # Ensure X and y are NumPy arrays
        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    # assign label
    def most_common_label(self, y): 
        uniqe_classes, counts_unique_classes = np.unique(y, return_counts=True) 
        index = counts_unique_classes.argmax() 
        return uniqe_classes[index]
    
# Modified from Random Forest code of patrickloeber.
# Check: https://github.com/patrickloeber/MLfromscratch/blob/master/mlfromscratch/random_forest.py
