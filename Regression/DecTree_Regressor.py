import numpy as np

# calculate MSE for a given set of target values
def mse(y): 
    if len(y) == 0: 
        mse = 0 
    else: 
        prediction = np.mean(y)
        mse = np.mean((y - prediction)**2)
    return mse

# Make tree nodes
class Node:

    # parameter initialization
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature   = feature
        self.threshold = threshold
        self.left  = left
        self.right = right
        self.value = value

    # check if the node is a leaf node
    def is_leaf_node(self):
        return self.value is not None

class Decision_Tree_Regressor:

    # parameter initialization
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats   = n_feats
        self.root      = None

    # fitting process
    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    # prediction process
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # grow decision tree recursively
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria for growing the tree
        if (
            depth >= self.max_depth
            or np.var(y) < 1e-7
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # randomly select features for the best split
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left  = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    # find best feature and threshold for splitting
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh
    
    # calculate information gain for a given split
    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = mse(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        proba_left, proba_right= len(left_idxs)/len(y), len(right_idxs)/len(y)
        mse_left, mse_right    = mse(y[left_idxs]), mse(y[right_idxs])
        child_mse = proba_left * mse_left + proba_right * mse_right

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_mse
        return ig

    # split data based on threshold
    def _split(self, X_column, split_thresh):
        left_idxs  = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        # handle edge case where one split is empty
        if len(left_idxs) == 0 or len(right_idxs) == 0: 
            left_idxs = np.arange(len(X_column))
            right_idsx= []
        return left_idxs, right_idxs

    # traverse the tree for predictions
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    # calculate the value for a leaf node
    def _leaf_value(self, y):
        return np.mean(y)

# Adapted from Decision Tree code of janaSunrise/patrickloeber and SebastianMantey.
# Check: https://github.com/patrickloeber/MLfromscratch/blob/master/mlfromscratch/decision_tree.py
# Check: https://github.com/SebastianMantey/Decision-Tree-from-Scratch/blob/master/notebooks/decision_tree_functions.py