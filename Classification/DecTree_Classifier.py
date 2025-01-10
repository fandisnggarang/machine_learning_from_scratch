import numpy as np
import pandas as pd

# calculate entropy to check quality of splitting
def entropy(y): 
    _, counts = np.unique(y, return_counts=True)
    proba     = counts/len(y) 
    entropy   = -np.sum([p * np.log2(p) for p in proba if p > 0])
    return entropy 

# Make tree nodes
class Node:
    # parameter initialization
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None, feature_type=None
    ):
        self.feature   = feature
        self.threshold = threshold
        self.left  = left
        self.right = right
        self.value = value
        self.feature_type = feature_type

    # check if the node is a leaf node
    def is_leaf_node(self):
        return self.value is not None

class Decision_Tree_Classifier:
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
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # randomly select features for the best split
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh, feature_type = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh, feature_type)

        # handle empty split case
        if len(left_idxs) == 0 or len(right_idxs) == 0: 
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # grow the children that result from the split
        left  = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right, feature_type=feature_type)

    # find best feature and threshold for splitting
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh, feature_type = None, None, None 
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            type_of_feature = self._type_of_feature(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold, type_of_feature)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
                    feature_type = type_of_feature
        return split_idx, split_thresh, feature_type
    
    # calculate information gain for a given split
    def _information_gain(self, y, X_column, split_thresh, feature_type):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh, feature_type)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        proba_left, proba_right     = len(left_idxs)/len(y), len(right_idxs)/len(y)
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = proba_left * entropy_left + proba_right * entropy_right
        # information gain is difference in loss before vs. after split
        return parent_entropy - child_entropy
    
    # split data based on threshold
    def _split(self, X_column, split_thresh, feature_type):
        if feature_type == 'continuous':
            left_idxs  = np.argwhere(X_column <= split_thresh).flatten()
            right_idxs = np.argwhere(X_column > split_thresh).flatten()
        elif feature_type == 'categorical':
            left_idxs  = np.argwhere(X_column == split_thresh).flatten()
            right_idxs = np.argwhere(X_column != split_thresh).flatten()
        else: 
            raise ValueError(f'Unknown feature type: {feature_type}') 
        return left_idxs, right_idxs
    
    # find out whether a column is numerical or categorical
    def _type_of_feature(self, X_column): 
        if pd.api.types.is_numeric_dtype(X_column):
            return 'continuous'
        else: 
            return 'categorical'

    # traverse the tree for predictions
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
            
        if node.feature_type == 'categorical':
            if x[node.feature] == node.threshold: 
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else: 
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)

    # assign label
    def _most_common_label(self, y):
        unique_classes, counts_unique_classes = np.unique(y, return_counts=True)
        index = counts_unique_classes.argmax() 
        leaf  = unique_classes[index]
        return leaf
    
# Modified from code of janaSunrise/patrickloeber and SebastianMantey.
# Check: https://github.com/patrickloeber/MLfromscratch/blob/master/mlfromscratch/decision_tree.py
# Check: https://github.com/SebastianMantey/Decision-Tree-from-Scratch/blob/master/notebooks/decision_tree_functions.py
