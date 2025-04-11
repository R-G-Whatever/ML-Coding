import numpy as np

class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, split_threshold):
        left_ind = np.argwhere(X_column <= split_threshold).flatten()
        right_ind = np.argwhere(X_column > split_threshold).flatten()
        return left_ind, right_ind

    def information_gain(self, y, X_column, threshold):
        parent_entropy = self.entropy(y)
        left_ind, right_ind = self._split(X_column, threshold)
        if len(left_ind) == 0 or len(right_ind) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_ind), len(right_ind)
        e_left, e_right = self.entropy(y[left_ind]), self.entropy(y[right_ind])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def _get_split(self, X, y):
        best_gain = -1
        split_ind, split_thres = None, None

        num_feat = X.shape[1]
        for ind in range(num_feat):
            X_col = X[:, ind]
            thres = np.unique(X_col)
            for t in thres:
                gain = self.information_gain(y, X_col, t)

                if gain > best_gain:
                    best_gain = gain
                    split_ind, split_thres = ind, t

        return split_ind, split_thres

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def fit(self, X, y, depth=0):
        num_sample, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_sample < self.min_sample_split:
            leaf_value = self._most_common_label(y)
            return {"type": 'leaf', "value": leaf_value}

        feature_ind, thres = self._get_split(X, y)
        if feature_ind is None:
            leaf_value = self._most_common_label(y)
            return {'type': 'leaf', "value": leaf_value}

        left_index, right_index = self._split(X[:, feature_ind], thres)
        left_subtree = self.fit(X[left_index, :], y[left_index], depth + 1)
        right_subtree = self.fit(X[right_index, :], y[right_index], depth + 1)

        return {
            'type': 'node',
            'feature_index': feature_ind,
            'threshold': thres,
            'left': left_subtree,
            'right': right_subtree
        }

    def predict_single(self, x):
        node = self.tree
        while node['type'] != 'leaf':
            if x[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']

    def predict(self, X):
        return [self.predict_single(x) for x in X]

if __name__ == "__main__":
    X = np.array([
        [2.771244718, 1.784783929],
        [1.728571309, 1.169761413],
        [3.678319846, 2.81281357],
        [3.961043357, 2.61995032],
        [2.999208922, 2.209014212],
        [7.497545867, 3.162953546],
        [9.00220326, 3.339047188],
        [7.444542326, 0.476683375],
        [10.12493903, 3.234550982],
        [6.642287351, 3.319983761]
    ])

    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Initialize and train the decision tree
    tree = DecisionTree(min_sample_split=2, max_depth=3)
    tree.tree = tree.fit(X, y)  # Set the tree attribute after fitting

    # Test predictions
    test_samples = np.array([
        [1.5, 1.8],  # Should predict 0
        [8.5, 3.0]  # Should predict 1
    ])

    prediction = tree.predict(test_samples)
    print(prediction)