import pandas as pd
import numpy as np

class treeNode:
    def __init__(self, threshold=None, feature_index=None, value=None):
        self.threshold = threshold
        self.feature_index = feature_index
        self.value = value
        self.left = None
        self.right = None

    def is_leaf_Node(self):
        return self.value is not None

class XGBoostClassifier:
    def __init__(self, n_estimators=500, learning_rate=0.5, max_depth=6,
                 lamda=3.0, subsample_features=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lamda = lamda
        self.subsample_features = subsample_features
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _g(self, y_true, y_pred):
        return self._sigmoid(y_pred) - y_true

    def _h(self, y_true, y_pred):
        sig = self._sigmoid(y_pred)
        return sig * (1 - sig)

    def _exact_greedysplit_vectorized(self, X_col, y_true, y_pred):
        g = self._g(y_true, y_pred)
        h = self._h(y_true, y_pred)

        # skipping null columns
        nonzero_mask = X_col != 0
        X_nz = X_col[nonzero_mask]
        g_nz = g[nonzero_mask]
        h_nz = h[nonzero_mask]

        if X_nz.size < 2:
            return -np.inf, None

        G_total, H_total = np.sum(g), np.sum(h)
        G_zero = np.sum(g[~nonzero_mask])
        H_zero = np.sum(h[~nonzero_mask])

        sorted_idx = np.argsort(X_nz)
        X_sorted = X_nz[sorted_idx]
        g_sorted = g_nz[sorted_idx]
        h_sorted = h_nz[sorted_idx]

        G_L = G_zero + np.cumsum(g_sorted)
        H_L = H_zero + np.cumsum(h_sorted)
        G_R = G_total - G_L
        H_R = H_total - H_L

        gain = (G_L**2) / (H_L + self.lamda + 1e-6) + (G_R**2) / (H_R + self.lamda + 1e-6) - (G_total**2) / (H_total + self.lamda + 1e-6)

        best_idx = np.argmax(gain)
        best_gain = gain[best_idx]
        best_threshold = X_sorted[best_idx]

        return best_gain, best_threshold

    def _build_tree(self, X, y_true, y_pred, depth):
        n_samples, n_features = X.shape
        if (n_samples < 3) or (depth >= self.max_depth):
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)

        # feature subsampling:
        feature_indices = np.random.choice(
            n_features,
            int(max(1, self.subsample_features * n_features)),
            replace=False
        )

        best_gain, best_threshold, best_feature = -np.inf, None, None

        for feature_index in feature_indices:
            gain, threshold = self._exact_greedysplit_vectorized(X[:, feature_index], y_true, y_pred)
            if gain > best_gain:
                best_gain, best_threshold, best_feature = gain, threshold, feature_index

        if best_gain < 1e-6:
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_mask], y_true[left_mask], y_pred[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y_true[right_mask], y_pred[right_mask], depth + 1)

        node = treeNode(threshold=best_threshold, feature_index=best_feature)
        node.left = left_subtree
        node.right = right_subtree
        return node

    def _predict_tree(self, X, tree):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = tree
            while not node.is_leaf_Node():
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.value
        return y_pred

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_mean = np.mean(y)
        y_pred = np.full(y.shape, np.log(y_mean / (1 - y_mean + 1e-6)))

        for _ in range(self.n_estimators):
            tree = self._build_tree(X, y, y_pred, 0)
            self.trees.append(tree)
            update = self._predict_tree(X, tree)
            y_pred += self.learning_rate * update

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(X, tree)
        y_pred = self._sigmoid(y_pred)
        return y_pred 


class PCA:
    def __init__(self, n_components):
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        # Centering the data here:
        self.mean_ = X.mean(axis=0, keepdims=True) 
        Xc = X - self.mean_
        # Performing SVD:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Taking the top n_components principal directions:
        self.components_ = Vt[:self.n_components].T
        return self

    def transform(self, X):
        """
        Projecting X into PCA space using top singular vectors we obtained in fit function.
        Returns: (n_samples, n_components)
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        Xc = X - self.mean_
        return Xc @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FastWeightedKNN:
    def __init__(self, k=5, n_classes=None, eps=1e-8, dtype=np.float32):
        self.k = int(k)
        self.n_classes = n_classes
        self.eps = eps
        self.dtype = dtype

        self.X_train = None
        self.y_train = None
        self.train_norm2 = None
        self.n_train = None
        self.n_features = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=np.int32)

        self.X_train = np.ascontiguousarray(X)
        self.y_train = y
        self.n_train, self.n_features = self.X_train.shape

        if self.n_classes is None:
            self.n_classes = int(np.max(y) + 1)

        # Precomputing squared norms of training points
        self.train_norm2 = np.sum(self.X_train ** 2, axis=1).astype(self.dtype)
        return self

    def _squared_distances(self, Q):
        """
        Computing full squared distance matrix between query batch Q and all train points.
        """
        Q = np.asarray(Q, dtype=self.dtype)
        q_norm2 = np.sum(Q ** 2, axis=1, keepdims=True)
        cross = Q @ self.X_train.T
        dist2 = q_norm2 + self.train_norm2[None, :] - 2.0 * cross
        return dist2

    def kneighbors(self, Q, return_distance=True):
        """
        Returns k nearest neighbors for query batch Q.
        """
        dist2 = self._squared_distances(Q)
        # getting indices of k smallest distances, but unsorted using argpartition function. This makes finding neighbors quicker than just simple sorting.
        idx_part = np.argpartition(dist2, self.k - 1, axis=1)[:, :self.k]
        dist2_part = np.take_along_axis(dist2, idx_part, axis=1)
        order = np.argsort(dist2_part, axis=1)
        idx_sorted = np.take_along_axis(idx_part,  order, axis=1)
        dist2_sorted = np.take_along_axis(dist2_part, order, axis=1)

        if return_distance:
            return np.sqrt(dist2_sorted), idx_sorted
        return idx_sorted

    def predict_proba(self, X, batch_size=None):
        """
        Returns class probabilities for each sample in X.
        Weighted voting: w_i = 1 / (d_i + eps).
        """
        X = np.asarray(X, dtype=self.dtype)
        n_q = X.shape[0]
        if batch_size is None:
            batch_size = n_q

        probs = np.zeros((n_q, self.n_classes), dtype=self.dtype)

        out_idx = 0
        while out_idx < n_q:
            end = min(out_idx + batch_size, n_q)
            Q_batch = X[out_idx:end]

            dist_batch, idx_batch = self.kneighbors(Q_batch, return_distance=True)
            labels_batch = self.y_train[idx_batch]

            # For each query in the batch:
            b = labels_batch.shape[0]
            for i in range(b):
                labels = labels_batch[i]
                dists = dist_batch[i]

                # Converting distances to weights using inverse distance weighting given in docstring:
                weights = 1.0 / (dists + self.eps)

                scores = np.bincount(
                    labels,
                    weights=weights,
                    minlength=self.n_classes
                ).astype(self.dtype)

                total = scores.sum()
                if total > 0:
                    probs[out_idx + i] = scores / total
                else:
                    probs[out_idx + i] = 1.0 / self.n_classes
            out_idx = end

        return probs

    def predict(self, X, batch_size=None):
        proba = self.predict_proba(X, batch_size=batch_size)
        return np.argmax(proba, axis=1)

