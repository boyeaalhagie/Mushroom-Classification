"""
Author: Daniel Fischer 

"""
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


class RotationForest:
    def __init__(self, n_trees, n_subsets, base_model=DecisionTreeClassifier):
        self.n_trees = n_trees
        self.n_subsets = n_subsets
        self.base_model = base_model
        self.models = []
        self.pca_matrices = []

    def split_features(self, X):
        """
        Randomly split features into subsets.
        """
        n_features = X.shape[1]
        subsets = np.array_split(np.random.permutation(n_features), self.n_subsets)
        return subsets

    def apply_pca(self, X, feature_indicies):
        """
        Applay PCA to the selected feature subset.
        """
        pca = PCA()
        X_subset = X[:, feature_indicies]
        X_transformed = pca.fit_transform(X_subset)
        return X_transformed, pca
    
    def fit(self, X, y):
        self.models = []
        self.pca_matrices = []

        for _ in range(self.n_trees):
            # Split features into subsets
            subsets = self.split_features(X)
            transformed_features = []
            pca_per_tree = []
            
            for subset in subsets:
                # Apply PCA to each subset
                X_transformed, pca = self.apply_pca(X, subset)
                transformed_features.append(X_transformed)
                pca_per_tree.append((subset, pca))

        # Combine transformed features
        X_rotated = np.hstack(transformed_features)
        model = clone(self.base_model)
        model.fit(X_rotated, y)
        self.models.append(model)
        self.pca_matrices.append(pca_per_tree)

    def transform_sample(self, X, pca_matrices):
        """Transform a sample using stored PCA matrices."""
        transformed_features = []
        for subset, pca in pca_matrices:
            X_subset = X[:, subset]
            transformed_features.append(pca.transform(X_subset))
        
        return np.hstack(transformed_features)
    
    def predict(self, X):
        predictions = []
        
        for model, pca_matrices in zip(self.models, self.pca_matrices):
            # Transform the dataset
            X_rotated = self.transform_sample(X, pca_matrices)
            predictions.append(model.predict(X_rotated))

        # Aggregate predictions (majority vote)
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax, axis=0, arr=predictions)