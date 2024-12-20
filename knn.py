import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.values
        self.y_train = y

    def predict(self, X):
        X_test = X.values
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = []
        for i in k_indices:
            if i < len(self.y_train):
                k_nearest_labels.append(self.y_train.iloc[i])
            else:
                print(f"Indeks {i} di luar batas")

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]