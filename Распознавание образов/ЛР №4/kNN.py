import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


def euclidean_distance(point_1, point_2):
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


class KNN:
    def __init__(self, k=3, metric=euclidean_distance):
        self.k = k
        self.metric = metric

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        return np.array([self.__predict(x) for x in x_test])

    def __predict(self, x_test):
        distances = [self.metric(x_test, x) for x in self.x_train]
        indices = np.argsort(distances)[:self.k]
        labels = [self.y_train[i] for i in indices]
        return Counter(labels).most_common()[0][0]


def display_results(y_test, predictions):
    result = pd.DataFrame({'Real classes': y_test, 'Predicted classes': predictions})
    with pd.option_context('display.colheader_justify', 'left'):
        print(result, '\nAccuracy: ', accuracy_score(predictions, y_test))


def main():
    file_name = "/Users/kirpro/Desktop/iris.csv.gz"
    dataset = pd.read_csv(file_name)

    x = dataset.values[:, 0:4]
    y = dataset.values[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    classifier_1 = KNN(k=8)
    classifier_1.fit(x_train, y_train)
    predictions_1 = classifier_1.predict(x_test)
    display_results(y_test, predictions_1)

    classifier_2 = KNeighborsClassifier(n_neighbors=3)
    classifier_2.fit(x_train, y_train)
    predictions_2 = classifier_2.predict(x_test)
    display_results(y_test, predictions_2)


if __name__ == '__main__':
    main()
