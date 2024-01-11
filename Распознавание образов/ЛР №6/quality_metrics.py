import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
from sklearn import preprocessing, svm
from collections import Counter
from sklearn.cluster import KMeans


def my_accuracy_score(y_true, y_pred):
    return np.sum(np.equal(y_true, y_pred)) / len(y_true)


def my_precision_score(y_true, y_pred):
    cm = my_confusion_matrix(y_true, y_pred)
    precision = 0
    for i in range(cm.shape[0]):
        column_sum = np.sum(cm[:,i])
        precision += (cm[i, i] / column_sum if column_sum != 0 else 0) * np.sum(cm[i])
    return precision / np.sum(cm)


def my_recall_score(y_true, y_pred):
    cm = my_confusion_matrix(y_true, y_pred)
    recall = 0
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i])
        recall += (cm[i, i] / row_sum if row_sum != 0 else 0) * row_sum
    return recall / np.sum(cm)


def my_f1_score(y_true, y_pred):
    cm = my_confusion_matrix(y_true, y_pred)
    f1 = 0
    for i in range(cm.shape[0]):
        column_sum = np.sum(cm[:,i])
        precision = cm[i, i] / column_sum if column_sum != 0 else 0
        row_sum = np.sum(cm[i])
        recall = cm[i, i] / row_sum if row_sum != 0 else 0
        f1 += 2 * precision * recall / (precision + recall) * row_sum
    return f1 / np.sum(cm)


def my_roc_curve(y_true, y_pred):
    cm = my_confusion_matrix(y_true, y_pred)
    tpr = np.trace(cm) / np.sum(cm)
    fpr = (np.sum(cm) - np.trace(cm)) / np.sum(cm)
    plt.plot([0,fpr,1],[0,tpr,1])
    plt.title('ROC-curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def my_cross_validation_score(classifier, x, y, k=5):
    accuracy = []
    samples_per_step = len(x) // k
    valid_indices = np.arange(0, samples_per_step)
    for _ in range(k):
        x_valid = x.take(valid_indices, axis=0)
        y_valid = y.take(valid_indices, axis=0)
        x_train = np.delete(x, valid_indices, axis=0)
        y_train = np.delete(y, valid_indices, axis=0)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_valid)
        accuracy.append(my_accuracy_score(y_valid, predictions))
        valid_indices += samples_per_step
    return accuracy


def my_mean_squared_error(y_true, y_pred):
    return np.sum(np.square(y_true != y_pred)) / len(y_true)


def my_confusion_matrix(y_true, y_pred):
    labels=np.union1d(np.unique(y_true), np.unique(y_pred))
    le = preprocessing.LabelEncoder().fit(labels)
    matrix = np.zeros((len(labels), len(labels)), dtype='uint8')
    for true, pred in zip(le.transform(y_true), le.transform(y_pred)):
        matrix[true, pred] += 1
    return matrix


    return np.sqrt(np.sum((point_1 - point_2) ** 2))


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
        indices = np.argsort(distances) [:self.k]
        labels = [self.y_train[i] for i in indices]
        return Counter(labels).most_common()[0][0]


def my_r2_score(data, labels, centroids):
    data = data.to_numpy()
    rss = 0
    for index, point in enumerate(data):
        rss += np.square(centroids[labels[index]] - point)
    tss = 0
    mean = np.mean(data)
    for point in data:
        tss += np.square((point - mean))
    return 1 - np.mean(rss)/np.mean(tss)


def my_silhouette_score(data, labels, metric=euclidean_distance):
    data = data.to_numpy()
    scores = []
    for index, point in enumerate(data):
        indices = np.where(labels == labels[index])[0]
        a = np.mean([metric(point, p) for p in np.take(data, indices, axis=0) if not np.array_equal(point, p)])
        current_labels = np.delete(labels, indices, axis=0)
        nearest_cluster = current_labels[np.argmin([metric(point, p) for p in np.delete(data, indices, axis=0)])]
        indices = np.where(labels == nearest_cluster)[0]
        b = np.mean([metric(point, p) for p in np.take(data, indices, axis=0)])
        scores.append((b - a) / max(a, b))
    return np.mean(scores)


def my_dunn_score(data, labels, centroids, metric=euclidean_distance):
    data = data.to_numpy()
    max_distance = 0
    for i in range(len(data)):
        for j in range(len(data)):
            distance = metric(data[i], data[j])
            if labels[i] == labels[j] and distance > max_distance:
                max_distance = distance

    min_distance = np.inf
    for centroid_1 in centroids:
        for centroid_2 in centroids:
            distance = metric(centroid_1, centroid_2)
            if not np.array_equal(centroid_1, centroid_2) and distance < min_distance:
                min_distance = distance

    return min_distance / max_distance


def my_davies_bouldin_score(data, labels, centroids, metric=euclidean_distance):
    data = data.to_numpy()
    n = len(centroids)

    clusters = [[] for _ in range(n)]
    for index, point in enumerate(data):
        clusters[labels[index]].append(point)

    s = []
    for i in range(n):
        s.append(np.sum([metric(centroids[i], point) for point in clusters[i]]) / len(clusters[i]))

    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = metric(centroids[i], centroids[j])

    d = []
    r = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r[i][j] = (s[i] + s[j]) / m[i][j]
        d.append(np.max(r[i]))

    return np.sum(d) / n


def calc_avg_distance(centroids):
    distance = 0
    for c_1 in centroids:
        for c_2 in centroids:
            if c_2 > c_1:
                distance += euclidean_distance(dataset.iloc[c_1], dataset.iloc[c_2])

    return distance / len(centroids)


def find_centroids():
    centers = []
    points = np.arange(len(dataset))

    center = rnd.randint(0, len(dataset) - 1)
    centers.append(center)
    points = np.delete(points, center)

    while True:
        max_min = -1
        for point in points:
            distances = []
            for center in centers:
                distance = euclidean_distance(dataset.iloc[center], dataset.iloc[point])
                distances.append(distance)
            min_distance = min(distances)
            if min_distance > max_min:
                max_min = min_distance
                new_center = point

        centers.append(new_center)
        points = np.delete(points, new_center)

        if max_min <= calc_avg_distance(centers):
            break

    return centers


def max_min_clustering():
    centers = find_centroids()
    dataset_1 = dataset.to_numpy()
    labels = np.zeros(len(dataset_1), dtype='uint8')
    number = 0
    for index in range(len(centers)):
        labels[centers[index]] = number
        number += 1
    for i in range(len(dataset_1)):
        labels[i] = np.argmin([euclidean_distance(dataset_1[i], dataset_1[j]) for j in range(len(centers))])
    return np.take(dataset_1, centers, axis=0), labels

if __name__ == '__main__':
    file_name = "/Users/kirpro/Desktop/iris.csv.gz"
    dataset = pd.read_csv(file_name)
    x = dataset.values[:, :4]
    y = dataset.values[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    classifier = KNN()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print('KNN classifier metrics:')
    print(f'Confusion matrix:\n{my_confusion_matrix(y_test, predictions)}')
    print(f'Accuracy: {my_accuracy_score(y_test, predictions)}')
    print(f'Precision: {my_precision_score(y_test, predictions)}')
    print(f'Recall: {my_recall_score(y_test, predictions)}')
    print(f'F1-score: {my_f1_score(y_test, predictions)}')
    print(f'MSE: {my_mean_squared_error(y_test, predictions)}')
    print(f'Cross validation score:\n{my_cross_validation_score(classifier, x_train, y_train)}')
    my_roc_curve(y_test, predictions)

    print('sklearn metrics:')
    print(f'Confusion matrix:\n{confusion_matrix(y_test, predictions)}')
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(f'Precision: {precision_score(y_test, predictions, average="weighted")}')
    print(f'Recall: {recall_score(y_test, predictions, average="weighted")}')
    print(f'F1-score: {f1_score(y_test, predictions, average="weighted")}')

    dataset = pd.read_csv(file_name, nrows=100)
    dataset['species'] = dataset['species'].apply(lambda x: 1 if x == 'setosa' else -1)
    x = dataset.iloc[:, 0:4]
    y = dataset.iloc[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    classifier = svm.SVC(kernel='linear', C=1.0)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print('SVM classifier metrics:')
    print(f'Confusion matrix:\n{my_confusion_matrix(y_test, predictions)}')
    print(f'Accuracy: {my_accuracy_score(y_test, predictions)}')
    print(f'Precision: {my_precision_score(y_test, predictions)}')
    print(f'Recall: {my_recall_score(y_test, predictions)}')
    print(f'F1-score: {my_f1_score(y_test, predictions)}')
    print(f'MSE: {my_mean_squared_error(y_test, predictions)}')
    print(f'Cross validation score:\n{my_cross_validation_score(classifier, x_train.to_numpy(), y_train.to_numpy())}')
    my_roc_curve(y_test, predictions)

    print('sklearn metrics:')
    print(f'Confusion matrix:\n{confusion_matrix(y_test, predictions)}')
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(f'Precision: {precision_score(y_test, predictions, average="weighted")}')
    print(f'Recall: {recall_score(y_test, predictions, average="weighted")}')
    print(f'F1-score: {f1_score(y_test, predictions, average="weighted")}')

    fileName = "/Users/kirpro/Desktop/iris.csv.gz"
    dataset = pd.read_csv(fileName, usecols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y_true = pd.read_csv(fileName, usecols=['species'])

    kmeans = KMeans(n_clusters=3).fit(dataset)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print('KMeans metrics:')
    print(f'Silhouette score:{my_silhouette_score(dataset, labels)}')
    print(f'Davies-Bouldin score: {my_davies_bouldin_score(dataset, labels, centroids)}')
    print(f'Dunn score: {my_dunn_score(dataset, labels, centroids)}')
    print(f'R2 score: {my_r2_score(dataset, labels, centroids)}')

    print('sklearn metrics:')
    print(f'Silhouette score:{silhouette_score(dataset, labels)}')
    print(f'Davies-Bouldin score: {davies_bouldin_score(dataset, labels)}')

    print('MaxMin metrics:')
    print(f'Silhouette score:{my_silhouette_score(dataset, labels)}')
    print(f'Davies-Bouldin score: {my_davies_bouldin_score(dataset, labels, centroids)}')
    print(f'Dunn score: {my_dunn_score(dataset, labels, centroids)}')
    print(f'R2 score: {my_r2_score(dataset, labels, centroids)}')