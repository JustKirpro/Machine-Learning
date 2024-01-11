import numpy as np
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans

file_name = "/Users/kirpro/Desktop/iris.csv.gz"
dataset = pd.read_csv(file_name, usecols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])


def euclidean_distance(point_1, point_2):
    return sqrt(((point_1 - point_2) ** 2).sum())


def k_means_clustering(k):
    points = np.arange(len(dataset))
    centers = dataset.sample(3).to_numpy()

    while True:
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = []
            for center in centers:
                distance = euclidean_distance(dataset.iloc[point], center)
                distances.append(distance)
            clusters[np.argmin(distances)].append(point)

        new_centers = []
        for cluster in clusters:
            new_center = [0 for _ in range(len(dataset.iloc[point]))]
            for point in cluster:
                for feature_index in range(len(dataset.iloc[point])):
                    new_center[feature_index] += dataset.iloc[point, feature_index] / len(cluster)
            new_centers.append(new_center)

        if np.all(new_centers == centers):
            break

        centers = new_centers

    return centers, clusters


def main():
    centers, clusters = k_means_clustering(3)

    for center in centers:
        print(center)

    for point in range(len(dataset)):
        for cluster_number in range(3):
            if point in clusters[cluster_number]:
                if point % 37 == 0:
                    print()
                print(cluster_number, end=' ')

    kmeans = KMeans(n_clusters=3).fit(dataset)
    for center in kmeans.cluster_centers_:
        print(center)
    print(kmeans.predict(dataset))


if __name__ == "__main__":
    main()
