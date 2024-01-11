import numpy as np
import pandas as pd
import random as rnd
from math import sqrt

file_name = "/Users/kirpro/Desktop/iris.csv.gz"
dataset = pd.read_csv(file_name)
dataset = dataset.set_index('species')


def euclidean_distance(point_1, point_2):
    return sqrt(((point_1 - point_2) ** 2).sum())


def calc_avg_distance(centers):
    distance = 0
    for c_1 in centers:
        for c_2 in centers:
            if c_2 > c_1:
                distance += euclidean_distance(dataset.iloc[c_1], dataset.iloc[c_2])

    return distance / len(centers)


def find_centers():
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
    centers = find_centers()
    points = np.arange(len(dataset))
    cluster_numbers = []
    for point in points:
        min_distance = float('inf')
        cluster_number = 0
        for center in centers:
            distance = euclidean_distance(dataset.iloc[center], dataset.iloc[point])
            if distance < min_distance:
                min_distance = distance
                cluster_number = center
        cluster_numbers.append(cluster_number)
    dataset['cluster'] = cluster_numbers
    return centers


def main():
    centers = max_min_clustering()
    print('Центры кластеров:', centers)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('Результаты кластеризации:\n', dataset)


if __name__ == "__main__":
    main()
