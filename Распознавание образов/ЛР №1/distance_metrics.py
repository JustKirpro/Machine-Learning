import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_name = "/Users/kirpro/Desktop/iris.csv.gz"
dataset = pd.read_csv(file_name)

x = dataset.values[:, 0:4]
y = dataset.values[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


def euclidean_distance(point_1, point_2):
    return sqrt(((point_1 - point_2) ** 2).sum())


def hamming_distance(point_1, point_2):
    return np.absolute(point_1 - point_2).sum()


def manhattan_distance(point_1, point_2):
    return np.absolute(point_1 - point_2).max()


def jaccard_index(point_1, point_2):
    return 1 - len(np.intersect1d(point_1, point_2)) / len(np.union1d(point_1, point_2))


def cosine_similarity(point_1, point_2):
    return 1 - (point_1 * point_2).sum() / (sqrt(np.square(point_1).sum()) * sqrt(np.square(point_2).sum()))


def predict(metric):
    predictions = []
    for test_point in x_test:
        distances = []
        for train_point in x_train:
            distance = metric(train_point, test_point)
            distances = np.append(distances, distance)
        min_index = np.argmin(distances)
        predictions = np.append(predictions, y_train[min_index])
    return predictions


def main():
    predictions = predict(euclidean_distance)
    df = pd.DataFrame({'Настоящие классы': y_test, 'Предсказанные классы': predictions})
    print('Метрика Евклида:\n', df)
    print('Точность: ', accuracy_score(predictions, y_test))

    predictions = predict(hamming_distance)
    df = pd.DataFrame({'Настоящие классы': y_test, 'Предсказанные классы': predictions})
    print('Метрика Хэмминга:\n', df)
    print('Точность: ', accuracy_score(predictions, y_test))

    predictions = predict(manhattan_distance)
    df = pd.DataFrame({'Настоящие классы': y_test, 'Предсказанные классы': predictions})
    print('Метрика городских кварталов:\n', df)
    print('Точность: ', accuracy_score(predictions, y_test))

    predictions = predict(jaccard_index)
    df = pd.DataFrame({'Настоящие классы': y_test, 'Предсказанные классы': predictions})
    print('Метрика Жаккарда:\n', df)
    print('Точность: ', accuracy_score(predictions, y_test))

    predictions = predict(cosine_similarity)
    df = pd.DataFrame({'Настоящие классы': y_test, 'Предсказанные классы': predictions})
    print('Косинусная метрика:\n', df)
    print('Точность: ', accuracy_score(predictions, y_test))


if __name__ == "__main__":
    main()
