import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, learning_rate=0.01, alpha=0.1, epoch=100):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epoch = epoch
        self.w = None
        self.b = None

    def fit(self, x_train, y_train):
        self.w = np.zeros(x_train.shape[1])
        self.b = 0
        for _ in range(self.epoch):
            for i, x in enumerate(x_train):
                margin = y_train[i] * (np.dot(x, self.w) - self.b)
                if margin >= 1:
                    self.w -= self.learning_rate * (2 * self.alpha * self.w)
                else:
                    self.w -= self.learning_rate * (2 * (self.alpha * self.w - np.dot(y_train[i], x)))
                    self.b -= self.learning_rate * y_train[i]

    def predict(self, x_test):
        return np.sign(np.dot(x_test, self.w) - self.b)


def main():
    file_name = "/Users/kirpro/Desktop/iris.csv.gz"
    dataset = pd.read_csv(file_name, nrows=100)
    dataset['species'] = dataset['species'].apply(lambda x: 1 if x == 'setosa' else -1)

    x = dataset.values[:, 0:4]
    y = dataset.values[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    svm = SVM()
    svm.fit(x_train, y_train)
    predictions = svm.predict(x_test)
    result = pd.DataFrame({'Real classes': y_test, 'Predicted classes': predictions})
    print(result, '\nAccuracy: ', accuracy_score(predictions, y_test))


if __name__ == '__main__':
    main()
    