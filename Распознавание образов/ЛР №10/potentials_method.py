import os
from os import path
import cv2 as cv
import numpy as np
import csv


def get_potential_map(image):
    n = image.shape[0] - 1
    map = np.zeros(image.shape, dtype=float)

    for i in range(n):
        for j in range(n):
            map[i][j] += get_point_potential(image, i, j, n)

    return map


def get_point_potential(image, x, y, n):
    sum = 0

    if image[x][y] != 255:
        sum += 1
    if x - 1 >= 0 and y - 1 >= 0 and image[x-1][y-1] != 255:
        sum += 1/12
    if y - 1 >= 0 and image[x][y-1] != 255:
        sum += 1/6
    if y - 1 >= 0 and x + 1 <= n and image[x+1][y-1] != 255:
        sum += 1/12
    if x - 1 >= 0 and image[x-1][y] != 255:
        sum += 1/6
    if x + 1 <= n and image[x+1][y] != 255:
        sum += 1/6
    if x - 1 >= 0 and y + 1 <= n and image[x-1][y+1] != 255:
        sum += 1/12
    if y + 1 >= 0 and image[x][y+1] != 255:
        sum += 1/6
    if y + 1 >= 0 and x + 1 <= n and image[x+1][y+1] != 255:
        sum += 1/12

    return sum


def get_potential_maps(train_path):
    maps = dict()
    for character in os.listdir(train_path):
        if not character.startswith('.'):
            if character not in maps.keys():
                maps[character] = []
            character_path = path.join(train_path, character)
            for file in os.listdir(character_path):
                if not file.startswith('.'):
                    file_path = path.join(character_path, file)
                    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    maps[character].append(get_potential_map(image))
    return maps


def save_model(maps):
    with open('/Users/kirpro/Kirpro/8 триместр/Распознавание образов/maps.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['character', 'map'])
        for key in maps.keys():
            for value in maps[key]:
                writer.writerow([key, value.tostring()])


def load_model():
    maps = dict()
    with open('/Users/kirpro/Kirpro/8 триместр/Распознавание образов/maps.csv', mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['character'] not in maps.keys():
                maps[row['character']] = []
            elems = np.fromstring(row['map'], dtype=float)
            n = int(np.sqrt(elems.shape[0]))
            maps[row['character']].append(elems.reshape(n, n))
    return maps


def classify_image(maps, map):
    min_sum = np.inf
    character = ''
    for key in maps.keys():
        for value in maps[key]:
            difference = np.sum(np.abs(map - value))
            if difference < min_sum:
                min_sum = difference
                character = key
    return character


def classify_images(maps, test_path):
    tot, cor = 0, 0
    for character in os.listdir(test_path):
        if not character.startswith('.'):
            total, correct = 0, 0
            character_path = path.join(test_path, character)
            for file in os.listdir(character_path):
                if not file.startswith('.'):
                    file_path = path.join(character_path, file)
                    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    map = get_potential_map(image)
                    predict = classify_image(maps, map)
                    if predict == character:
                        correct += 1
                        cor += 1
                    total += 1
                    tot += 1
            print(f'Class: {character}, accuracy = {round(100 * correct / total, 2)}%')
    print(round(100 * cor / tot, 2))


if __name__ == "__main__":
    train_path = '/Users/kirpro/Kirpro/8 триместр/Распознавание образов/preprocessed train'
    maps = get_potential_maps(train_path)
    
    test_path = '/Users/kirpro/Kirpro/8 триместр/Распознавание образов/preprocessed test'
    classify_images(maps, test_path)
