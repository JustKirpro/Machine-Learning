import os
from os import path
import cv2 as cv
import numpy as np
import re
import csv
import difflib


def get_start_position(image):
    character_pixels = np.array(np.where(image != 255))
    column_number = np.min(character_pixels[1])
    row_number = character_pixels[0][np.max(np.where(character_pixels[1] == column_number))]
    return row_number, column_number


def bypass_image(image, row, column, visited, n):
    visited[row][column] = 1
    neighbours, positions = get_neighbours(image, row, column, visited, n)
    path = ''

    for i in range(len(neighbours)):
        visited[positions[i][0]][positions[i][1]] = 2

    for i in range(len(neighbours)):
        if visited[positions[i][0]][positions[i][1]] != 1:
            if len(neighbours) > 1:
                path += '('
            path += str(neighbours[i]) + bypass_image(image, positions[i][0], positions[i][1], visited, n)
            if len(neighbours) > 1:
                path += ')'

    return path


def get_neighbours(image, row, column, visited, n):
    neighbours = []
    positions = []

    if row - 1 >= 0 and visited[row-1][column] == 0 and image[row-1][column] != 255:
        neighbours.append(1)
        positions.append((row - 1, column))
    if row - 1 >= 0 and column + 1 <= n and visited[row-1][column+1] == 0 and image[row-1][column+1] != 255:
        neighbours.append(2)
        positions.append((row - 1, column + 1))
    if column + 1 <= n and visited[row][column+1] == 0 and image[row][column+1] != 255:
        neighbours.append(3)
        positions.append((row, column + 1))
    if row + 1 <= n and column + 1 <= n and visited[row+1][column+1] == 0 and image[row+1][column+1] != 255:
        neighbours.append(4)
        positions.append((row + 1, column + 1))
    if row + 1 <= n and visited[row+1][column] == 0 and image[row+1][column] != 255:
        neighbours.append(5)
        positions.append((row + 1, column))
    if row + 1 <= n and column - 1 >= 0 and visited[row+1][column-1] == 0 and image[row+1][column-1] != 255:
        neighbours.append(6)
        positions.append((row + 1, column - 1))
    if column - 1 >= 0 and visited[row][column-1] == 0 and image[row][column-1] != 255:
        neighbours.append(7)
        positions.append((row, column - 1))
    if row - 1 >= 0 and column - 1 >= 0 and visited[row-1][column-1] == 0 and image[row-1][column-1] != 255:
        neighbours.append(8)
        positions.append((row - 1, column - 1))

    return neighbours, positions


def simplify_path(path):
    path = replace_sequence(path)
    pathChanged = True
    while pathChanged:
        path, pathChanged = replace_characters_between(path)
        path = replace_sequence(path)
    return path


def replace_sequence(path):
    for i in range(1, 9):
        pattern = str(i) + '{2,}'
        path = re.sub(pattern, str(i), path)
    return path


def replace_characters_between(path):
    pathChanged = False
    new_path = ''
    prev, cur = '', ''

    for next in path:
        if cur == '':
            cur = next
        elif prev == '':
            prev = cur
            cur = next
        elif prev in '()' or cur in '()' or next in '()':
            new_path += prev
            prev = cur
            cur = next
        elif next == prev:
            new_path += next
            prev = cur = ''
            pathChanged = True
        else:
            new_path += prev
            prev = cur
            cur = next

    new_path += prev + cur

    return new_path, pathChanged


def get_grammatic(path):
    path = path.replace('(', '+(')
    path = path.replace('1', '+a')
    path = path.replace('2', '+c')
    path = path.replace('3', '+b')
    path = path.replace('4', '+d')
    path = path.replace('5', '-a')
    path = path.replace('6', '-c')
    path = path.replace('7', '-b')
    path = path.replace('8', '-d')
    path = path.replace('(+', '(')
    return path if path[0] != '+' else path[1:]


def process_image(image):
    start_row, start_column = get_start_position(image)
    visited = np.zeros(image.shape, dtype='uint8')
    n = image.shape[0] - 1
    path = bypass_image(image, start_row, start_column, visited, n)
    path = simplify_path(path)
    return get_grammatic(path)


def get_grammatics(train_path):
    grammatics = dict()
    for character in os.listdir(train_path):
        if not character.startswith('.'):
            if character not in grammatics.keys():
                grammatics[character] = []
            character_path = path.join(train_path, character)
            for file in os.listdir(character_path):
                if not file.startswith('.'):
                    file_path = path.join(character_path, file)
                    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    grammatics[character].append(process_image(image))
    return grammatics


def save_model(grammatics):
    with open('/Users/kirpro/Kirpro/8 триместр/Распознавание образов/grammatics.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['character', 'grammatic'])
        for key in grammatics.keys():
            for value in grammatics[key]:
                writer.writerow([key, value])


def load_model():
    grammatics = dict()
    with open('/Users/kirpro/Kirpro/8 триместр/Распознавание образов/grammatics.csv', mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['character'] not in grammatics.keys():
                grammatics[row['character']] = []
            grammatics[row['character']].append(row['grammatic'])
    return grammatics


def classify_image(grammatics, grammatic):
    max_similarity = 0
    character = ''
    for key in grammatics.keys():
        sum = 0
        for value in grammatics[key]:
            sum += difflib.SequenceMatcher(None, grammatic, value).ratio()
        sum /= len(grammatics[key])
        if sum > max_similarity:
            max_similarity = sum
            character = key
    return character


def classify_images(grammatics, test_path):
    for character in os.listdir(test_path):
        if not character.startswith('.'):
            total, correct = 0, 0
            character_path = path.join(test_path, character)
            for file in os.listdir(character_path):
                if not file.startswith('.'):
                    file_path = path.join(character_path, file)
                    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    map = process_image(image)
                    predict = classify_image(grammatics, map)
                    if predict == character:
                        correct += 1
                    total += 1
            print(f'Class: {character}, accuracy = {round(100 * correct / total, 2)}%')


if __name__ == "__main__":
    train_path = '/Users/kirpro/Kirpro/8 триместр/Распознавание образов/preprocessed train'
    grammatics = get_grammatics(train_path)
    save_model(grammatics)

    grammatics = load_model()
    test_path = '/Users/kirpro/Kirpro/8 триместр/Распознавание образов/preprocessed test'
    classify_images(grammatics, test_path)
