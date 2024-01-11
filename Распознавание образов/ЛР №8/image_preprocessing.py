import os
from os import path
import shutil
import cv2 as cv
import numpy as np


def desaturate_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (5,5), 0)
    image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)[1]
    if np.mean(image) < 100:
        image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV)[1]
    return image


def get_empty_rows_number(image):
    value = 255 * image.shape[0]
    empty_rows_number = 0
    for row in image:
        if np.sum(row) != value:
            break
        empty_rows_number += 1
    return empty_rows_number


def crop_image(image):
    top = get_empty_rows_number(image)
    rotated_image = np.rot90(image)
    right = get_empty_rows_number(rotated_image)
    rotated_image = np.rot90(rotated_image)
    bot = get_empty_rows_number(rotated_image)
    rotated_image = np.rot90(rotated_image)
    left = get_empty_rows_number(rotated_image)
    return image[top:(image.shape[0]-bot), left:(image.shape[1]-right)]


def add_border(image, n):
    height, width = image.shape
    if (height, width) != (n, n):
        if height > width:
            border_width = (height - width) // 2
            image = cv.copyMakeBorder(image, 0, 0, border_width, border_width, cv.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            border_height = (width - height) // 2
            image = cv.copyMakeBorder(image, border_height, border_height, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
    return image


def resize_image(image, n):
    image = crop_image(image)
    image = add_border(image, n)
    return cv.resize(image, (n, n), interpolation=cv.INTER_AREA)


def preprocess_images(input_path, output_path, n):
    if path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    extensions = ['.jpg', '.bmp', '.png', '.webp', '.xbm', '.gif']
    characters = dict()

    for guid in os.listdir(input_path):
        if not guid.startswith('.'):
            guid_path = path.join(input_path, guid)
            for character in os.listdir(guid_path):
                if not character.startswith('.'):
                    if character not in characters:
                        os.makedirs(path.join(output_path, character))
                        characters[character] = 0
                    character_path = path.join(guid_path, character)
                    for file in os.listdir(character_path):
                        extension = path.splitext(file)[1]
                        if not file.startswith('.') and extension in extensions:
                            file_path = path.join(character_path, file)
                            image = cv.VideoCapture(file_path).read()[1] if extension == '.gif' else cv.imread(file_path)
                            image = desaturate_image(image)
                            image = resize_image(image, n)
                            cv.imwrite(path.join(output_path, f'{character}/{characters[character]}.png'), image)
                            characters[character] += 1


if __name__ == '__main__':
    input_path = '/Users/kirpro/Desktop/Pattern recognition/train'
    output_path = '/Users/kirpro/Desktop/Pattern recognition/preprocessed train'
    preprocess_images(128)