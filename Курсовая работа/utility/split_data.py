import os
from os import path
import shutil


def move_pictures(input_path):
    train_path = '/Users/kirpro/Desktop/train'
    valid_path = '/Users/kirpro/Desktop/valid'
    test_path = '/Users/kirpro/Desktop/test'

    os.makedirs(train_path)
    os.makedirs(valid_path)
    os.makedirs(test_path)

    for class_directory in os.listdir(input_path):
        if not class_directory.startswith('.'):
            os.makedirs(path.join(train_path, class_directory))
            os.makedirs(path.join(valid_path, class_directory))
            os.makedirs(path.join(test_path, class_directory))
            class_path = path.join(input_path, class_directory)

            total_files = len(os.listdir(class_path)) - 1

            for index, file in enumerate(os.listdir(class_path)):
                if not file.startswith('.'):
                    if index < 0.7 * total_files:
                        shutil.copyfile(path.join(class_path, file), path.join(train_path, class_directory, file))
                    elif index < 0.85 * total_files:
                        shutil.copyfile(path.join(class_path, file), path.join(valid_path, class_directory, file))
                    else:
                        shutil.copyfile(path.join(class_path, file), path.join(test_path, class_directory, file))


if __name__ == '__main__':
    move_pictures('/Users/kirpro/Desktop/data/dataset 3')
