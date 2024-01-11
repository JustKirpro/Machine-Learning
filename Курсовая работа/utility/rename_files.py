import os
import uuid


def rename_files(path, first_walk=False):
    for set_directory in os.listdir(path):
        if not set_directory.startswith('.'):
            set_directory_path = os.path.join(path, set_directory)
            for class_directory in os.listdir(set_directory_path):
                if not class_directory.startswith('.'):
                    class_directory_path = os.path.join(set_directory_path, class_directory)
                    met_ds_store = False
                    for index, file in enumerate(os.listdir(class_directory_path)):
                        if file.startswith('.'):
                            met_ds_store = True
                            continue
                        old_file_name = os.path.join(class_directory_path, file)
                        index = index - 1 if met_ds_store else index
                        if first_walk:
                            new_file_name = os.path.join(class_directory_path, ''.join([str(uuid.uuid4()), '.jpg']))
                        else:
                            new_file_name = os.path.join(class_directory_path, ''.join([str(index), '.jpg']))
                        os.rename(old_file_name, new_file_name)


if __name__ == '__main__':
    input_path = '/Users/kirpro/Desktop/datasets/dataset 3'
    rename_files(input_path, True)
    rename_files(input_path)
