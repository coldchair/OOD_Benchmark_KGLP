# This file defines some utils of directory operations
import os

# Create folder {dir}
def make_dir(dir):
    if os.path.isdir(dir):
        print("Folder %s is already existed." % dir)
    else:
        os.makedirs(dir)
        print("Success to create folder %s." % dir)

# Get direct folders' name under {models_root_path}
def get_models_list(models_root_path):
    name_list = []
    for dir in os.listdir(models_root_path):
        if os.path.isdir(os.path.join(models_root_path, dir)):
            name_list.append(dir)
    return name_list