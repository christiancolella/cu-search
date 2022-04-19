import os

def path_exists(filepath):
    """Determines if a filepath is in an existing directory"""

    path_list = filepath.split("/")                             # Split file path by folder

    for i in range(len(path_list)):                             # Check if each folder in the path exists
        temp_path = ""
        for j in range(i):
            temp_path = temp_path + path_list[j] + "/"
            if not os.path.isdir(temp_path):
                return False

    return True

def make_path(filepath):
    """Creates folders to a filepath"""

    path_list = filepath.split("/")                             # Split file path by folder

    for i in range(len(path_list)):
        temp_path = ""
        for j in range(i):
            temp_path = temp_path + path_list[j] + "/"
            try:
                os.mkdir(temp_path)
            except FileExistsError:
                pass
