import os
import shutil
import random
from PIL import Image

# Number of images for Trial dataset
num_images=50

def create_dir(dist):
    """
    Create a new directory if it does not exist.

    Parameters:
    - dist (str): Path to the directory to be created.
    """
    os.makedirs(dist, exist_ok=True)

def is_image(file_path):
    """
    Check if the file at the given path is an image.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file is an image, False otherwise.
    """
    try:
        with Image.open(file_path):
            return True
    except (IOError, OSError):
        return False

def display_img_list(folder_path, dist_path, limit=10):
    """
    Display and copy a limited number of images from the source folder to the destination folder.

    Parameters:
    - folder_path (str): Path to the source folder containing images.
    - dist_path (str): Path to the destination folder where images will be copied.
    - limit (int): Maximum number of images to display and copy.
    """
    images_dist_folder = os.path.isdir(folder_path)
    if images_dist_folder:
        img_path = os.listdir(folder_path)
        # Sort the list of file names alphabetically
        sorted_img_paths = sorted(img_path)
        for i, img in enumerate(sorted_img_paths):
            if i <= limit:
                final_path = os.path.join(folder_path, img)
                final_dist_path = os.path.join(dist_path, img)
                create_dir(dist_path)
                # Copy the image to the destination folder
                shutil.copy2(final_path, final_dist_path)
                print(f"Copied: {img}")

def copy_images(source_folder, destination_folder, num_images):
    """
    Copy a specified number of images from each class in the source folder to the destination folder.

    Parameters:
    - source_folder (str): Path to the source folder containing class subfolders.
    - destination_folder (str): Path to the destination folder where images will be copied.
    - num_images (int): Number of images to copy from each class.
    """
    for class_folder in os.listdir(source_folder):
        if class_folder != ".DS_Store":
            class_path = os.path.join(source_folder, class_folder)
            # Create destination subfolder
            dist_folder = os.path.join(destination_folder, class_folder)
            if class_folder == 'testX':
                display_img_list(class_path, dist_folder, limit=num_images)
            else:
                for sf in ['bcc', 'mel','scc']:
                    n_class_path = os.path.join(class_path, sf)
                    n_class_path_dist = os.path.join(dist_folder, sf)
                    display_img_list(n_class_path, n_class_path_dist, limit=num_images)

# Set your dataset paths
train_source = '/Users/taiaburrahman/Desktop/Udg/CADx/Challenge/tf/dataset/multiCLASS'
new_train_destination = '/Users/taiaburrahman/Desktop/Udg/CADx/Challenge/tf/trial_dataset/multiCLASS'

# Create new folders for the selected images
os.makedirs(new_train_destination, exist_ok=True)

# Copy 10 images from each class in the training set to the new destination
copy_images(train_source, new_train_destination, num_images)
