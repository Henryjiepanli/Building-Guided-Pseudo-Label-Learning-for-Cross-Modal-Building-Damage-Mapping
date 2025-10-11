import os
import numpy as np
from scipy import ndimage
import cv2
import argparse
from tqdm import tqdm

def landcover_to_colormap(pred):
    """
    Convert prediction array to a color map for visualization.

    Parameters:
    pred (numpy array): 2D array of class predictions.

    Returns:
    numpy array: 3D array representing the color map.
    """
    # Define the colormap as a dictionary of class index to RGB values
    class_to_color = {
        0: [0, 0, 0],   # Background (#000000)
        1: [255, 255, 255],       # Intact (#6CB27D)
    }

    # Create an empty color map
    colormap = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    # Assign RGB colors to the colormap based on the prediction classes
    for class_index, color in class_to_color.items():
        colormap[pred == class_index] = color

    return colormap

def remove_small_objects(pseudo_labels, min_size=100):

    labeled, num_labels = ndimage.label(pseudo_labels)

    label_sizes = ndimage.sum(pseudo_labels, labeled, range(num_labels + 1))
    for label in range(1, num_labels + 1):
        if label_sizes[label] < min_size:
            pseudo_labels[labeled == label] = 0 

    return pseudo_labels

def morphological_processing(pseudo_labels, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(pseudo_labels, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default=None, help='')
    parser.add_argument('--save_path', type=str, default=None, help='')
    parser.add_argument('--vis_save_path', type=str, default=None, help='')
    opt = parser.parse_args()

    pred_path = opt.pred_path
    save_path = opt.save_path
    vis_save_path = opt.vis_save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    for img_name in tqdm(os.listdir(pred_path)):
        img_path = os.path.join(pred_path, img_name)

        pseudo_labels = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pseudo_labels = remove_small_objects(pseudo_labels, min_size=50)
        pseudo_labels = morphological_processing(pseudo_labels, kernel_size=3)

        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, pseudo_labels)

        colormap = landcover_to_colormap(pseudo_labels)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        vis_img_path = os.path.join(vis_save_path, img_name)
        cv2.imwrite(vis_img_path, colormap_rgb)

