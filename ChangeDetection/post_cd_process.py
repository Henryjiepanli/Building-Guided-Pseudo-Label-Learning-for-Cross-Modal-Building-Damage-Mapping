import numpy as np
import os
from scipy import ndimage
import cv2
from tqdm import tqdm
import argparse

def get_connected_regions(labels):
    labeled, num_labels = ndimage.label(labels)
    return labeled, num_labels

def merge_labels_with_intersection(pred_labels, building_labels, threshold=0.5):
    damaged_area = np.isin(pred_labels, [2, 3]).astype(np.uint8)

    building_labeled, num_buildings = get_connected_regions(building_labels)

    for building_id in range(1, num_buildings + 1):
        building_region = (building_labeled == building_id).astype(np.uint8)

        intersection = np.logical_and(damaged_area, building_region)

        if np.sum(intersection) > threshold * np.sum(building_region):
            if np.sum(pred_labels[intersection == 1] == 3) > np.sum(pred_labels[intersection == 1] == 2):

                pred_labels[building_region == 1] = 3
            else:
                pred_labels[building_region == 1] = 2

    return pred_labels

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
        0: [255, 255, 255],   # Background (#FFFFFF)
        1: [108, 178, 125],    # Intact (#6CB27D)
        2: [219, 190, 144],    # Damaged (#DBBE90)
        3: [163, 78, 73],      # Destroyed (#A34E49)
    }

    # Create an empty color map
    colormap = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    # Assign RGB colors to the colormap based on the prediction classes
    for class_index, color in class_to_color.items():
        colormap[pred == class_index] = color

    return colormap

def remove_small_regions(labels, min_size=100):
    labeled, num_labels = get_connected_regions(labels > 0) 
    for region_id in range(1, num_labels + 1):
        if np.sum(labeled == region_id) < min_size:
            labels[labeled == region_id] = 0 
    return labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default=None, help='')
    parser.add_argument('--building_path', type=str, default=None, help='')
    parser.add_argument('--save_path', type=str, default=None, help='')
    parser.add_argument('--vis_save_path', type=str, default=None, help='')
    opt = parser.parse_args()

    pred_path = opt.pred_path
    building_path = opt.building_path
    save_path = opt.save_path
    vis_save_path = opt.vis_save_path

    save_names=['testarea-disaster_00000011_building_damage.png','testarea-disaster_00000085_building_damage.png', 'testarea-disaster_00000087_building_damage.png',\
            'testarea-disaster_00000133_building_damage.png','testarea-disaster_00000154_building_damage.png']

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    for img_name in tqdm(os.listdir(pred_path)):
        pred_img_path = os.path.join(pred_path, img_name)
        building_img_path = os.path.join(building_path, img_name)
        pred_labels = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
        building_labels = cv2.imread(building_img_path, cv2.IMREAD_GRAYSCALE)
        if img_name in save_names:
            merged_labels = building_labels.copy()
            merged_labels[pred_labels==2] = 2
            merged_labels[pred_labels==3] = 3
        else:
            building_labels = cv2.imread(building_img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            merged_labels = merge_labels_with_intersection(pred_labels, building_labels, threshold=0.2)
        
        remove_labels = remove_small_regions(merged_labels, min_size=50)

        save_remove_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_remove_path, remove_labels)
        remove_vis = landcover_to_colormap(remove_labels)
        remove_rgb = cv2.cvtColor(remove_vis, cv2.COLOR_BGR2RGB)
        vis_remove_path = os.path.join(vis_save_path, img_name)
        cv2.imwrite(vis_remove_path, remove_rgb)
        
