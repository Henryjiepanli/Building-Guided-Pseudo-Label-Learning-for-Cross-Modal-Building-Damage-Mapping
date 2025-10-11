import numpy as np
import os
import cv2

def convert_label_to_npy(label_path, save_path, num_classes=4):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        print(f"Error reading image: {label_path}")
        return
    
    h, w = label.shape
    one_hot_label = np.zeros((num_classes, h, w), dtype=np.uint8)

    for i in range(num_classes):
        one_hot_label[i] = (label == i).astype(np.uint8)

    np.save(save_path, one_hot_label)
    print(f"Saved: {save_path}")

label_dir = "/home/henry/DFC2025/dfc25_track2_trainval/Divided_train/hawaii-wildfire/target/"
save_dir = "/home/henry/DFC2025/dfc25_track2_trainval/Divided_train/hawaii-wildfire/prob/"

os.makedirs(save_dir, exist_ok=True)

for img_name in os.listdir(label_dir):
    if img_name.endswith((".png", ".tif", ".jpg")):  
        label_path = os.path.join(label_dir, img_name)
        save_path = os.path.join(save_dir, img_name.replace(".png", ".npy").replace(".tif", ".npy").replace(".jpg", ".npy"))
        convert_label_to_npy(label_path, save_path)
