###################扣取mask模板####################

import cv2
import numpy as np
import os
import random
from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# Define your folder paths
folder_A = 'E:/masklablechange/txtlabels/tongdaoyinhuan-gy-2'  # Folder containing masks (as .txt files)

# Create output directories for results in BB and CC
output_mask = 'E:/masklablechange/txtlabels/masks'

# Specify the class label to mask (e.g., '1' for class 1)
target_labels = ['0', '1', '2','3','4','5','6','7','8','9','10','11']  # Specify all possible target class labels

# Ensure directories exist
os.makedirs(output_mask, exist_ok=True)

count=0
# Traverse mask files in folder A
for mask_filename in os.listdir(folder_A):
    if mask_filename.endswith('.txt'):
        count=count+1
        print("xxxxxxxxxxxxxxxxxxxxx",count,mask_filename)
        
        mask_path = os.path.join(folder_A, mask_filename)
        image_path_A = os.path.join(folder_A, mask_path.replace(".txt", ".jpg")) 
        image_A = cv2.imread(image_path_A)

        # Create binary mask and filled overlay images
        binary_mask = np.zeros((image_A.shape[0], image_A.shape[1]), dtype=np.uint8)
        filled_image_A = np.zeros_like(image_A)

        # Read mask coordinates
        with open(mask_path, 'r') as file:
            flag = 0
            for line in file:
                parts = line.strip().split()
                if len(parts) > 1 and parts[0] in target_labels:#parts[0] == target_class_label:
                    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    pixel_coords = np.zeros_like(coords)
                    pixel_coords[:, 0] = coords[:, 0] * image_A.shape[1]
                    pixel_coords[:, 1] = coords[:, 1] * image_A.shape[0]
                    coords = np.array(pixel_coords, dtype=np.int32)
                    cv2.fillPoly(binary_mask, [coords], 255)

                    # Copy overlay area from B and C based on the mask
                    filled_image_A[binary_mask == 255] = image_A[binary_mask == 255]
            rgb_mask_filename = os.path.join(output_mask, mask_filename.replace(".txt", ".png"))
            cv2.imwrite(rgb_mask_filename, filled_image_A)  # 保存更新后的图像