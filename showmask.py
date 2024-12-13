
import cv2
import numpy as np
import glob
import os
import random
from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# Flags and parameters
random_hsv_flag = 1
augment_polygon_flag = 1

alpha_ori = 0.5
alpha_aug = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
text_color = (0, 0, 255) 
thickness = 2

# Paths to images and annotation folders
# image_folder = "E:/masklablechange/test1/"
# annotation_folder = "E:/masklablechange/test1/"
# output_folder = "E:/masklablechange/showout/"
image_folder = "E:/masklablechange/showout/"
annotation_folder = "E:/masklablechange/showout/"
output_folder = "E:/masklablechange/ok/2/"
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Specified labels to be processed
specified_labels = [1,2,3,4]

# Define a dictionary to map class ids to colors with at least 20 colors
color_map = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (255, 0, 255),
    6: (128, 0, 0),
    7: (0, 128, 0),
    8: (0, 0, 128),
    9: (128, 128, 0),
    10: (0, 128, 128),
    11: (128, 0, 128),
    12: (192, 192, 192),
    13: (128, 128, 128),
}
# Loop through all images in the folder
for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
        # Extract image filename without extension
        image_name = os.path.basename(image_path).split(".")[0]
        annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
        
        # Load the image
        image = cv2.imread(image_path)
        aug_image0=image.copy()

        # Save the original image
        output_image_path = os.path.join(output_folder, f"{image_name}_original.jpg")
        cv2.imwrite(output_image_path, image)
        
        # Check if corresponding annotation file exists
        if not os.path.exists(annotation_path):
            print(f"Annotation file for {image_name} not found.")
            continue

        # Read the annotation file
        with open(annotation_path, "r") as file:
            lines = file.readlines()

        polygons = []
        labels = []

        # Loop through each line in the annotation file
        for line in lines:
            # Extract the class label and coordinates
            parts = list(map(float, line.split()))
            class_id = int(parts[0])
            coordinates = parts[1:]
            
            # Gather points for the polygon
            points = []
            for i in range(0, len(coordinates), 2):
                x = int(coordinates[i] * image.shape[1])
                y = int(coordinates[i + 1] * image.shape[0])
                points.append((x, y))

            if class_id in specified_labels:
                polygons.append(np.array([points], dtype=np.int32))
                labels.append(class_id)

        # Prepare the output file for all masks
        mask_txt_path = os.path.join(output_folder, f"{image_name}_original.txt")
        with open(mask_txt_path, "w") as mask_file:
            #Draw specified polygons and masks
            for polygon, label in zip(polygons, labels):
                color = color_map[label]
                alpha = 0.5
                overlay = image.copy()
                cv2.fillPoly(overlay, polygon, color)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                # Calculate the centroid of the polygon
                centroid = get_polygon_centroid(polygon[0])
                
                # Draw the class label at the centroid
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale =2
                thickness = 2
                text_color = (0, 0, 255)  # White color for text
                cv2.putText(image, str(label), centroid, font, font_scale, text_color, thickness, cv2.LINE_AA)

                # Save the original mask in txt format
                mask_coords = []
                mask_file.write(f"{label} ")
                for point in polygon[0]:  # polygon[0] contains the list of points
                    mask_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
                    #mask_file.write(f"{label} {' '.join(mask_coords)}\n")
                mask_file.write(f"{' '.join(mask_coords)}\n")
                print(f"Original annotation for class {label} saved at {mask_txt_path}")

        # Save annotated image
        output_annotated_path = os.path.join(output_folder, f"{image_name}_orimask.jpg")
        cv2.imwrite(output_annotated_path, image)
        
        if len(labels)==0:
           os.remove(mask_txt_path)
           os.remove(output_annotated_path)
           os.remove(output_image_path)
