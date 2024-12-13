
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
image_folder = "E:/masklablechange/test2/"
annotation_folder = "E:/masklablechange/test2/"
output_folder = "E:/masklablechange/showout/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Specified labels to be processed
specified_labels = [1]

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


for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
    # Extract image filename without extension
    image_name = os.path.basename(image_path).split(".")[0]
    annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
    # Load the image
    image = cv2.imread(image_path)

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


   ###############################################################################################
    original_txt_path = os.path.join(output_folder, f"{image_name}_original.txt")
    with open(original_txt_path, "w") as mask_ori_file:
        for polygon, label in zip(polygons, labels):
            # Save original coordinates
            original_coords = []
            for point in polygon[0].tolist():
                original_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
            mask_ori_file.write(f"{label} {' '.join(original_coords)}\n")

    ############################## Augmentation #####################################

    aug_txt_path = os.path.join(output_folder, f"{image_name}_combined.txt")
    combined_image_path = os.path.join(output_folder, f"{image_name}_combined.jpg")  # Define combined_image_path here
    # Initialize a blank composite mask for all labels
    composite_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create an empty mask


    if random_hsv_flag:
        image = random_hsv(image)
    # Initialize combined image to the original image
    combined_image=image.copy()

    with open(aug_txt_path, "w") as mask_aug_file:
        for polygon, label in zip(polygons, labels):
            ###############################################################AUGMENT#################################################
            # Apply random shift, scale, and HSV changes
            aug_polygon = polygon[0].tolist()
            if augment_polygon_flag:
                aug_polygon = augment_polygon(aug_polygon, combined_image.shape[1], combined_image.shape[0])
            
            # Save original coordinates
            original_coords = []
            for point in polygon[0].tolist():
                original_coords.append(f"{point[0] / combined_image.shape[1]} {point[1] / combined_image.shape[0]}")
            mask_aug_file.write(f"{label} {' '.join(original_coords)}\n")

            # Save augmented coordinates
            aug_coords = []
            for point in aug_polygon:
                aug_coords.append(f"{point[0] / combined_image.shape[1]} {point[1] / combined_image.shape[0]}")
            mask_aug_file.write(f"{label} {' '.join(aug_coords)}\n")

            # Create a mask for the original polygon
            original_polygon_np = np.array([polygon[0]], dtype=np.int32)
            mask = np.zeros(combined_image.shape[:2], dtype=np.uint8)  # Create an empty mask
            cv2.fillPoly(mask, original_polygon_np, 255)  # Fill the original polygon in the mask

            # Create a new image for the transformed mask
            transformed_polygon_np = np.array([aug_polygon], dtype=np.int32)
            transformed_mask = np.zeros(combined_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(transformed_mask, transformed_polygon_np, 255)

            # Now we want to warp the original mask content to the new coordinates
            original_mask_content = cv2.bitwise_and(image, image, mask=mask)  # Get original image pixels where the original mask is




            # Create a transformation matrix to map the original polygon to the augmented polygon
            src_pts = np.array(original_polygon_np[0], dtype=np.float32)
            dst_pts = np.array(transformed_polygon_np[0], dtype=np.float32)

            # Calculate the transformation matrix
            transformation_matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])

            # Apply the warp to the original mask content
            warped_mask_content = cv2.warpAffine(original_mask_content, transformation_matrix, (combined_image.shape[1], combined_image.shape[0]))

            # Combine the transformed mask with the warped content
            transformed_mask_image = np.zeros_like(combined_image)
            transformed_mask_image = cv2.bitwise_or(transformed_mask_image, warped_mask_content)

            mask_non_zero = transformed_mask_image > 0  # Get a boolean mask where transformed_mask_image is not zero
            combined_image[mask_non_zero] = transformed_mask_image[mask_non_zero]


                    
        
            # Add the binary mask for the current label to the composite mask
            binary_mask = np.zeros_like(transformed_mask, dtype=np.uint8)
            #binary_mask[mask_non_zero] = 255
            binary_mask[mask_non_zero[:, :, 0]] = 255
            composite_mask = cv2.bitwise_or(composite_mask, binary_mask)
 

    cv2.imwrite(combined_image_path, combined_image)
    # Save the composite mask image for all labels
    composite_mask_path = combined_image_path.replace('.jpg', '_composite_mask.png')  # Adjust the path as necessary
    cv2.imwrite(composite_mask_path, composite_mask)

    if len(labels) == 0:
        os.remove(output_image_path)
        os.remove(aug_txt_path) 
