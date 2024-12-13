import cv2
import numpy as np
import glob
import os
import random
from shapely.geometry import Polygon



random_hsv_flag=1
augment_polygon_flag=1

alpha_ori = 0.5
alpha_aug = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale =2
text_color = (0, 0, 255) 
thickness = 2

# Paths to images and annotation folders
image_folder = "E:/masklablechange/test1/"
annotation_folder = "E:/masklablechange/test1/"
output_folder = "E:/masklablechange/showout/"

# Ensure the output and augmentation folders exist
os.makedirs(output_folder, exist_ok=True)

# Specified labels to be processed
specified_labels = [4]


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

def random_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[..., 0] += random.randint(-10, 10)  # Randomize hue
    hsv_image[..., 1] *= random.uniform(0.8, 1.2)  # Randomize saturation
    hsv_image[..., 2] *= random.uniform(0.8, 1.2)  # Randomize value
    hsv_image[hsv_image > 255] = 255  # Cap values
    hsv_image[hsv_image < 0] = 0
    return cv2.cvtColor(np.array(hsv_image, dtype=np.uint8), cv2.COLOR_HSV2BGR)
import random


def polygon_iou(points1, points2):
    """
    计算两个多边形的IOU
    :param points1: 第一个多边形的顶点
    :param points2: 第二个多边形的顶点
    :return: IOU值
    """
    poly1 = Polygon(points1)
    poly2 = Polygon(points2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0  # 如果多边形无效，返回IOU为0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union == 0:
        return 0  # 避免除以0的情况
    return intersection / union

#def augment_polygon(points, image_width, image_height, max_shift=0.1, drop_prob=0.2, add_prob=0.2):
#def augment_polygon(points, image_width, image_height, iou_threshold=0.1,max_shift=1, drop_prob=-1, add_prob=-1)
def augment_polygon(points, image_width, image_height, iou_threshold=0.1,max_shift=1, drop_prob=0.1, add_prob=0.1):
    """
    对多边形进行缩放、平移并随机增加或删除点，所有操作确保坐标在图像内，不发生截断。
    :param points: 多边形的顶点集合
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :param max_shift: 最大平移量控制
    :param drop_prob: 删除点的概率
    :param add_prob: 增加点的概率
    :return: 增强后的多边形顶点集合
    """
    while True:
        # Step 1: Scaling
        min_x = min([p[0] for p in points])
        max_x = max([p[0] for p in points])
        min_y = min([p[1] for p in points])
        max_y = max([p[1] for p in points])

        # Calculate maximum allowable scaling factor
        max_scale_x = image_width / (max_x - min_x)
        max_scale_y = image_height / (max_y - min_y)
        max_scale_factor = min(max_scale_x, max_scale_y)

        # Random scaling factor between 0 and the calculated max_scale_factor
        max_scale_factor=2 if max_scale_factor>2 else max_scale_factor
        scale_factor=random.uniform(0.5, max_scale_factor)

        # Calculate the center of the polygon
        center_x = sum([p[0] for p in points]) / len(points)
        center_y = sum([p[1] for p in points]) / len(points)

        # Apply scaling to all points
        scaled_points = [(int(center_x + (x - center_x) * scale_factor), 
                        int(center_y + (y - center_y) * scale_factor)) for (x, y) in points]

        # Ensure all points remain within image boundaries
        scaled_points = [(max(0, min(x, image_width - 1)), max(0, min(y, image_height - 1))) for (x, y) in scaled_points]

        # Step 2: Shifting
        min_x = min([p[0] for p in scaled_points])
        max_x = max([p[0] for p in scaled_points])
        min_y = min([p[1] for p in scaled_points])
        max_y = max([p[1] for p in scaled_points])

        max_shift_x = min(image_width - max_x, min_x) * max_shift
        max_shift_y = min(image_height - max_y, min_y) * max_shift

        shift_x = random.uniform(-max_shift_x, max_shift_x)
        shift_y = random.uniform(-max_shift_y, max_shift_y)

        # Apply shifting to all points
        shifted_points = [(int(x + shift_x), int(y + shift_y)) for (x, y) in scaled_points]

        # Step 3: Modify points (drop or add)
        modified_points = shifted_points.copy()

        # Randomly drop points
        if len(modified_points) > 3:  # Ensure at least three points remain
            modified_points = [p for p in modified_points if random.random() > drop_prob]

        # Randomly add points between edges
        new_points = []
        for i in range(len(modified_points)):
            p1 = modified_points[i]
            p2 = modified_points[(i + 1) % len(modified_points)]
            if random.random() < add_prob:
                new_x = int((p1[0] + p2[0]) / 2 + random.uniform(-0.05, 0.05) * image_width)
                new_y = int((p1[1] + p2[1]) / 2 + random.uniform(-0.05, 0.05) * image_height)
                new_x = max(0, min(new_x, image_width - 1))
                new_y = max(0, min(new_y, image_height - 1))
                new_points.append((new_x, new_y))

        # Add new points to the polygon
        modified_points += new_points

        # Step 4: Calculate IOU and decide whether to keep the augmented points
        iou = polygon_iou(points, modified_points)
        #print(iou)

        if iou < iou_threshold:
            return modified_points  # If IOU is below the threshold, return the original points

# Calculate centroid of a polygon
def get_polygon_centroid(points):
    points = np.array(points)
    M = cv2.moments(points)
    if M["m00"] == 0:  # Prevent division by zero
        return points.mean(axis=0)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    return (centroid_x, centroid_y)



            
########################################################################################################################################

# Loop through all images in the folder
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

    ############################## Augmentation #####################################
    aug_image_path = os.path.join(output_folder, f"{image_name}_augmented.jpg")
    aug_txt_path = os.path.join(output_folder, f"{image_name}_augmented.txt")
    hsv_image_path = os.path.join(output_folder, f"{image_name}_hsv.jpg")
    aug_image = image.copy()
    overlay_aug = image.copy()

    if random_hsv_flag:
        aug_image = random_hsv(aug_image)
        image_hsv = aug_image.copy()

    with open(aug_txt_path, "w") as mask_aug_file:
        # Draw specified polygons and masks
        for polygon, label in zip(polygons, labels):
            ###############################################################AUGMENT#################################################
            # Apply random shift, scale, and HSV changes
            aug_polygon = polygon[0].tolist()

            # Save augmented coordinates
            original_coords = []
            for point in aug_polygon:
                original_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
            mask_aug_file.write(f"{label} {' '.join(original_coords)}\n")


            if augment_polygon_flag:
                aug_polygon = augment_polygon(aug_polygon, image.shape[1], image.shape[0])

            # Create a mask for the original polygon
            original_polygon_np = np.array([polygon[0]], dtype=np.int32)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create an empty mask
            cv2.fillPoly(mask, original_polygon_np, 255)  # Fill the original polygon in the mask

            # Create a new image for the transformed mask
            transformed_mask_image = np.zeros_like(image)
            # Get the transformed polygon points
            transformed_polygon_np = np.array([aug_polygon], dtype=np.int32)

            # Fill the transformed polygon
            transformed_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(transformed_mask, transformed_polygon_np, 255)

            # Now we want to warp the original mask content to the new coordinates
            original_mask_content = cv2.bitwise_and(image, image, mask=mask)  # Get original image pixels where the original mask is

            # Create a transformation matrix to map the original polygon to the augmented polygon
            src_pts = np.array(original_polygon_np[0], dtype=np.float32)
            dst_pts = np.array(transformed_polygon_np[0], dtype=np.float32)

            # Calculate the transformation matrix
            transformation_matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])

            # Apply the warp to the original mask content
            warped_mask_content = cv2.warpAffine(original_mask_content, transformation_matrix, (image.shape[1], image.shape[0]))

            # Combine the transformed mask with the warped content
            transformed_mask_image = cv2.bitwise_or(transformed_mask_image, warped_mask_content)


            mask_non_zero = transformed_mask_image > 0  # Get a boolean mask where transformed_mask_image is not zero
            # Create an empty combined image
            combined_image = np.zeros_like(image)
            # Use transformed_mask_image in the regions where it's non-zero
            combined_image[mask_non_zero] = transformed_mask_image[mask_non_zero]
            # Use the original image in the regions where transformed_mask_image is zero
            combined_image[~mask_non_zero] = image[~mask_non_zero]
            # Save the combined image
            combined_image_path = os.path.join(output_folder, f"{image_name}_combined.jpg")
            cv2.imwrite(combined_image_path, combined_image)



            # Save augmented coordinates
            aug_coords = []
            for point in aug_polygon:
                aug_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
            mask_aug_file.write(f"{label} {' '.join(aug_coords)}\n")

            # Overlay the original polygon on the overlay image
            color = color_map[label]
            cv2.fillPoly(overlay_aug, transformed_polygon_np, color)  # Fill transformed polygon on overlay
            aug_image = cv2.addWeighted(overlay_aug, alpha_aug, aug_image, 1 - alpha_aug, 0)

            # Draw the label at the centroid of the augmented polygon
            aug_centroid = get_polygon_centroid(aug_polygon)
            aug_centroid = (int(aug_centroid[0]), int(aug_centroid[1]))
            cv2.putText(aug_image, str(label), aug_centroid, font, font_scale, text_color, thickness, cv2.LINE_AA)              

        print(f"Augmented image and mask saved: {aug_image_path}, {aug_txt_path}")

    mask_aug_file.close()
    cv2.imwrite(aug_image_path, aug_image)
    cv2.imwrite(hsv_image_path, image_hsv)

    if len(labels) == 0:
        os.remove(output_image_path)
        os.remove(aug_image_path)
        os.remove(aug_txt_path)
        os.remove(hsv_image_path)
