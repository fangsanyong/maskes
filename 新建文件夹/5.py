# import cv2
# import numpy as np
# import glob
# import os
# import random

# # Paths to images and annotation folders
# image_folder = "E:/masklablechange/test-imgs/"
# annotation_folder = "E:/masklablechange/test-imgs/"

# output_folder = "E:/masklablechange/showout/"
# augmentation_folder = "E:/masklablechange/augmentation/"

# # Ensure the output and augmentation folders exist
# os.makedirs(output_folder, exist_ok=True)
# os.makedirs(augmentation_folder, exist_ok=True)

# # Specified labels to be processed
# specified_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# # Define a dictionary to map class ids to colors with at least 20 colors
# color_map = {
#     0: (255, 0, 0),
#     1: (0, 255, 0),
#     2: (0, 0, 255),
#     3: (255, 255, 0),
#     4: (0, 255, 255),
#     5: (255, 0, 255),
#     6: (128, 0, 0),
#     7: (0, 128, 0),
#     8: (0, 0, 128),
#     9: (128, 128, 0),
#     10: (0, 128, 128),
#     11: (128, 0, 128),
#     12: (192, 192, 192),
#     13: (128, 128, 128),
# }

# # Augmentation functions
# def random_shift(points, max_shift=0.1):
#     shift_x = random.uniform(-max_shift, max_shift)
#     shift_y = random.uniform(-max_shift, max_shift)
#     shifted_points = [(int(x + shift_x), int(y + shift_y)) for (x, y) in points]
#     return shifted_points

# def random_scale(points, scale_range=(0.8, 1.2)):
#     scale_factor = random.uniform(*scale_range)
#     center_x = sum([p[0] for p in points]) / len(points)
#     center_y = sum([p[1] for p in points]) / len(points)
#     scaled_points = [(int(center_x + (x - center_x) * scale_factor), int(center_y + (y - center_y) * scale_factor)) for (x, y) in points]
#     return scaled_points

# def random_hsv(image):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hsv_image = np.array(hsv_image, dtype=np.float64)
#     hsv_image[..., 0] += random.randint(-10, 10)  # Randomize hue
#     hsv_image[..., 1] *= random.uniform(0.8, 1.2)  # Randomize saturation
#     hsv_image[..., 2] *= random.uniform(0.8, 1.2)  # Randomize value
#     hsv_image[hsv_image > 255] = 255  # Cap values
#     hsv_image[hsv_image < 0] = 0
#     return cv2.cvtColor(np.array(hsv_image, dtype=np.uint8), cv2.COLOR_HSV2BGR)

# # Calculate centroid of a polygon
# def get_polygon_centroid(points):
#     points = np.array(points)
#     M = cv2.moments(points)
#     if M["m00"] == 0:  # Prevent division by zero
#         return points.mean(axis=0)
#     centroid_x = int(M["m10"] / M["m00"])
#     centroid_y = int(M["m01"] / M["m00"])
#     return (centroid_x, centroid_y)

# # Loop through all images in the folder
# for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
#     # Extract image filename without extension
#     image_name = os.path.basename(image_path).split(".")[0]
#     annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
    
#     # Load the image
#     image = cv2.imread(image_path)

#     # Save the original image
#     output_image_path = os.path.join(output_folder, f"{image_name}_original.jpg")
#     cv2.imwrite(output_image_path, image)
    
#     # Check if corresponding annotation file exists
#     if not os.path.exists(annotation_path):
#         print(f"Annotation file for {image_name} not found.")
#         continue

#     # Read the annotation file
#     with open(annotation_path, "r") as file:
#         lines = file.readlines()

#     polygons = []
#     labels = []

#     # Loop through each line in the annotation file
#     for line in lines:
#         # Extract the class label and coordinates
#         parts = list(map(float, line.split()))
#         class_id = int(parts[0])
#         coordinates = parts[1:]
        
#         # Gather points for the polygon
#         points = []
#         for i in range(0, len(coordinates), 2):
#             x = int(coordinates[i] * image.shape[1])
#             y = int(coordinates[i + 1] * image.shape[0])
#             points.append((x, y))

#         if class_id in specified_labels:
#             polygons.append(np.array([points], dtype=np.int32))
#             labels.append(class_id)

#     # Prepare the output file for all masks
#     mask_txt_path = os.path.join(output_folder, f"{image_name}_all_masks.txt")
#     with open(mask_txt_path, "w") as mask_file:
#         #Draw specified polygons and masks
#         for polygon, label in zip(polygons, labels):
#             color = color_map[label]
#             alpha = 0.5
#             overlay = image.copy()
#             cv2.fillPoly(overlay, polygon, color)
#             image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

#             # Calculate the centroid of the polygon
#             centroid = get_polygon_centroid(polygon[0])
            
#             # Draw the class label at the centroid
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale =2
#             thickness = 2
#             text_color = (0, 0, 255)  # White color for text
#             cv2.putText(image, str(label), centroid, font, font_scale, text_color, thickness, cv2.LINE_AA)

#             # Save the original mask in txt format
#             mask_coords = []
#             mask_file.write(f"{label} ")
#             for point in polygon[0]:  # polygon[0] contains the list of points
#                 mask_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
#                 #mask_file.write(f"{label} {' '.join(mask_coords)}\n")
#             mask_file.write(f"{' '.join(mask_coords)}\n")
#             print(f"Original annotation for class {label} saved at {mask_txt_path}")


#             # Augmentations for specified labels
#             print(f"Performing augmentations for class {label}")
            
#             for i in range(5):  # Generate 5 augmented images
#                 # Apply random shift, scale, and HSV changes
#                 aug_image = random_hsv(image.copy())
#                 aug_polygon = random_shift(polygon[0].tolist())
#                 aug_polygon = random_scale(aug_polygon)

#                 aug_polygon_np = np.array([aug_polygon], dtype=np.int32)
                
#                 # Overlay the augmented polygon
#                 cv2.fillPoly(aug_image, aug_polygon_np, color)

#                 # Draw the label at the centroid of the augmented polygon
#                 aug_centroid = get_polygon_centroid(aug_polygon)
#                 cv2.putText(aug_image, str(label), aug_centroid, font, font_scale, text_color, thickness, cv2.LINE_AA)
                
#                 # Save the augmented image and its mask as txt
#                 aug_image_path = os.path.join(augmentation_folder, f"{image_name}_augmented_{i}.jpg")
#                 aug_txt_path = os.path.join(augmentation_folder, f"{image_name}_augmented_{i}.txt")
                
#                 cv2.imwrite(aug_image_path, aug_image)
                
#                 # Save new mask in txt format, including original mask coordinates as a comment
#                 with open(aug_txt_path, "w") as aug_file:            
#                 # Save original coordinates as a comment
#                     original_coords = " ".join([f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}" for point in polygon[0]])
#                     aug_file.write(f"{label} {original_coords}\n")

#                     # Save augmented coordinates
#                     aug_coords = []
#                     for point in aug_polygon:
#                         aug_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
#                     aug_file.write(f"{label} {' '.join(aug_coords)}\n")

#                 print(f"Augmented image and mask saved: {aug_image_path}, {aug_txt_path}")
        
#     # Save annotated image
#     output_annotated_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
#     cv2.imwrite(output_annotated_path, image)


# print("Processing completed.")


import cv2
import numpy as np
import glob
import os
import random
from shapely.geometry import Polygon


origin=0
augment=0
augment_and_origin=1

random_shift_flag=0
random_scale_flag=0
random_hsv_flag=0
modify_polygon_points_flag=0
augment_polygon_flag=1

# Paths to images and annotation folders
image_folder = "E:/masklablechange/test1/"
annotation_folder = "E:/masklablechange/test1/"

output_folder = "E:/masklablechange/showout/"
augmentation_folder = "E:/masklablechange/showout/"

# Ensure the output and augmentation folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(augmentation_folder, exist_ok=True)

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
def augment_polygon(points, image_width, image_height, iou_threshold=0.1,max_shift=1, drop_prob=-1, add_prob=-1):
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
        print(iou)

        if iou < iou_threshold:
            return modified_points  # If IOU is below the threshold, return the original points





# Augmentation functions
def random_shift(points, image_width, image_height, max_shift=1):
    # 计算多边形的边界框
    min_x = min([p[0] for p in points])
    max_x = max([p[0] for p in points])
    min_y = min([p[1] for p in points])
    max_y = max([p[1] for p in points])

    # 计算偏移范围，确保点不会超出图像边界
    max_shift_x = min(image_width - max_x, min_x) * max_shift  # 控制横向偏移范围
    max_shift_y = min(image_height - max_y, min_y) * max_shift  # 控制纵向偏移范围

    # 随机生成偏移量
    shift_x = random.uniform(-max_shift_x, max_shift_x)
    shift_y = random.uniform(-max_shift_y, max_shift_y)

    # 对所有点进行偏移，确保在图像范围内
    shifted_points = [(int(x + shift_x), int(y + shift_y)) for (x, y) in points]
    
    return shifted_points

def random_scale(points,img_width, img_height,scale_range=(0.8, 1.2)):
    scale_factor = random.uniform(*scale_range)
    center_x = sum([p[0] for p in points]) / len(points)
    center_y = sum([p[1] for p in points]) / len(points)
    scaled_points = [(int(center_x + (x - center_x) * scale_factor), int(center_y + (y - center_y) * scale_factor)) for (x, y) in points]

    # 边界保护
    protected_points = [(max(0, min(x, img_width - 1)), max(0, min(y, img_height - 1))) for (x, y) in scaled_points]
    return protected_points


def modify_polygon_points(points, image_width, image_height, drop_prob=0.2, add_prob=0.2):
    """
    修改多边形点集，随机丢弃或增加点。
    :param points: 多边形的顶点集合
    :param image_width: 图像的宽度
    :param image_height: 图像的高度
    :param drop_prob: 丢弃点的概率
    :param add_prob: 增加点的概率
    :return: 修改后的顶点集合
    """
    modified_points = points.copy()

    # 随机丢弃一些点
    if len(modified_points) > 3:  # 保证多边形至少有三个点
        modified_points = [p for p in modified_points if random.random() > drop_prob]

    # 随机在多边形边上增加一些点
    new_points = []
    for i in range(len(modified_points)):
        p1 = modified_points[i]
        p2 = modified_points[(i + 1) % len(modified_points)]  # 下一个点，形成一条边
        if random.random() < add_prob:
            # 在边p1-p2之间生成一个新点
            new_x = int((p1[0] + p2[0]) / 2 + random.uniform(-0.05, 0.05) * image_width)
            new_y = int((p1[1] + p2[1]) / 2 + random.uniform(-0.05, 0.05) * image_height)

            # 确保新点在图像范围内
            new_x = max(0, min(new_x, image_width - 1))
            new_y = max(0, min(new_y, image_height - 1))

            new_points.append((new_x, new_y))

    # 将新点加入到多边形点集中
    modified_points += new_points

    return modified_points


# Calculate centroid of a polygon
def get_polygon_centroid(points):
    points = np.array(points)
    M = cv2.moments(points)
    if M["m00"] == 0:  # Prevent division by zero
        return points.mean(axis=0)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    return (centroid_x, centroid_y)


###########################################################################################################################################

if origin:
   
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

#################################################################################################################################

if augment:
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

    ##############################augment#####################################
        aug_image_path = os.path.join(augmentation_folder, f"{image_name}_augmented.jpg")
        aug_txt_path = os.path.join(augmentation_folder, f"{image_name}_augmented.txt")
        aug_image=image.copy()
        overlay_aug = image.copy()
        alpha_aug = 0.2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale =2
        text_color = (0, 0, 255) 
        thickness = 2
        with open(aug_txt_path, "w") as mask_aug_file:
            #Draw specified polygons and masks
            for polygon, label in zip(polygons, labels):
               ###############################################################AUGMENT#################################################
                # Apply random shift, scale, and HSV changes
                aug_polygon=polygon[0].tolist()
                if random_hsv_flag:
                     aug_image = random_hsv(aug_image)
                if random_shift_flag:
                     aug_polygon = random_shift(aug_polygon, image.shape[1], image.shape[0])
                if random_scale_flag:
                    aug_polygon = random_scale(aug_polygon,image.shape[1], image.shape[0])
                if modify_polygon_points_flag:
                   aug_polygon=modify_polygon_points(aug_polygon,image.shape[1], image.shape[0])
                if augment_polygon_flag:
                    aug_polygon=augment_polygon(aug_polygon,image.shape[1], image.shape[0])

                aug_polygon_np = np.array([aug_polygon], dtype=np.int32)       
                # # Overlay the augmented polygon
                color = color_map[label]
                cv2.fillPoly(overlay_aug, aug_polygon_np, color)
                aug_image = cv2.addWeighted(overlay_aug, alpha_aug, aug_image, 1 - alpha_aug, 0)

                # Draw the label at the centroid of the augmented polygon
                aug_centroid = get_polygon_centroid(aug_polygon)
                aug_centroid = (int(aug_centroid[0]), int(aug_centroid[1]))
                cv2.putText(aug_image, str(label), aug_centroid, font, font_scale, text_color, thickness, cv2.LINE_AA)              
                #####################################保存扰动mask####################################
                # Save augmented coordinates
                aug_coords = []
                for point in aug_polygon:
                    aug_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
                mask_aug_file.write(f"{label} {' '.join(aug_coords)}\n")
                print(f"Augmented image and mask saved: {aug_image_path}, {aug_txt_path}")
        
        mask_aug_file.close()
        cv2.imwrite(aug_image_path, aug_image)
            
########################################################################################################################################
if augment_and_origin:
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

    ##############################augment#####################################
        aug_image_path = os.path.join(augmentation_folder, f"{image_name}_ori_and_augmented.jpg")
        aug_txt_path = os.path.join(augmentation_folder, f"{image_name}_ori_and_augmented.txt")
        aug_image=image.copy()
        overlay_ori = image.copy()
        overlay_aug = image.copy()
        alpha_ori = 0.3
        alpha_aug = 0.2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale =2
        text_color = (0, 0, 255) 
        thickness = 2
        with open(aug_txt_path, "w") as mask_aug_file:
            #Draw specified polygons and masks

            for polygon, label in zip(polygons, labels):
                color = color_map[label]
                cv2.fillPoly(overlay_ori, polygon, color)
                aug_image = cv2.addWeighted(overlay_ori, alpha_ori, aug_image, 1 - alpha_ori, 0)
                # Calculate the centroid of the polygon
                centroid = get_polygon_centroid(polygon[0])
                # Draw the class label at the centroid
                cv2.putText(aug_image, str(label), centroid, font, font_scale, text_color, thickness, cv2.LINE_AA)

               ###############################################################AUGMENT#################################################
                # Apply random shift, scale, and HSV changes
                # Apply random shift, scale, and HSV changes
                aug_polygon=polygon[0].tolist()
                if random_hsv_flag:
                     aug_image = random_hsv(aug_image)
                if random_shift_flag:
                     aug_polygon = random_shift(aug_polygon, image.shape[1], image.shape[0])
                if random_scale_flag:
                    aug_polygon = random_scale(aug_polygon,image.shape[1], image.shape[0])
                if modify_polygon_points_flag:
                   aug_polygon=modify_polygon_points(aug_polygon,image.shape[1], image.shape[0])
                if augment_polygon_flag:
                    aug_polygon=augment_polygon(aug_polygon,image.shape[1], image.shape[0])

                aug_polygon_np = np.array([aug_polygon], dtype=np.int32)
                
                # # Overlay the augmented polygon
                color = color_map[label]
                #alpha = 0.5
                #overlay = aug_image.copy()
                cv2.fillPoly(overlay_aug, aug_polygon_np, color)
                aug_image = cv2.addWeighted(overlay_aug, alpha_aug, aug_image, 1 - alpha_aug, 0)

                # Draw the label at the centroid of the augmented polygon
                aug_centroid = get_polygon_centroid(aug_polygon)
                aug_centroid = (int(aug_centroid[0]), int(aug_centroid[1]))
                #print("xxxxxxxxxxxxxxxxxxx1"+str(aug_centroid))
                cv2.putText(aug_image, str(label), aug_centroid, font, font_scale, text_color, thickness, cv2.LINE_AA)
                

                #####################################保存原始mask+扰动mask####################################
                # Save the original mask in txt format
                mask_coords = []
                mask_aug_file.write(f"{label} ")
                for point in polygon[0]:  # polygon[0] contains the list of points
                    mask_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
                mask_aug_file.write(f"{' '.join(mask_coords)}\n")

                # Save augmented coordinates
                aug_coords = []
                for point in aug_polygon:
                    aug_coords.append(f"{point[0] / image.shape[1]} {point[1] / image.shape[0]}")
                mask_aug_file.write(f"{label} {' '.join(aug_coords)}\n")
                print(f"Augmented image and mask saved: {aug_image_path}, {aug_txt_path}")
        
        mask_aug_file.close()
        cv2.imwrite(aug_image_path, aug_image)
            
