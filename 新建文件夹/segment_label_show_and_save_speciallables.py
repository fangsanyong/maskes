# import cv2
# import numpy as np
# import glob
# import os

# # Paths to images and annotation folders
# image_folder = "E:/masklablechange/test-imgs/"
# annotation_folder = "E:/masklablechange/test-imgs/"
# output_folder = "E:/masklablechange/showout/"

# # Ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # Loop through all images in the folder
# for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
#     # Extract image filename without extension
#     image_name = os.path.basename(image_path).split(".")[0]
#     annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
    
#     # Load the image
#     image = cv2.imread(image_path)

#     # Check if corresponding annotation file exists
#     if not os.path.exists(annotation_path):
#         print(f"Annotation file for {image_name} not found.")
#         continue

#     # Read the annotation file
#     with open(annotation_path, "r") as file:
#         lines = file.readlines()

#     # Loop through each line in the annotation file
#     for line in lines:
#         # Extract the class label and coordinates
#         parts = list(map(float, line.split()))
#         class_id = int(parts[0])
#         coordinates = parts[1:]
        
#         # Gather points for the polygon
#         points = []
#         for i in range(0, len(coordinates), 2):
#             x = int(coordinates[i] * image.shape[1])  # Scale x to image width
#             y = int(coordinates[i + 1] * image.shape[0])  # Scale y to image height
#             points.append((x, y))

#         ####边框多边形显示
#         #points = np.array(points, dtype=np.int32)
#         #color = (0, 255, 0) if class_id == 1 else (255, 0, 0)  # Colors by class
#         #cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        
#         ####多边形填充显示
#         points = np.array([points], dtype=np.int32)
#         color = (0, 255, 0) if class_id == 1 else (255, 0, 0)  # Different color for each class
#         alpha = 0.5  # Transparency level (0 = fully transparent, 1 = fully opaque)
#         overlay = image.copy()
#         cv2.fillPoly(overlay, points, color)
#         image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
#         #cv2.fillPoly(image, points, color)  # **This line fills the polygon**

#     # Save the annotated image
#     output_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
#     cv2.imwrite(output_path, image)
#     print(f"Annotated image saved at {output_path}")


# import cv2
# import numpy as np
# import glob
# import os

# # Paths to images and annotation folders
# image_folder = "E:/masklablechange/test-imgs/"
# annotation_folder = "E:/masklablechange/test-imgs/"
# output_folder = "E:/masklablechange/showout/"

# # Ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # Define a dictionary to map class ids to colors with at least 20 colors
# color_map = {
#     1: (255, 0, 0),
#     2: (0, 255, 0),
#     3: (0, 0, 255),
#     4: (255, 255, 0),
#     5: (0, 255, 255),
#     6: (255, 0, 255),
#     7: (128, 0, 0),
#     8: (0, 128, 0),
#     9: (0, 0, 128),
#     10: (128, 128, 0),
#     11: (0, 128, 128),
#     12: (128, 0, 128),
#     13: (192, 192, 192),
#     14: (128, 128, 128),
#     15: (64, 0, 0),
#     16: (0, 64, 0),
#     17: (0, 0, 64),
#     18: (64, 64, 0),
#     19: (0, 64, 64),
#     20: (64, 0, 64)
# }

# # Loop through all images in the folder
# for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
#     # Extract image filename without extension
#     image_name = os.path.basename(image_path).split(".")[0]
#     annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
    
#     # Load the image
#     image = cv2.imread(image_path)

#     # Check if corresponding annotation file exists
#     if not os.path.exists(annotation_path):
#         print(f"Annotation file for {image_name} not found.")
#         continue

#     # Read the annotation file
#     with open(annotation_path, "r") as file:
#         lines = file.readlines()

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

#         # Get the color for the current class id
#         if class_id in color_map:
#             color = color_map[class_id]
#         else:
#             color = (128, 128, 128)

#         # Polygon fill display
#         points = np.array([points], dtype=np.int32)
#         alpha = 0.5
#         overlay = image.copy()
#         cv2.fillPoly(overlay, points, color)
#         image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

#         # Calculate the bounding box of the polygon
#         x_min, y_min, w, h = cv2.boundingRect(points)
#         x_max = x_min + w
#         y_max = y_min + h

#         # Determine the position to display the label
#         label_position = (x_min - 10, y_min - 10)

#         # Display the class label
#         if class_id in color_map:
#             class_label = f"Class {class_id}"
#             cv2.putText(image, class_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Save the annotated image
#     output_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
#     cv2.imwrite(output_path, image)
#     print(f"Annotated image saved at {output_path}")



#########################可以显示特定标签的mask并显示出来
# import cv2
# import numpy as np
# import glob
# import os

# # Paths to images and annotation folders
# image_folder = "E:/masklablechange/test-imgs/"
# annotation_folder = "E:/masklablechange/test-imgs/"
# output_folder = "E:/masklablechange/showout/"

# # 指定要绘制的标签列表
# #specified_labels = [0,1,2,3,4,5,6,11]
# specified_labels = [4]

# # Ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)

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
#     14: (64, 0, 0),
#     15: (0, 64, 0),
#     16: (0, 0, 64),
#     17: (64, 64, 0),
#     18: (0, 64, 64),
#     19: (64, 0, 64)
# }
# # Loop through all images in the folder
# for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
#     # Extract image filename without extension
#     image_name = os.path.basename(image_path).split(".")[0]
#     annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
    
#     # Load the image
#     image = cv2.imread(image_path)

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

#     # Draw specified polygons and masks
#     for polygon, label in zip(polygons, labels):
#         color = color_map[label]
#         alpha = 0.5
#         overlay = image.copy()
#         cv2.fillPoly(overlay, polygon, color)
#         image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

#         # Calculate the bounding box of the polygon
#         x_min, y_min, w, h = cv2.boundingRect(polygon)
#         x_max = x_min + w
#         y_max = y_min + h

#         # Determine the position to display the label
#         label_position = (x_min + int(w/2), y_min + int(h/2))

#         # Display the class label
#         class_label = f"Class {label}"
#         cv2.putText(image, class_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

#     # Save the annotated image
#     output_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
#     cv2.imwrite(output_path, image)
#     print(f"Annotated image saved at {output_path}")


####################################################保存特定的标签，并显示
import cv2
import numpy as np
import glob
import os

# Paths to images and annotation folders
image_folder = "E:/masklablechange/test-imgs/"
annotation_folder = "E:/masklablechange/test-imgs/"
output_folder = "E:/masklablechange/showout/"

# 指定要绘制和保存 mask 的标签列表
specified_labels = [0,4]  # 可以更改为 [0,1,2,3,4,5,6,11] 等

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define a dictionary to map class ids to colors
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
    14: (64, 0, 0),
    15: (0, 64, 0),
    16: (0, 0, 64),
    17: (64, 64, 0),
    18: (0, 64, 64),
    19: (64, 0, 64)
}

# Loop through all images in the folder
for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
    # Extract image filename without extension
    image_name = os.path.basename(image_path).split(".")[0]
    annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
    
    # Load the image
    image = cv2.imread(image_path)

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

    # Draw specified polygons and masks
    for polygon, label in zip(polygons, labels):
        color = color_map.get(label, (255, 255, 255))  # Default color if label not in color_map
        alpha = 0.5
        overlay = image.copy()
        cv2.fillPoly(overlay, polygon, color)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Calculate the bounding box of the polygon
        x_min, y_min, w, h = cv2.boundingRect(polygon)
        x_max = x_min + w
        y_max = y_min + h

        # Determine the position to display the label
        label_position = (x_min + int(w / 2), y_min + int(h / 2))

        # Display the class label
        class_label = f"Class {label}"
        cv2.putText(image, class_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    # Save the annotated image
    output_image_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved at {output_image_path}")

    # Save the original image in the output folder
    original_image_path = os.path.join(output_folder, f"{image_name}_original.jpg")
    cv2.imwrite(original_image_path, cv2.imread(image_path))  # 保存原图
    print(f"Original image saved at {original_image_path}")

    # Save the masks for specified labels in a TXT file
    mask_output_path = os.path.join(output_folder, f"{image_name}_mask.txt")
    with open(mask_output_path, "w") as mask_file:
        for polygon, label in zip(polygons, labels):
            # Normalize coordinates to [0, 1] and save
            normalized_points = []
            for point in polygon[0]:  # Extracting the points from the polygon
                norm_x = point[0] / image.shape[1]
                norm_y = point[1] / image.shape[0]
                normalized_points.append(f"{norm_x:.6f} {norm_y:.6f}")
            mask_file.write(f"{label} {' '.join(normalized_points)}\n")

    print(f"Masks for specified labels saved at {mask_output_path}")
