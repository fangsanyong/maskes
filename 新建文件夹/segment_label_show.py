import cv2
import numpy as np
import glob
import os

# Paths to images and annotation folders
image_folder = "E:/masklablechange/test-imgs/"
annotation_folder = "E:/masklablechange/test-imgs/"
output_folder = "E:/masklablechange/showout/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

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

    # Loop through each line in the annotation file
    for line in lines:
        # Extract the class label and coordinates
        parts = list(map(float, line.split()))
        class_id = int(parts[0])
        coordinates = parts[1:]
        
        # Gather points for the polygon
        points = []
        for i in range(0, len(coordinates), 2):
            x = int(coordinates[i] * image.shape[1])  # Scale x to image width
            y = int(coordinates[i + 1] * image.shape[0])  # Scale y to image height
            points.append((x, y))


        ####边框多边形显示
        #points = np.array(points, dtype=np.int32)
        #color = (0, 255, 0) if class_id == 1 else (255, 0, 0)  # Colors by class
        #cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        
        ####多边形填充显示
        points = np.array([points], dtype=np.int32)
        color = (0, 255, 0) if class_id == 1 else (255, 0, 0)  # Different color for each class
        alpha = 0.5  # Transparency level (0 = fully transparent, 1 = fully opaque)
        overlay = image.copy()
        cv2.fillPoly(overlay, points, color)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        #cv2.fillPoly(image, points, color)  # **This line fills the polygon**

    # Save the annotated image
    output_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved at {output_path}")








# import cv2
# import numpy as np
# import glob
# import os

# # 设置字体为黑体
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 0.8
# font_thickness = 2

# # Paths to images and annotation folders
# image_folder = "E:/masklablechange/wuxi20231127lzjs/"
# annotation_folder = "E:/masklablechange/wuxi20231127lzjs/"
# output_folder = "E:/masklablechange/showout/"
# label_file = "E:/masklablechange/predefined_classes.txt"

# # Read predefined classes from the label file and create a dictionary
# class_dict = {}
# with open(label_file, "r", encoding="utf-8") as f:
#     line_number = 0
#     for line in f:
#         class_dict[line_number] = line.strip()
#         line_number += 1

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
#     with open(annotation_path, "r", encoding="utf-8") as file:
#         lines = file.readlines()

#     # Loop through each line in the annotation file
#     for line in lines:
#         parts = line.strip().split()
#         class_id = int(parts[0])
#         center_x = float(parts[1])
#         center_y = float(parts[2])
#         width = float(parts[3])
#         height = float(parts[4])

#         # Calculate top-left and bottom-right corners of the rectangle
#         x_min = int((center_x - width / 2) * image.shape[1])
#         y_min = int((center_y - height / 2) * image.shape[0])
#         x_max = int((center_x + width / 2) * image.shape[1])
#         y_max = int((center_y + height / 2) * image.shape[0])

#         # Draw rectangle
#         color = (0, 255, 0) if class_id == 1 else (255, 0, 0)
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

#         # Display class label at the top left corner of the corresponding object
#         if class_id in class_dict:
#             class_label = f"Class: {class_dict[class_id]}"
#             try:
#                 encoded_label = class_label.encode('utf-8').decode('utf-8')
#                 label_position = (x_min + 5, y_min + 20)
#                 cv2.putText(image, encoded_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
#             except UnicodeEncodeError:
#                 print(f"Error encoding label: {class_label}")
#         else:
#             class_label = f"Class: Unknown"
#             try:
#                 encoded_label = class_label.encode('utf-8').decode('utf-8')
#                 label_position = (x_min + 5, y_min + 20)
#                 cv2.putText(image, encoded_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
#             except UnicodeEncodeError:
#                 print(f"Error encoding label: {class_label}")

#     # Save the annotated image
#     output_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
#     cv2.imwrite(output_path, image)
#     print(f"Annotated image saved at {output_path}")