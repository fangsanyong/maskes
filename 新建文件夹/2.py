
#########################可以显示特定标签的mask并显示出来
import cv2
import numpy as np
import glob
import os

# Paths to images and annotation folders
image_folder = "E:/masklablechange/test-imgs/"
annotation_folder = "E:/masklablechange/test-imgs/"
output_folder = "E:/masklablechange/showout/"

# 指定要绘制的标签列表
#specified_labels = [0,1,2,3,4,5,6,11]
specified_labels = [4]

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

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
        color = color_map[label]
        alpha = 0.5
        overlay = image.copy()
        cv2.fillPoly(overlay, polygon, color)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Calculate the bounding box of the polygon
        x_min, y_min, w, h = cv2.boundingRect(polygon)
        x_max = x_min + w
        y_max = y_min + h

        # Determine the position to display the label
        label_position = (x_min + int(w/2), y_min + int(h/2))

        # Display the class label
        class_label = f"Class {label}"
        cv2.putText(image, class_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    # Save the annotated image
    output_path = os.path.join(output_folder, f"{image_name}_annotated.jpg")
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved at {output_path}")

