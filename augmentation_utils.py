import cv2
import numpy as np
import random
from shapely.geometry import Polygon

def random_hsv1(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[..., 0] += random.randint(-10, 10)  # Randomize hue
    hsv_image[..., 1] *= random.uniform(0.8, 1.2)  # Randomize saturation
    hsv_image[..., 2] *= random.uniform(0.8, 1.2)  # Randomize value
    hsv_image[hsv_image > 255] = 255  # Cap values
    hsv_image[hsv_image < 0] = 0
    return cv2.cvtColor(np.array(hsv_image, dtype=np.uint8), cv2.COLOR_HSV2BGR)

def random_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[..., 0] += random.randint(-30, 30)  # More randomization in hue
    hsv_image[..., 1] *= random.uniform(0.7, 1.3)  # More randomization in saturation
    hsv_image[..., 2] *= random.uniform(0.7, 1.3)  # More randomization in value
    hsv_image[hsv_image > 255] = 255  # Cap values
    hsv_image[hsv_image < 0] = 0
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)


def polygon_iou(points1, points2):
    # Ensure both polygons have at least 4 points to be valid
    poly1 = Polygon(points1)
    poly2 = Polygon(points2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0  # If polygon is invalid, return IOU of 0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union

def augment_polygon(points, image_width, image_height, iou_threshold=0.25, max_shift=1, drop_prob=-1, add_prob=-1):
    while True:
        #print("XXXXXXXXXXXXXXXXXXXXXXXX1")
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
        max_scale_factor = 2 if max_scale_factor > 2 else max_scale_factor
        scale_factor = random.uniform(0.5, max_scale_factor)

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
        #iou = polygon_iou(points, modified_points)

        # Check if modified_points has more than 3 points
        if len(modified_points) > 3:
            #print("XXXXXXXXXXXXXXXXXXXXXXXX3")

            iou = polygon_iou(points, modified_points)

            #print("XXXXXXXXXXXXXXXXXXXXXXXXiou",iou)

            # Ensure modified_points meets the IOU threshold
            if iou < iou_threshold:
                return modified_points  # Return the modified points if conditions are met
        #print("XXXXXXXXXXXXXXXXXXXXXXXX2")

def get_polygon_centroid(points):
    points = np.array(points)
    M = cv2.moments(points)
    if M["m00"] == 0:  # Prevent division by zero
        return points.mean(axis=0)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    return (centroid_x, centroid_y)
