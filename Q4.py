import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
    return image


def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def color_segmentation_hsv(image_hsv, lower_color, upper_color):
    mask = cv2.inRange(image_hsv, lower_color, upper_color)
    result = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    return result


def display_results(original, segmented):
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB))
    plt.title('Segmented Image (HSV)')
    plt.axis('off')

    plt.show()


def color_segmentation_workflow(image_path, lower_color, upper_color):
    # טעינת התמונה
    image = load_image(image_path)
    if image is None:
        return

    # המרה ל-HSV
    image_hsv = convert_to_hsv(image)

    # סגמנטציה מבוססת צבע במרחב HSV
    segmented_image = color_segmentation_hsv(image_hsv, lower_color, upper_color)

    # הצגת התוצאות
    display_results(image, segmented_image)


if __name__ == '__main__':
    maze = 'example_maze.jpg'

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    color_segmentation_workflow(maze, lower_blue, upper_blue)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    color_segmentation_workflow(maze, lower_green, upper_green)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    color_segmentation_workflow(maze, lower_red, upper_red)


