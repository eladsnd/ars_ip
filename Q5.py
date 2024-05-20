import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
    return image


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def display_results(original, resized, rotated, scale_percent, angle):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    resized_height, resized_width = resized.shape[:2]
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title(f'Resized Image\nScale: {scale_percent}%\nSize: {resized_width}x{resized_height}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    rotated_height, rotated_width = rotated.shape[:2]
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.title(f'Rotated Image\nAngle: {angle}°\nSize: {rotated_width}x{rotated_height}')
    plt.axis('off')

    plt.show()


def scale_and_rotate_workflow(image_path, scale_percent, angle):
    # טעינת התמונה
    image = load_image(image_path)
    if image is None:
        return

    # ביצוע שינויי קנה מידה
    resized_image = resize_image(image, scale_percent)

    # ביצוע סיבוב תמונה
    rotated_image = rotate_image(image, angle)

    # הצגת התוצאות
    display_results(image, resized_image, rotated_image, scale_percent, angle)


if __name__ == '__main__':
    image_path = 'example_maze.jpg'
    scale_percent = 350
    angle = 45
    scale_and_rotate_workflow(image_path, scale_percent, angle)
