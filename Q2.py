import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
    return image


def prewitt_edge_detection(image):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)
    edges = cv2.bitwise_or(edges_x, edges_y)
    return edges


def sobel_edge_detection(image):
    edges_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.bitwise_or(np.uint8(np.absolute(edges_x)), np.uint8(np.absolute(edges_y)))
    return edges


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def display_edge_detection_results(original, prewitt_edges, sobel_edges, canny_edges):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(prewitt_edges, cmap='gray')
    plt.title('Prewitt Edge Detection')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')

    plt.show()


def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "salt_and_pepper":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.04
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1]] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1]] = 0
        return noisy


def combined_edge_detection(image):
    sobel_edges = sobel_edge_detection(image)
    canny_edges = canny_edge_detection(image)
    combined_edges = cv2.bitwise_or(sobel_edges, canny_edges)
    return combined_edges


def edge_detection_workflow(image_path):
    # טעינת התמונה
    image = load_image(image_path)
    if image is None:
        return

    # זיהוי קצוות
    prewitt_edges = prewitt_edge_detection(image)
    sobel_edges = sobel_edge_detection(image)
    canny_edges = canny_edge_detection(image)

    # הצגת תוצאות זיהוי קצוות
    display_edge_detection_results(image, prewitt_edges, sobel_edges, canny_edges)

    # הוספת רעש וזיהוי קצוות על תמונות עם רעש
    noisy_gaussian = add_noise(image, "gaussian")
    noisy_sp = add_noise(image, "salt_and_pepper")

    prewitt_edges_noisy_gaussian = prewitt_edge_detection(noisy_gaussian)
    sobel_edges_noisy_gaussian = sobel_edge_detection(noisy_gaussian)
    canny_edges_noisy_gaussian = canny_edge_detection(noisy_gaussian)

    prewitt_edges_noisy_sp = prewitt_edge_detection(noisy_sp)
    sobel_edges_noisy_sp = sobel_edge_detection(noisy_sp)
    canny_edges_noisy_sp = canny_edge_detection(noisy_sp)

    # הצגת תוצאות זיהוי קצוות על תמונות עם רעש
    print("Edges on Gaussian Noisy Image")
    display_edge_detection_results(noisy_gaussian, prewitt_edges_noisy_gaussian, sobel_edges_noisy_gaussian,
                                   canny_edges_noisy_gaussian)

    print("Edges on Salt and Pepper Noisy Image")
    display_edge_detection_results(noisy_sp, prewitt_edges_noisy_sp, sobel_edges_noisy_sp, canny_edges_noisy_sp)

    # אלגוריתם משולב לזיהוי קצוות
    combined_edges = combined_edge_detection(image)

    # הצגת תוצאות האלגוריתם המשולב
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(combined_edges, cmap='gray')
    plt.title('Combined Edge Detection')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    image_path = 'example_maze.jpg'
    edge_detection_workflow(image_path)
