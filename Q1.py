import os

import cv2
import matplotlib.pyplot as plt
import Q1_helpers as helpers


def separate_and_display_channels(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    R, G, B = cv2.split(image_rgb)

    H, S, V = cv2.split(image_hsv)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(R, cmap='gray')
    axs[0, 0].set_title('Red Channel')
    axs[0, 1].imshow(G, cmap='gray')
    axs[0, 1].set_title('Green Channel')
    axs[0, 2].imshow(B, cmap='gray')
    axs[0, 2].set_title('Blue Channel')
    axs[1, 0].imshow(H, cmap='gray')
    axs[1, 0].set_title('Hue Channel')
    axs[1, 1].imshow(S, cmap='gray')
    axs[1, 1].set_title('Saturation Channel')
    axs[1, 2].imshow(V, cmap='gray')
    axs[1, 2].set_title('Value Channel')

    for ax in axs.flat:
        ax.axis('off')

    plt.show()


def detect_maze_walls(image_path):
    #load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #split image into R, G, B channels
    R, G, B = cv2.split(image_rgb)

    #apply Gaussian blur to each channel
    # R_blur = cv2.GaussianBlur(R, (5, 5), 0)
    # G_blur = cv2.GaussianBlur(G, (5, 5), 0)
    # B_blur = cv2.GaussianBlur(B, (5, 5), 0)

    R_blur, G_blur, B_blur = helpers.GaussianBlur(R, G, B, (5, 5), 0)

    #apply Canny edge detection to each channel
    R_edges = cv2.Canny(R_blur, 50, 150)
    G_edges = cv2.Canny(G_blur, 50, 150)
    B_edges = cv2.Canny(B_blur, 50, 150)

    #combine edges from all channels
    combined_edges = cv2.bitwise_or(R_edges, G_edges)
    combined_edges = cv2.bitwise_or(combined_edges, B_edges)

    #display original image and combined edges
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(combined_edges, cmap='gray')
    plt.title('Combined Edges (RGB)')
    plt.axis('off')

    plt.show()
    return


def detect_maze_walls_hsv(image_path):
    # check if file exists
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # convert image to RGB and HSV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # split image into H, S, V channels
    H, S, V = cv2.split(image_hsv)

    # noise reduction with median blur on saturation channel
    S_blur = cv2.medianBlur(S, 5)

    # canny edge detection on saturation channel
    S_edges = cv2.Canny(S_blur, 50, 150)

    # noise reduction with Gaussian blur on RGB channels
    R, G, B = cv2.split(image_rgb)
    R_blur = cv2.GaussianBlur(R, (5, 5), 0)
    G_blur = cv2.GaussianBlur(G, (5, 5), 0)
    B_blur = cv2.GaussianBlur(B, (5, 5), 0)

    # canny edge detection on RGB channels
    R_edges = cv2.Canny(R_blur, 50, 150)
    G_edges = cv2.Canny(G_blur, 50, 150)
    B_edges = cv2.Canny(B_blur, 50, 150)

    # combine edges from RGB channels
    combined_edges_rgb = cv2.bitwise_or(R_edges, G_edges)
    combined_edges_rgb = cv2.bitwise_or(combined_edges_rgb, B_edges)

    # הצגת התוצאות
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image (RGB)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(combined_edges_rgb, cmap='gray')
    plt.title('Combined Edges (RGB)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(S, cmap='gray')
    plt.title('Saturation Channel (HSV)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(S_edges, cmap='gray')
    plt.title('Edges on Saturation Channel (HSV)')
    plt.axis('off')

    plt.show()


def main():
    separate_and_display_channels('example_maze.jpg')
    detect_maze_walls('example_maze.jpg')


if __name__ == '__main__':
    detect_maze_walls_hsv('example_maze.jpg')
