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


def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        row, col = image.shape
        mean = 0
        var = 30
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
        noisy[coords[0], coords[1]] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1]] = 0
        return noisy


def average_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))


def median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)


def psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssim(original, filtered):
    C1 = 6.5025
    C2 = 58.5225
    original = original.astype(np.float64)
    filtered = filtered.astype(np.float64)
    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(filtered, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(filtered ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * filtered, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + C1) * (mu2_sq + C2))
    return ssim_map.mean()


def display_results(original, noisy, avg_filtered, median_filtered, noise_type):
    psnr_avg = psnr(original, avg_filtered)
    ssim_avg = ssim(original, avg_filtered)
    psnr_median = psnr(original, median_filtered)
    ssim_median = ssim(original, median_filtered)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'Noisy Image ({noise_type})')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(avg_filtered, cmap='gray')
    plt.title(f'Average Filter (PSNR: {psnr_avg:.2f}, SSIM: {ssim_avg:.4f})')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(median_filtered, cmap='gray')
    plt.title(f'Median Filter (PSNR: {psnr_median:.2f}, SSIM: {ssim_median:.4f})')
    plt.axis('off')

    plt.show()


def noise_reduction_workflow(image_path):
    # טעינת התמונה
    image = load_image(image_path)
    if image is None:
        return

    # הוספת רעש גאוסי וסינון
    noisy_gaussian = add_noise(image, "gaussian")
    avg_filtered_gaussian = average_filter(noisy_gaussian)
    median_filtered_gaussian = median_filter(noisy_gaussian)
    print("Gaussian Noise")
    display_results(image, noisy_gaussian, avg_filtered_gaussian, median_filtered_gaussian, "Gaussian")

    # הוספת רעש מלח ופלפל וסינון
    noisy_sp = add_noise(image, "salt_and_pepper")
    avg_filtered_sp = average_filter(noisy_sp)
    median_filtered_sp = median_filter(noisy_sp)
    print("Salt and Pepper Noise")
    display_results(image, noisy_sp, avg_filtered_sp, median_filtered_sp, "Salt and Pepper")


# דוגמה לשימוש בפונקציה הראשית
if __name__ == '__main__':
    noise_reduction_workflow('example_maze.jpg')
