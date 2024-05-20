import cv2


def GaussianBlur(R, G, B, kernel_size, sigma):
    R_blur = cv2.GaussianBlur(R, kernel_size, sigma)
    G_blur = cv2.GaussianBlur(G, kernel_size, sigma)
    B_blur = cv2.GaussianBlur(B, kernel_size, sigma)

    return R_blur, G_blur, B_blur
