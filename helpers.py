import cv2
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

def apply_gaussian_blur(image, kernel):
    k = kernel.shape[0] // 2
    blurred = np.zeros_like(image)

    for i in range(k, image.shape[0] - k):
        for j in range(k, image.shape[1] - k):
            region = image[i-k:i+k+1, j-k:j+k+1]
            if image.ndim == 3:
                # For color images, apply kernel to each channel independently
                blurred[i, j] = np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1))
            else:
                # For grayscale images
                blurred[i, j] = np.sum(region * kernel)
    
    return blurred

def apply_grayscale(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

def gaussian_kernel(size, sigma):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def sharpen_image(image, alpha=1.0):
    image = image.astype(np.float64)

    blurred = apply_gaussian_blur(
        image,
        gaussian_kernel(5, 1.0)
    )

    sharpened = image + alpha * (image - blurred)
    
    return np.clip(sharpened, 0, 255)

def to_uint8(image):
    if image.dtype == np.uint8:
        return image
    
    img = image.astype(np.float64)

    if img.max() <= 1.0:
        img = img * 255
    
    return np.clip(img, 0, 255).astype(np.uint8)

def histogram_equalisation(image):
    """
    Apply histogram equalization to improve image contrast.
    Works with both grayscale (2D) and color (3D) images.
    For color images, equalizes the luminance channel in LAB color space.
    """
    # Convert to uint8 if needed
    img = to_uint8(image)
    
    if img.ndim == 2:
        # Grayscale image - direct histogram equalisation
        equalised = cv2.equalizeHist(img)
        return equalised.astype(np.float64)
    else:
        # Color image - convert to LAB, equalise L channel, convert back
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Convert to LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        # Equalize the L (lightness) channel
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        # Convert back to BGR
        bgr_equalised = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Convert back to RGB
        rgb_equalised = cv2.cvtColor(bgr_equalised, cv2.COLOR_BGR2RGB)
        return rgb_equalised.astype(np.float64)