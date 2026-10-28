import cv2
import numpy as np

def detect_blur(image, threshold=100):
    """
    Detect blur in an image using variance of Laplacian.

    Args:
        image: Input image (RGB)
        threshold: Variance threshold below which image is considered blurry

    Returns:
        tuple: (variance_score, is_blurry_boolean)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute the Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute variance
    variance = laplacian.var()

    # Determine if blurry
    is_blurry = variance < threshold

    return variance, is_blurry