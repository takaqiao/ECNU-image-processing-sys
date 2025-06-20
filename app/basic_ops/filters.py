import cv2
import numpy as np

def gaussian_blur(img, radius):
    return cv2.GaussianBlur(img, (2 * radius + 1, 2 * radius + 1), 0)