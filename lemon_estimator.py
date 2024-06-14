import cv2
import numpy as np
from skimage import color

def resize_image(image, size):
    return cv2.resize(image, size)

def remove_background(image):
    blue_channel = image[:,:,2]
    _, binary_mask = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
    return masked_image

def extract_features(image):
    lab_image = color.rgb2lab(image)
    L = 116 * ((lab_image[:,:,0] / 100) ** (1/3)) - 16
    a = 500 * (((lab_image[:,:,1] + 128) / 255) ** (1/3) - ((lab_image[:,:,0] / 100) ** (1/3)))
    b = 200 * (((lab_image[:,:,0] / 100) ** (1/3)) - ((lab_image[:,:,2] + 128) / 255) ** (1/3))
    return L, a, b

def determine_maturity(a_values):
    mean_a = np.mean(a_values)
    if mean_a <= -8:
        return "Raw"
    elif -8 < mean_a <= -3:
        return "Near Ripe"
    elif -3 < mean_a <= 0:
        return "Ripe"
    else:
        return "Cannot be categorized"

image = cv2.imread('lemon.jpg')
resized_image = resize_image(image, (200, 200))
bg_removed_image = remove_background(resized_image)
final_image = resize_image(bg_removed_image, (64, 64))
L, a, b = extract_features(final_image)
maturity_level = determine_maturity(a)
print("Lemon maturity:", maturity_level)
