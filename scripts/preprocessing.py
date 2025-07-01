import os
import cv2
from tqdm import tqdm
import numpy as np

# Paths
input_folder = r"C:\Users\Sanya\OneDrive\Document\AISO\NASA-Night-Time-Traffic-Detection-1\dataset\raw test 10"
output_folder = r"C:\Users\Sanya\OneDrive\Document\AISO\NASA-Night-Time-Traffic-Detection-1\dataset\preProcessing\raw test 10"
os.makedirs(output_folder, exist_ok=True)

# Settings
brightness_boost = 40    # +ve to brighten, -ve to darken (range: -127 to +127)
contrast_boost = 1.4     # >1 to increase contrast, <1 to reduce (typical range: 1.0 to 3.0)

def apply_brightness_contrast(image, brightness=0, contrast=1.0):
    # Convert to float32 for precision
    img = image.astype('float32')
    img = img * contrast + brightness
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def suppress_headlights(image):
    # Convert to HSV and isolate bright spots (V channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Create a mask for very bright areas (like headlights)
    mask = cv2.inRange(v, 230, 255)

    # Apply Gaussian blur to the masked bright areas
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    image[mask > 0] = blurred[mask > 0]
    
    return image

# Process all images
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f" Failed to load {filename}")
            continue

        brightnes_processed = apply_brightness_contrast(image, brightness_boost, contrast_boost)
        final_processed = suppress_headlights(brightnes_processed)
        cv2.imwrite(os.path.join(output_folder, filename), final_processed)

print(" Brightness and contrast enhancement complete.")
