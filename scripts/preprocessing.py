import os
import cv2
from tqdm import tqdm

# Paths
input_folder = "/content/night_sunny-3/all/images"
output_folder = "/content/night_sunny-3/all/bright_images"
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

# Process all images
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Failed to load {filename}")
            continue

        processed = apply_brightness_contrast(image, brightness_boost, contrast_boost)
        cv2.imwrite(os.path.join(output_folder, filename), processed)

print(" Brightness and contrast enhancement complete.")
