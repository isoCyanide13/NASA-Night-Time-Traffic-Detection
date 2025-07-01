import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import albumentations as A
import shutil

# Configuration
RAW_DATA_DIR = "Train_raw"
AUGMENTED_DATA_DIR = "Train_augmented"
TARGET_SIZE = (640, 640)

def enhance_lights(image, intensity=0.3):
    """Enhances headlights/streetlights with localized yellow glow"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    light_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    yellow_glow = np.zeros_like(image)
    yellow_glow[light_mask > 0] = [0, 200, 200]  # BGR yellow
    yellow_glow = cv2.GaussianBlur(yellow_glow, (51, 51), 0)
    return cv2.addWeighted(image, 1, yellow_glow, intensity, 0)

def process_1_flip_light_enhance(image, bboxes):
    """Horizontal flip + light enhancement"""
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params=A.BboxParams(format='yolo'))
    processed = enhance_lights(image)
    return transform(image=processed, bboxes=bboxes)

def process_2_blur_bright_rotate(image, bboxes):
    """Gaussian Blur + brightness + rotation"""
    return A.Compose([
        # Gaussian Blur (3x3 to 7x7 kernel)
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        
        A.RandomBrightnessContrast(brightness_limit=0.2, p=1.0),
        A.Affine(rotate=(-10, 10), p=1.0),
        A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params=A.BboxParams(format='yolo'))(image=image, bboxes=bboxes)

def process_3_noise_bright_rotate(image, bboxes):
    """Subtle Gaussian Noise + reduced brightness + rotation"""
    return A.Compose([
        # Reduced Gaussian Noise (very subtle)
        A.GaussNoise(var_limit=(5, 15), p=1.0),  # Reduced from (10, 30)
        
        # Reduced brightness adjustment
        A.RandomBrightnessContrast(brightness_limit=0.1, p=1.0),  # Reduced from 0.2
        A.Affine(rotate=(-10, 10), p=1.0),
        A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params=A.BboxParams(format='yolo'))(image=image, bboxes=bboxes)

def process_4_grayscale_crop(image, bboxes):
    """Grayscale + crop + pad"""
    return A.Compose([
        A.ToGray(p=1.0),
        A.RandomCrop(width=600, height=600, p=1.0),
        A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params=A.BboxParams(format='yolo'))(image=image, bboxes=bboxes)

def process_image(image_path, output_prefix):
    try:
        # Load image and labels
        image = cv2.imread(image_path)
        if image is None: return 0
        
        with open(os.path.splitext(image_path)[0] + '.txt') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f if len(line.strip().split()) == 5]
        if not bboxes: return 0

        # Resize original
        base = A.Compose([
            A.LongestMaxSize(max_size=max(TARGET_SIZE)),
            A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        ], bbox_params=A.BboxParams(format='yolo'))(image=image, bboxes=[b[1:] for b in bboxes])
        
        base_img, base_boxes = base['image'], base['bboxes']
        class_ids = [int(b[0]) for b in bboxes]

        # Save original
        cv2.imwrite(f"{output_prefix}_original.jpg", base_img)
        with open(f"{output_prefix}_original.txt", 'w') as f:
            for (x1,y1,x2,y2), cls in zip(base_boxes, class_ids):
                f.write(f"{cls} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n")

        # Process pipelines
        processors = [
            ("flip_light", process_1_flip_light_enhance),
            ("blur_bright", process_2_blur_bright_rotate),
            ("noise_bright", process_3_noise_bright_rotate),
            ("grayscale", process_4_grayscale_crop)
        ]

        for name, processor in processors:
            try:
                result = processor(base_img.copy(), base_boxes.copy())
                cv2.imwrite(f"{output_prefix}_{name}.jpg", result['image'])
                with open(f"{output_prefix}_{name}.txt", 'w') as f:
                    for (x1,y1,x2,y2), cls in zip(result['bboxes'], class_ids):
                        f.write(f"{cls} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n")
            except Exception as e:
                print(f"Skipping {name} for {os.path.basename(image_path)}: {str(e)}")
        
        return 1
    except Exception as e:
        print(f"Failed {os.path.basename(image_path)}: {str(e)}")
        return 0

def main():
    # Setup
    if os.path.exists(AUGMENTED_DATA_DIR):
        shutil.rmtree(AUGMENTED_DATA_DIR)
    os.makedirs(AUGMENTED_DATA_DIR)

    # Process all images
    images = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    with tqdm(images, desc="Augmenting") as pbar:
        for img in pbar:
            process_image(
                os.path.join(RAW_DATA_DIR, img),
                os.path.join(AUGMENTED_DATA_DIR, os.path.splitext(img)[0])
            )

    # Verification
    output_files = os.listdir(AUGMENTED_DATA_DIR)
    print(f"\nGenerated {len([f for f in output_files if f.endswith('.jpg')])} images")
    print(f"Generated {len([f for f in output_files if f.endswith('.txt')])} labels")

if __name__ == "__main__":
    main()