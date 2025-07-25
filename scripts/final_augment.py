import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import shutil

# === CONFIG ===
DATASET_ROOT = r"C:\Users\Sanya\OneDrive\Document\projects\YOLOv8BDD100k\dataset\Preprocessed\BDD100K Night time.yolov8"
RAW_TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
RAW_TRAIN_LABELS = os.path.join(DATASET_ROOT, "train", "labels")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "train_augmented")

TARGET_SIZE = (640, 640)

# === CLEAN OUTPUT ===
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(os.path.join(OUTPUT_DIR, "images"))
os.makedirs(os.path.join(OUTPUT_DIR, "labels"))

# === AUGMENTATION PIPELINES ===
AUGMENTATIONS = {
    "flip_light": A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params=A.BboxParams(format='yolo')),

    "blur_bright": A.Compose([
        A.MotionBlur(blur_limit=3, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=1.0),
        A.Rotate(limit=10, p=1.0),
        A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params=A.BboxParams(format='yolo')),

    "noise_bright": A.Compose([
        A.GaussNoise(var_limit=(2,3), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Rotate(limit=10, p=1.0),
        A.PadIfNeeded(*TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params=A.BboxParams(format='yolo'))
}

# === MAIN PROCESS ===
images = [f for f in os.listdir(RAW_TRAIN_IMAGES) if f.lower().endswith(('.jpg','.jpeg','.png'))]

for img_file in tqdm(images, desc="Augmenting"):
    img_path = os.path.join(RAW_TRAIN_IMAGES, img_file)
    label_path = os.path.join(RAW_TRAIN_LABELS, os.path.splitext(img_file)[0] + '.txt')

    if not os.path.exists(label_path):
        print(f"Missing label for {img_file}, skipping.")
        continue

    # Read image and labels
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to read {img_file}, skipping.")
        continue

    with open(label_path, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    if not lines:
        print(f"No labels in {label_path}, skipping.")
        continue

    class_ids = [int(parts[0]) for parts in lines]
    bboxes = [list(map(float, parts[1:])) for parts in lines]  # YOLO format: xc, yc, w, h

    for aug_name, aug in AUGMENTATIONS.items():
        try:
            result = aug(image=image, bboxes=bboxes)
            aug_img = result['image']
            aug_bboxes = result['bboxes']

            # skip if no boxes survived augmentation
            if not aug_bboxes:
                print(f"All bboxes removed by {aug_name} on {img_file}, skipping.")
                continue

            # Save image
            out_img_name = f"{os.path.splitext(img_file)[0]}_{aug_name}.jpg"
            out_label_name = f"{os.path.splitext(img_file)[0]}_{aug_name}.txt"

            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", out_img_name), aug_img)

            # Save labels
            with open(os.path.join(OUTPUT_DIR, "labels", out_label_name), 'w') as f:
                for cls, box in zip(class_ids, aug_bboxes):
                    f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")

        except Exception as e:
            print(f"Failed augmentation {aug_name} on {img_file}: {str(e)}")

print("\n✅ Done! Augmented images and labels are in:", OUTPUT_DIR)
