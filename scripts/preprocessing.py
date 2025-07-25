import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

input_root = r"C:\Users\Sanya\OneDrive\Document\projects\YOLOv8BDD100k\dataset\BDD100K Night time.yolov8"
output_root = r"C:\Users\Sanya\OneDrive\Document\projects\YOLOv8BDD100K\dataset\Preprocessed\BDD100K Night time.yolov8"
brightness_boost = -20
contrast_boost = 0.9

def apply_brightness_contrast(image, brightness, contrast):
    img = image.astype('float32') * contrast + brightness
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

for split in ['train', 'val', 'test']:
    input_folder = os.path.join(input_root, split, 'images')
    label_folder = os.path.join(input_root, split, 'labels')
    output_img_folder = os.path.join(output_root, split, 'images')
    output_lbl_folder = os.path.join(output_root, split, 'labels')
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_lbl_folder, exist_ok=True)
    
    counter = 0
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(os.path.join(input_folder, filename))
            if image is None: continue
            new_name = f"B{counter:04d}.jpg"
            final_img = apply_brightness_contrast(image, brightness_boost, contrast_boost)
            cv2.imwrite(os.path.join(output_img_folder, new_name), final_img)
            # copy & rename label safely
            label_name = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(label_folder, label_name)
            dst_label = os.path.join(output_lbl_folder, f"B{counter:04d}.txt")
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            counter +=1

print("All done.")



'''def apply_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def glare_removal(image, threshold=200, dilation_kernel=(15,15), inpaint_radius=3, blend_ratio=0.7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones(dilation_kernel, np.uint8))
    
    inpainted = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_NS)  # or INPAINT_TELEA
    return cv2.addWeighted(image, blend_ratio, inpainted, 1 - blend_ratio, 0)
'''