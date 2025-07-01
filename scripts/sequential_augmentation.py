import cv2
import os
import numpy as np
import albumentations as A
import json
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms.functional import adjust_brightness, adjust_contrast

class AdvancedAugmentor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = []
        
        # AI-powered augmentation pipeline (Albumentations)
        self.transform = A.Compose([
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(p=0.5),
            ], p=0.5),
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2)  # Changed from Cutout
        ])

    def apply_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def smart_brightness_contrast(self, img):
        """AI-adjusted brightness/contrast based on image stats"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if torch.rand(1) > 0.5:  # 50% chance to adjust
            brightness_factor = 0.8 + 0.4 * torch.rand(1).item()  # 0.8-1.2
            img_pil = adjust_brightness(img_pil, brightness_factor)
        if torch.rand(1) > 0.5:
            contrast_factor = 0.8 + 0.4 * torch.rand(1).item()  # 0.8-1.2
            img_pil = adjust_contrast(img_pil, contrast_factor)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def augment_image(self, img_path):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            return
            
        # Track original stats
        meta = {'filename': base_name, 'original_size': img.shape}
        
        # 1. Grayscale
        gray = self.apply_grayscale(img)
        gray_path = os.path.join(self.output_dir, f"{base_name}_gy.jpg")
        cv2.imwrite(gray_path, gray)
        meta['grayscale'] = gray_path
        
        # 2. AI-adjusted brightness/contrast
        enhanced = self.smart_brightness_contrast(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        enhanced_path = os.path.join(self.output_dir, f"{base_name}_gy_br.jpg")
        cv2.imwrite(enhanced_path, enhanced)
        meta['brightness_contrast'] = enhanced_path
        
        # 3. Advanced augmentations (Albumentations)
        augmented = self.transform(image=enhanced)['image']
        aug_path = os.path.join(self.output_dir, f"{base_name}_gy_br_ai.jpg")
        cv2.imwrite(aug_path, augmented)
        meta['advanced_aug'] = aug_path
        
        # 4. GPU-accelerated flip
        if self.device == 'cuda':
            tensor = torch.from_numpy(augmented).permute(2,0,1).to(self.device)
            flipped = torch.flip(tensor, [2]).cpu().numpy().transpose(1,2,0)
        else:
            flipped = cv2.flip(augmented, 1)
        flip_path = os.path.join(self.output_dir, f"{base_name}_gy_br_ai_fl.jpg")
        cv2.imwrite(flip_path, flipped)
        meta['flipped'] = flip_path
        
        self.metadata.append(meta)

    def save_metadata(self):
        with open(os.path.join(self.output_dir, 'augmentation_metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)

def main():
    input_dir = "output_frames"  # Your images folder
    output_dir = "augmented_frames"  # New folder for augmented images
    
    augmentor = AdvancedAugmentor(output_dir)
    
    print(f"Processing images with {'GPU' if augmentor.device == 'cuda' else 'CPU'} acceleration")
    for img_file in tqdm(os.listdir(input_dir)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            augmentor.augment_image(os.path.join(input_dir, img_file))
    
    augmentor.save_metadata()
    print(f"Augmentation complete! Results saved to {output_dir}")
    print(f"Metadata saved to augmentation_metadata.json")

if __name__ == "__main__":
    main()