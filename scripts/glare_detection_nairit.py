import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
from datetime import datetime
import torch
import torchvision
from PIL import Image
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='glare_reduction_advanced.log'
)
logger = logging.getLogger()

class GlareProcessor:
    def __init__(self, use_dl=True, gamma_dark=2.0, gamma_halo=3.0, gamma_glare=3.5,
                 adaptive_clip=True, denoise_strength=0.1):
        """
        Advanced glare reduction processor combining traditional CV and deep learning
        
        Parameters:
        - use_dl: Whether to use deep learning for glare detection (default True)
        - gamma_dark: Power for dark region enhancement (default 2.0)
        - gamma_halo: Power for halo region suppression (default 3.0)
        - gamma_glare: Power for glare region suppression (default 3.5)
        - adaptive_clip: Use adaptive histogram clipping (default True)
        - denoise_strength: Strength of denoising (0-1, default 0.1)
        """
        self.use_dl = use_dl
        self.gamma_dark = gamma_dark
        self.gamma_halo = gamma_halo
        self.gamma_glare = gamma_glare
        self.adaptive_clip = adaptive_clip
        self.denoise_strength = denoise_strength
        
        if self.use_dl:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                self.dl_model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
                self.dl_model = self.dl_model.to(self.device).eval()
                self.dl_transform = weights.transforms()
                logger.info("Deep learning model loaded successfully")
            except Exception as e:
                logger.error(f"DL model initialization failed: {str(e)}")
                self.use_dl = False
        
        logger.info("GlareProcessor initialized with config: %s", {
            'use_dl': self.use_dl,
            'gamma_dark': gamma_dark,
            'gamma_halo': gamma_halo,
            'gamma_glare': gamma_glare,
            'adaptive_clip': adaptive_clip,
            'denoise_strength': denoise_strength
        })

    def _calculate_thresholds(self, gray_img):
        """Calculate thresholds for different brightness regions"""
        try:
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / (hist.sum() + 1e-6)
            cdf = hist_norm.cumsum()
            
            regions = [
                (0, 50),    # Dark region
                (50, 150),  # Medium region
                (150, 200), # Bright region
                (200, 255)  # Glare region
            ]
            
            thresholds = []
            for a, b in regions:
                for _ in range(100):
                    if (b - a) <= 1: break
                    x1 = a + 0.382 * (b - a)
                    x2 = a + 0.618 * (b - a)
                    if cdf[int(x1)] <= cdf[int(x2)]:
                        b = x2
                    else:
                        a = x1
                threshold = int((a + b) / 2)
                thresholds.append(max(1, min(254, threshold)))
            
            return thresholds
        except Exception as e:
            logger.error(f"Threshold error: {e}")
            return [30, 100, 180, 220]

    def _detect_glare_dl(self, img):
        """Use deep learning to detect glare regions"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            input_tensor = self.dl_transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.dl_model(input_tensor)['out'][0]
            
            mask = torch.argmax(output, dim=0).cpu().numpy()
            glare_mask = ((mask == 15) | (mask == 2)).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            return cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
        except Exception as e:
            logger.error(f"DL detection failed: {str(e)}")
            return None

    def _detect_glare_cv(self, img):
        """Traditional CV method for glare detection"""
        try:
            # Convert to HSV and get value channel
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            v = hsv[:,:,2]
            
            # Adaptive thresholding for glare
            glare_mask = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)[1]
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
            glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
            
            return glare_mask
        except Exception as e:
            logger.error(f"CV detection failed: {str(e)}")
            return None

    def _adaptive_histogram_clip(self, img, clip_limit=2.0, grid_size=(8,8)):
        """Apply CLAHE for contrast enhancement"""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.error("CLAHE failed: %s", str(e))
            return img

    def _denoise_image(self, img):
        """Apply non-local means denoising"""
        try:
            return cv2.fastNlMeansDenoisingColored(
                img, None,
                h=10 * self.denoise_strength,
                hColor=10 * self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        except Exception as e:
            logger.error("Denoising failed: %s", str(e))
            return img

    def _enhance_details(self, img, strength=0.5):
        """Enhance image details using guided filter"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray) / 255.0
            radius = int(max(img.shape) * 0.01)
            eps = (0.01 * 255) ** 2
            
            base = cv2.ximgproc.guidedFilter(
                guide=gray,
                src=gray,
                radius=radius,
                eps=eps,
                dDepth=-1
            )
            
            detail = gray - base
            enhanced = base + (detail * strength)
            enhanced = np.uint8(np.clip(enhanced * 255, 0, 255))
            
            result = img.copy()
            for c in range(3):
                result[:,:,c] = cv2.addWeighted(
                    img[:,:,c], 1 - strength,
                    cv2.merge([enhanced]*3)[:,:,c], strength,
                    0
                )
            return result
        except Exception as e:
            logger.error("Detail enhancement failed: %s", str(e))
            return img

    def _reduce_glare_regions(self, img, glare_mask):
        """Apply glare reduction to detected regions"""
        try:
            if glare_mask is None or np.sum(glare_mask) == 0:
                return img
                
            # Method 1: Inpainting
            inpainted = cv2.inpaint(img, glare_mask, 3, cv2.INPAINT_NS)
            
            # Method 2: Blending with reduced brightness
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            v = hsv[:,:,2]
            v_reduced = np.clip(v * 0.7, 0, 255)
            hsv_reduced = hsv.copy()
            hsv_reduced[:,:,2] = np.where(glare_mask > 0, v_reduced, v)
            blended = cv2.cvtColor(hsv_reduced, cv2.COLOR_HSV2BGR)
            
            # Combine methods - use inpainted for small regions, blended for large
            small_regions = cv2.erode(glare_mask, None, iterations=2)
            result = np.where(
                cv2.merge([small_regions]*3) > 0,
                inpainted,
                blended
            )
            
            return result.astype(np.uint8)
        except Exception as e:
            logger.error(f"Glare reduction failed: {str(e)}")
            return img

    def _enhance_non_glare(self, img, glare_mask=None):
        """Enhance non-glare regions using adaptive techniques"""
        try:
            if glare_mask is None:
                glare_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t1, t2, t3, t4 = self._calculate_thresholds(gray)
            
            dark_mask = (gray <= (t1 // 2)).astype(np.float32)
            halo_mask = ((gray > (t2 + t3)//2) & (gray <= t4)).astype(np.float32)
            other_mask = 1 - (dark_mask + halo_mask)
            
            # Don't enhance glare regions
            other_mask = np.where(glare_mask > 0, 0, other_mask)
            
            log_t1 = np.log10(max(1, t1/2))
            log_t2t3 = np.log10(max(1, (t2 + t3)/2))
            
            G_dark = (log_t1 / log_t2t3) ** self.gamma_dark
            G_halo = (log_t2t3 / np.log10(max(1, t4))) ** self.gamma_halo
            
            enhanced = np.zeros_like(img, dtype=np.float32)
            for c in range(3):
                channel = img[:, :, c].astype(np.float32) / 255.0
                enhanced[:, :, c] = np.clip(
                    (channel ** G_dark * dark_mask) +
                    (channel ** G_halo * halo_mask) +
                    (channel * other_mask),
                    0, 1) * 255
            
            return enhanced.astype(np.uint8)
        except Exception as e:
            logger.error(f"Non-glare enhancement failed: {str(e)}")
            return img

    def process_image(self, img):
        """Complete glare reduction pipeline"""
        try:
            # 1. Pre-processing
            if self.denoise_strength > 0:
                img = self._denoise_image(img)
            
            # 2. Glare detection
            if self.use_dl:
                glare_mask = self._detect_glare_dl(img)
                if glare_mask is None:  # Fallback to CV if DL fails
                    glare_mask = self._detect_glare_cv(img)
            else:
                glare_mask = self._detect_glare_cv(img)
            
            # 3. Glare reduction
            if glare_mask is not None and np.sum(glare_mask) > 0:
                img = self._reduce_glare_regions(img, glare_mask)
            
            # 4. Enhance non-glare regions
            img = self._enhance_non_glare(img, glare_mask)
            
            # 5. Post-processing
            if self.adaptive_clip:
                img = self._adaptive_histogram_clip(img)
            
            img = self._enhance_details(img)
            
            return img
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return img

def process_single_file(args):
    """Process a single image file"""
    processor, input_path, output_path = args
    logger.info(f"Processing: {input_path}")
    
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to read image: {input_path}")
        
        result = processor.process_image(img)
        if result is None:
            raise RuntimeError("Processing returned None")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, result):
            raise RuntimeError(f"Failed to write: {output_path}")
            
        return output_path
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return None

def process_images_parallel(input_folder, output_folder, config=None):
    """Process images in parallel"""
    start_time = datetime.now()
    logger.info("Starting batch processing")
    
    if not os.path.exists(input_folder):
        logger.error("Input folder does not exist")
        return []
    
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        all_files = os.listdir(input_folder)
        image_files = [f for f in all_files 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            logger.error("No valid image files found")
            return []
    except Exception as e:
        logger.error("Error scanning input folder")
        return []
    
    processor = GlareProcessor(**(config or {}))
    
    args_list = [
        (processor, 
         os.path.join(input_folder, f), 
         os.path.join(output_folder, f))
        for f in image_files
    ]
    
    try:
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            results = list(tqdm(pool.imap(process_single_file, args_list),
                          total=len(args_list),
                          desc="Processing images"))
        
        success_count = sum(1 for r in results if r is not None)
        logger.info("Processing completed in %s", datetime.now() - start_time)
        logger.info("Results: %d successes, %d failures", success_count, len(results) - success_count)
        
        return results
    except Exception as e:
        logger.error("Parallel processing failed")
        return []

if __name__ == "__main__":
    # Configuration
    config = {
        'use_dl': True,  # Use deep learning if available
        'gamma_dark': 2.0,
        'gamma_halo': 3.0,
        'gamma_glare': 3.5,
        'adaptive_clip': True,
        'denoise_strength': 0.05
    }
    
    # Path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "input_images")
    output_folder = os.path.join(script_dir, "output_images_new")
    
    print("=== Advanced Glare Reduction Processor ===")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    
    results = process_images_parallel(input_folder, output_folder, config)
    
    if not results:
        print("\nERROR: No images were processed. Check the log file for details.")
    else:
        success_count = sum(1 for r in results if r is not None)
        print(f"\nCompleted: {success_count}/{len(results)} images processed successfully")

'''
functions
1. calc threshold
2. detect glare
def process_single_file(args):
def process_images_parallel(input_folder, output_folder, config=None):
'''