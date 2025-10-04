# backend/utils.py
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageFilter, ImageEnhance
import io

def enhance_frame_cv2(img_bgr):
    """
    Lightweight enhancement pipeline:
    - Convert to LAB and apply CLAHE on L channel (contrast)
    - Denoise with bilateralFilter
    - Simple white balance (gray-world scaling)
    - Unsharp masking (sharpen)
    """
    # --- white balance (gray world) ---
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    # avoid divide by zero
    gb = np.mean(b) + 1e-8
    gg = np.mean(g) + 1e-8
    gr = np.mean(r) + 1e-8
    k = (gb + gg + gr) / 3.0
    b = (b * (k / gb))
    g = (g * (k / gg))
    r = (r * (k / gr))
    img_balanced = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)

    # --- convert to LAB and CLAHE ---
    lab = cv2.cvtColor(img_balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- denoise (bilateral) ---
    denoised = cv2.bilateralFilter(img_clahe, d=7, sigmaColor=75, sigmaSpace=75)

    # --- sharpen (unsharp mask) ---
    gaussian = cv2.GaussianBlur(denoised, (0,0), sigmaX=3)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # Slight color boost (blue-green balance correction is left mild)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1]*1.05, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out

def is_image_file(filename):
    f = filename.lower()
    return f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))

def compute_metrics(original_bgr, enhanced_bgr):
    """
    Compute PSNR, SSIM (single-frame comparison using RGB).
    Also compute an approximate UIQM-like score: combination of
    colorfulness, contrast, and laplacian-based sharpness.
    """
    try:
        # Convert to RGB for skimage
        orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        # Ensure float range 0-255
        orig = orig_rgb.astype(np.float64)
        enh = enh_rgb.astype(np.float64)

        psnr = compare_psnr(orig, enh, data_range=255.0)
        # compute SSIM on luminance channel
        orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        ssim = compare_ssim(orig_gray, enh_gray, data_range=255.0)

        # Colorfulness (Hasler & Süsstrunk)
        rg = orig_rgb[:,:,0].astype(np.float32) - orig_rgb[:,:,1].astype(np.float32)
        yb = 0.5*(orig_rgb[:,:,0].astype(np.float32) + orig_rgb[:,:,1].astype(np.float32)) - orig_rgb[:,:,2].astype(np.float32)
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

        # Sharpness: variance of Laplacian on enhanced image
        lap = cv2.Laplacian(enh_gray, cv2.CV_64F)
        sharpness = lap.var()

        # Contrast: RMS contrast
        contrast = np.std(orig_gray)

        # approximate UIQM-like combining (normalized)
        uiqm = (0.35 * (colorfulness / (colorfulness + 1e-8)) +
                0.45 * (sharpness / (sharpness + 1e-8)) +
                0.20 * (contrast / (contrast + 1e-8))) * 100.0

        return {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "uiqm_approx": float(uiqm),
            "colorfulness": float(colorfulness),
            "sharpness": float(sharpness),
            "contrast": float(contrast)
        }
    except Exception as e:
        return {"error": str(e)}
