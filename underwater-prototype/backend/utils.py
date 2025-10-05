# backend/utils.py
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from pathlib import Path

def enhance_frame_cv2(img_bgr, contrast=1.2, brightness=20, clahe=True, denoise=True, sharpening=True):
    """
    Fast visual enhancement pipeline (BGR in, BGR out).
    """
    if img_bgr is None:
        return img_bgr
    img = img_bgr.copy()
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe_op.apply(l)
        lab = cv2.merge((l2, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if denoise:
        img = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
    img = cv2.convertScaleAbs(img.astype(np.float32) * contrast + brightness)
    if sharpening:
        gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    # slight saturation boost
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1]*1.05, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def is_image_file(filename):
    f = filename.lower()
    return f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))

def compute_metrics(original_bgr, enhanced_bgr):
    try:
        orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        enh_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        psnr = compare_psnr(orig_rgb, enh_rgb, data_range=255.0)
        orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        ssim = compare_ssim(orig_gray, enh_gray, data_range=255.0)
        rg = orig_rgb[:,:,0] - orig_rgb[:,:,1]
        yb = 0.5*(orig_rgb[:,:,0] + orig_rgb[:,:,1]) - orig_rgb[:,:,2]
        colorfulness = np.sqrt(rg.var() + yb.var()) + 0.3*np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        lap = cv2.Laplacian(enh_gray, cv2.CV_64F)
        sharpness = lap.var()
        contrast = np.std(orig_gray)
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

def save_bgr_image(path, img_bgr):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img_bgr)
    return str(p)

def make_annotated_image_from_detections(img_bgr, detections, box_color=(0,255,160), thickness=2):
    img = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det.get("label", det.get("class", "obj"))
        conf = det.get("conf", det.get("confidence", 0.0))
        cv2.rectangle(img, (x1,y1), (x2,y2), box_color, thickness)
        txt = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-6), (x1+tw+6, y1), box_color, -1)
        cv2.putText(img, txt, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2,2,2), 1, cv2.LINE_AA)
    return img
