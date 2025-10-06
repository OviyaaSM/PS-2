# convert_masks_to_yolo.py
import os, argparse, cv2, numpy as np
from pathlib import Path

def ensure_dir(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

def convert(images_dir, masks_dir, out_labels_dir, class_map=None, img_exts=(".jpg",".png",".jpeg"), min_area=100):
    ensure_dir(out_labels_dir)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    for img_path in images_dir.iterdir():
        if not img_path.suffix.lower() in img_exts:
            continue
        stem = img_path.stem
        # attempt common mask extensions
        mask_path = None
        for ext in (".png", ".jpg", ".tif", ".bmp"):
            p = masks_dir / (stem + ext)
            if p.exists():
                mask_path = p
                break
        if mask_path is None:
            # no mask: create empty label
            open(Path(out_labels_dir)/f"{stem}.txt","w").close()
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            open(Path(out_labels_dir)/f"{stem}.txt","w").close()
            continue
        # convert color masks to single-channel if needed
        if len(mask.shape) == 3:
            # if it is RGB but contains integer class ids stored in channels, convert to grayscale
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        h, w = mask.shape[:2]
        labels = []
        unique_vals = np.unique(mask)
        for val in unique_vals:
            if val == 0:
                continue
            # determine class id mapping
            if class_map:
                if str(int(val)) not in class_map:
                    continue
                cls = int(class_map[str(int(val))])
            else:
                cls = 0
            # create binary mask for this value
            bin_mask = (mask == val).astype('uint8')*255
            contours,_ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x,y,ww,hh = cv2.boundingRect(cnt)
                area = ww*hh
                if area < min_area:
                    continue
                x_center = (x + ww/2.0) / w
                y_center = (y + hh/2.0) / h
                w_norm = ww / w
                h_norm = hh / h
                labels.append((cls, x_center, y_center, w_norm, h_norm))
        # write labels
        out_file = Path(out_labels_dir)/f"{stem}.txt"
        with open(out_file, "w") as f:
            for item in labels:
                f.write(f"{item[0]} {item[1]:.6f} {item[2]:.6f} {item[3]:.6f} {item[4]:.6f}\n")
    print("Conversion complete. Labels written to", out_labels_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--out_labels_dir", required=True)
    parser.add_argument("--class_map", default=None, help="JSON-like mapping e.g. '1:0,2:1' meaning mask-value 1 -> class 0")
    args = parser.parse_args()

    class_map = None
    if args.class_map:
        class_map = {k:v for k,v in (x.split(":") for x in args.class_map.split(","))}
    convert(args.images_dir, args.masks_dir, args.out_labels_dir, class_map)
