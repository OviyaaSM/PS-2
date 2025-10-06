import os
from PyQt5.QtCore import QPointF
import json  # if your shapes are saved in JSON
from glob import glob

# Modify these paths
IMAGES_DIR = r'C:\Users\ADMIN\OneDrive\Desktop\PS-2\underwater-prototype\underwater_dataset\images'
ANNOTATIONS_DIR = r'C:\Users\ADMIN\OneDrive\Desktop\PS-2\underwater-prototype\underwater_dataset\annotations'  # where you saved shapes
YOLO_LABELS_DIR = r'C:\Users\ADMIN\OneDrive\Desktop\PS-2\underwater-prototype\underwater_dataset\labels'

# Class names must match your data.yaml
CLASS_NAMES = ['diver', 'submarine', 'boat']

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

def normalize(value, max_value):
    return float(value) / float(max_value)

def convert_shape_to_yolo(shape, image_width, image_height):
    # Assuming shape.points = [top-left, top-right, bottom-right, bottom-left]
    xs = [p.x() for p in shape.points]
    ys = [p.y() for p in shape.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = normalize((x_min + x_max) / 2, image_width)
    y_center = normalize((y_min + y_max) / 2, image_height)
    width = normalize(x_max - x_min, image_width)
    height = normalize(y_max - y_min, image_height)

    return x_center, y_center, width, height

# Example for JSON annotation format
for json_file in glob(os.path.join(ANNOTATIONS_DIR, '*.json')):
    with open(json_file, 'r') as f:
        data = xml.load(f)

    image_name = os.path.splitext(os.path.basename(json_file))[0]
    image_file = os.path.join(IMAGES_DIR, image_name + '.jpg')  # adjust extension if needed

    from PIL import Image
    img = Image.open(image_file)
    img_w, img_h = img.size

    yolo_lines = []
    for shape in data['shapes']:  # adjust key if your JSON structure differs
        label = shape['label']
        if label not in CLASS_NAMES:
            continue
        class_idx = CLASS_NAMES.index(label)
        points = [QPointF(p[0], p[1]) for p in shape['points']]
        # Dummy Shape object
        class DummyShape:
            def __init__(self, pts):
                self.points = pts
        s = DummyShape(points)
        x_c, y_c, w, h = convert_shape_to_yolo(s, img_w, img_h)
        yolo_lines.append(f"{class_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Write YOLO .txt file
    label_file = os.path.join(YOLO_LABELS_DIR, image_name + '.txt')
    with open(label_file, 'w') as f:
        f.write("\n".join(yolo_lines))

print("YOLO labels generated successfully!")
