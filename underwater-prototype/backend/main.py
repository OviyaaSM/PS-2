# backend/main.py
import os, json, sqlite3, tempfile, uuid
from datetime import datetime
from pathlib import Path
import asyncio

import cv2, numpy as np, aiofiles
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO

from utils import enhance_frame_cv2, compute_metrics, save_bgr_image, make_annotated_image_from_detections, is_image_file

# optional unet module (only loaded if weights exist)
USE_UNET = False
UNET_WEIGHTS = Path(__file__).parent / "unet_weights.h5"
if UNET_WEIGHTS.exists():
    try:
        from unet_model import load_trained_model, enhance_image as unet_enhance_image
        unet_model = load_trained_model(str(UNET_WEIGHTS))
        USE_UNET = True
        print("✅ U-Net loaded from", UNET_WEIGHTS)
    except Exception as e:
        print("⚠️ U-Net failed to load; falling back to CV2 enhancement:", e)
        USE_UNET = False
else:
    print("ℹ️ U-Net weights not found; using CV2 enhancement (fast fallback).")

# Directories
BASE_DIR = Path(__file__).resolve().parent
STORAGE = BASE_DIR / "storage"
ORIG_DIR = STORAGE / "originals"
OUT_DIR = STORAGE / "outputs"
DB_PATH = STORAGE / "meta.db"
for d in [STORAGE, ORIG_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# FastAPI
app = FastAPI(title="Underwater AI Enhancer & Detector")
# Serve a simple static UI optionally (we will produce /dashboard)
FRONTEND_DIR = BASE_DIR.parent / "extension"  # we reuse extension files to serve popup.html as site if needed
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Load YOLOv8n (lightweight). This will download weights if missing.
print("ℹ️ Loading YOLO model (yolov8n). This may download weights on first run...")
yolo = YOLO("yolov8n.pt")

# DB helpers
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id TEXT PRIMARY KEY,
        filename TEXT,
        timestamp TEXT,
        is_video INTEGER,
        alert TEXT,
        metrics TEXT,
        detections TEXT,
        orig_path TEXT,
        enhanced_path TEXT,
        annotated_path TEXT
    )
    """)
    conn.commit()
    conn.close()
init_db()

def insert_record(rec):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO records (id, filename, timestamp, is_video, alert, metrics, detections, orig_path, enhanced_path, annotated_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rec["id"], rec["filename"], rec["timestamp"], int(rec.get("is_video",0)),
        rec.get("alert",""), json.dumps(rec.get("metrics",{})), json.dumps(rec.get("detections",[])),
        str(rec.get("orig_path","")), str(rec.get("enhanced_path","")), str(rec.get("annotated_path",""))
    ))
    conn.commit()
    conn.close()

def get_history(limit=200):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, timestamp, is_video, alert, metrics, detections, orig_path, enhanced_path, annotated_path FROM records ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "filename": r[1],
            "timestamp": r[2],
            "is_video": bool(r[3]),
            "alert": r[4],
            "metrics": json.loads(r[5]) if r[5] else {},
            "detections": json.loads(r[6]) if r[6] else [],
            "orig_path": r[7],
            "enhanced_path": r[8],
            "annotated_path": r[9]
        })
    return items

# Utility: map COCO classes to domain labels
def map_detection_label(coco_label):
    l = coco_label.lower()
    if l == "person":
        return "Diver"
    if l in ("boat", "ship"):
        return "Surface Vessel"
    # keep as-is for other classes
    return coco_label.title()

# Heuristic for submarine candidate:
def flag_submarine_candidate(detections, frame_shape):
    """
    Heuristic: If there's a Surface Vessel large in frame and with low visible superstructure
    or very long slender bbox relative to frame size — mark as candidate.
    NOTE: This is a heuristic and not a reliable submarine detector; it's for alerting only.
    """
    h, w = frame_shape[:2]
    candidates = []
    for d in detections:
        raw = d.get("raw_label","").lower()
        mapped = d.get("label","")
        x1,y1,x2,y2 = d["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        area_ratio = (bw*bh) / (w*h)
        aspect = bw / (bh+1e-8)
        # heuristics: large object (area_ratio>0.02) & slim (aspect>1.8) -> possible vessel-submarine silhouette
        if mapped in ("Surface Vessel", "Ship", "Boat", "surface vessel") and area_ratio > 0.03 and aspect > 1.6:
            candidates.append(d)
    return candidates

# Run YOLO on image (BGR)
def run_yolo_on_frame(img_bgr, conf=0.25):
    # ultralytics expects RGB numpy
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = yolo.predict(source=img_rgb, conf=conf, imgsz=640, verbose=False)
    # results[0].boxes: each box has xyxy, conf, cls
    detections = []
    if len(results) == 0:
        return detections
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return detections
    # boxes.data is tensor-like; convert to numpy list
    for box in boxes.data.tolist():
        # box: [x1, y1, x2, y2, conf, cls]
        x1,y1,x2,y2,conf,cls = box
        cls = int(cls)
        cls_name = yolo.model.names[cls] if hasattr(yolo, "model") else yolo.names.get(cls, str(cls))
        mapped = map_detection_label(cls_name)
        detections.append({
            "label": mapped,
            "raw_label": cls_name,
            "conf": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": cls
        })
    return detections

# Process an image file buffer
async def process_image_buffer(contents, filename, conf_th=0.25):
    # load image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    # Enhance (U-Net if available, else CV2)
    if USE_UNET:
        # save temp file, run UNET enhancement and read output
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
            t.write(contents)
            t.flush()
            tpath = t.name
        out_unet = unet_enhance_image(unet_model, tpath, output_path=str(Path(OUT_DIR)/f"unet_{filename}"), size=(256,256))
        enhanced = cv2.imread(out_unet)
        if enhanced is None:
            # fallback
            enhanced = enhance_frame_cv2(img)
    else:
        enhanced = enhance_frame_cv2(img)
    # metrics
    metrics = compute_metrics(img, enhanced)
    # detect
    detections = run_yolo_on_frame(enhanced, conf=conf_th)
    # heuristic submarine candidates
    # normalize detection bbox ints
    for d in detections:
        d["bbox"] = [int(round(x)) for x in d["bbox"]]
    subs = flag_submarine_candidate(detections, enhanced.shape)
    # save files
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{ts}_{uuid.uuid4().hex[:8]}_{filename}"
    orig_path = ORIG_DIR / base
    enhanced_path = OUT_DIR / f"enh_{base}"
    annotated_path = OUT_DIR / f"ann_{base}"
    # write original
    cv2.imwrite(str(orig_path), img)
    save_bgr_image(enhanced_path, enhanced)
    ann_img = make_annotated_image_from_detections(enhanced, detections)
    save_bgr_image(annotated_path, ann_img)
    alert = ""
    # decide alert
    found_diver = any(d["label"].lower() == "diver" for d in detections)
    found_sub_candidate = len(subs) > 0
    if found_diver and found_sub_candidate:
        alert = "Diver(s) and Submarine candidate(s) detected"
    elif found_diver:
        alert = "Diver(s) detected"
    elif found_sub_candidate:
        alert = "Submarine candidate(s) detected"
    else:
        alert = "No diver/submarine detected"
    record = {
        "id": uuid.uuid4().hex,
        "filename": filename,
        "timestamp": datetime.utcnow().isoformat(),
        "is_video": False,
        "alert": alert,
        "metrics": metrics,
        "detections": detections,
        "orig_path": str(orig_path),
        "enhanced_path": str(enhanced_path),
        "annotated_path": str(annotated_path)
    }
    insert_record(record)
    return record

# Process a video buffer: sample frames every `frame_step` frames, aggregate detections
async def process_video_buffer(contents, filename, conf_th=0.25, frame_step=15):
    # write temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{uuid.uuid4().hex[:8]}_{filename}"
        out_video = OUT_DIR / f"enh_{base}.mp4"
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
        frame_idx = 0
        aggregated_detections = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # every frame_step frames do enhancement+detection
            if frame_idx % frame_step == 0:
                enhanced = enhance_frame_cv2(frame)
                detections = run_yolo_on_frame(enhanced, conf=conf_th)
                # annotate frame
                for d in detections:
                    d["bbox"] = [int(round(x)) for x in d["bbox"]]
                ann = make_annotated_image_from_detections(enhanced, detections)
                aggregated_detections.extend(detections)
                # write annotated frame to video writer (for simplicity write ann each time)
                writer.write(ann)
            else:
                # write original frame (or enhanced) to maintain length
                writer.write(frame)
            frame_idx += 1
        cap.release()
        writer.release()
        # aggregate summary: group by label and max confidence
        summary = {}
        for d in aggregated_detections:
            lbl = d.get("label","unknown")
            if lbl not in summary:
                summary[lbl] = {"count":0, "max_conf":0.0}
            summary[lbl]["count"] += 1
            if d.get("conf", d.get("confidence",0.0)) > summary[lbl]["max_conf"]:
                summary[lbl]["max_conf"] = float(d.get("conf", d.get("confidence",0.0)))
        # build record
        found_diver = any(d["label"].lower()=="diver" for d in aggregated_detections)
        subs = flag_submarine_candidate(aggregated_detections, (height, width))
        found_sub_candidate = len(subs) > 0
        if found_diver and found_sub_candidate:
            alert = "Diver(s) and Submarine candidate(s) detected"
        elif found_diver:
            alert = "Diver(s) detected"
        elif found_sub_candidate:
            alert = "Submarine candidate(s) detected"
        else:
            alert = "No diver/submarine detected"
        rec = {
            "id": uuid.uuid4().hex,
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(),
            "is_video": True,
            "alert": alert,
            "metrics": {},  # per-video metrics can be heavy to compute; leave empty or compute later
            "detections": aggregated_detections,
            "orig_path": str(tmp.name),
            "enhanced_path": str(out_video),
            "annotated_path": str(out_video)
        }
        insert_record(rec)
        return rec
    finally:
        # don't delete tmp here because orig_path uses it; caller may want to keep
        pass

# API endpoints

import cv2
import numpy as np
import torch
from datetime import datetime
import os
from ultralytics import YOLO

# Load YOLO model (adjust path if needed)
MODEL_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\PS-2\underwater-prototype\runs\detect\train\weights\best.pt"
model = YOLO(MODEL_PATH)

# Utility to check file type
def is_image_file(filename: str):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

# Enhance underwater image (simple method)
def enhance_underwater_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    merged = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

# Process a single image
async def process_image_buffer(image_bytes, filename, conf_th=0.25):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Step 1: Enhance
    enhanced = enhance_underwater_image(img)

    # Step 2: Run detection on enhanced image
    results = model.predict(enhanced, conf=conf_th, verbose=False)

    detections = []
    annotated = enhanced.copy()
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls]
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "label": label,
                "conf": conf,
                "bbox": xyxy
            })

            # Draw bounding boxes
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 255, 0) if "diver" in label.lower() else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Step 3: Save outputs
    base = os.path.splitext(os.path.basename(filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    enh_path = f"static/enhanced/{base}_{timestamp}.jpg"
    ann_path = f"static/annotated/{base}_{timestamp}.jpg"

    os.makedirs("static/enhanced", exist_ok=True)
    os.makedirs("static/annotated", exist_ok=True)
    cv2.imwrite(enh_path, enhanced)
    cv2.imwrite(ann_path, annotated)

    return {
        "enhanced_path": enh_path,
        "annotated_path": ann_path,
        "detections": detections
    }

# Process a video file
async def process_video_buffer(video_bytes, filename, conf_th=0.25):
    temp_video = f"temp_{filename}"
    with open(temp_video, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = os.path.splitext(os.path.basename(filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    enh_path = f"static/enhanced/{base}_{timestamp}.mp4"
    ann_path = f"static/annotated/{base}_{timestamp}.mp4"

    os.makedirs("static/enhanced", exist_ok=True)
    os.makedirs("static/annotated", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_enh = cv2.VideoWriter(enh_path, fourcc, fps, (width, height))
    out_ann = cv2.VideoWriter(ann_path, fourcc, fps, (width, height))

    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance
        enhanced = enhance_underwater_image(frame)
        out_enh.write(enhanced)

        # Detect
        results = model.predict(enhanced, conf=conf_th, verbose=False)

        annotated = enhanced.copy()
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "label": label,
                    "conf": conf,
                    "bbox": xyxy
                })

                x1, y1, x2, y2 = map(int, xyxy)
                color = (0, 255, 0) if "diver" in label.lower() else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        out_ann.write(annotated)

    cap.release()
    out_enh.release()
    out_ann.release()
    os.remove(temp_video)

    return {
        "enhanced_path": enh_path,
        "annotated_path": ann_path,
        "detections": detections
    }

@app.get("/download/{fname}")
async def download_file(fname: str):
    # first check outputs then originals
    p = OUT_DIR / fname
    if p.exists():
        return FileResponse(str(p), media_type='application/octet-stream', filename=fname)
    p2 = ORIG_DIR / fname
    if p2.exists():
        return FileResponse(str(p2), media_type='application/octet-stream', filename=fname)
    # allow absolute paths stored (for video original temp)
    if Path(fname).exists():
        return FileResponse(fname, media_type='application/octet-stream', filename=Path(fname).name)
    return JSONResponse({"status":"error","message":"File not found"}, status_code=404)

@app.get("/history")
async def history():
    items = get_history()
    return JSONResponse({"items": items})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    # basic HTML dashboard (table + preview)
    items = get_history()
    html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Underwater Startup Dashboard</title>
        <style>
          body{font-family:Inter, system-ui, Roboto, Arial;background:#071028;color:#e6eef8;padding:18px}
          .card{background:linear-gradient(180deg,#081229,#071424);padding:14px;border-radius:12px;margin-bottom:12px;box-shadow:0 6px 18px rgba(0,0,0,0.6)}
          table{width:100%;border-collapse:collapse}
          th,td{padding:8px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:left}
          .alert{color:#ffcc00;font-weight:700}
          a.btn{background:#0ea5a5;padding:6px 10px;border-radius:8px;color:#022;text-decoration:none}
        </style>
      </head>
      <body>
      <h1>Underwater Enhancer & Detector — History</h1>
    """
    for it in items:
        html += f"<div class='card'><b>File:</b> {it['filename']} &nbsp; <b>Time (UTC):</b> {it['timestamp']}<br>"
        html += f"<b>Alert:</b> <span class='alert'>{it['alert']}</span><br>"
        html += "<b>Detections:</b><table><tr><th>#</th><th>Label</th><th>Conf</th><th>BBox</th></tr>"
        for idx, d in enumerate(it['detections']):
            bbox = d.get('bbox',[])
            conf = d.get('conf', d.get('confidence', 0.0))
            label = d.get('label', d.get('class', ''))
            html += f"<tr><td>{idx+1}</td><td>{label}</td><td>{conf:.2f}</td><td>{bbox}</td></tr>"
        html += "</table>"
        # links to enhanced / annotated if present
        if it.get('enhanced_path'):
            enh_name = Path(it['enhanced_path']).name
            ann_name = Path(it['annotated_path']).name
            html += f"<br><a class='btn' href='/download/{enh_name}' target='_blank'>Download Enhanced</a> <a class='btn' href='/download/{ann_name}' target='_blank'>Download Annotated</a>"
        html += "</div>"
    html += "</body></html>"
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
