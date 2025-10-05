import os, json, sqlite3
from datetime import datetime
from pathlib import Path

import cv2, numpy as np, aiofiles
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO  # Pretrained YOLOv8

from utils import enhance_frame_cv2, is_image_file, compute_metrics  # same as before

# Directories
BASE_DIR = Path(__file__).resolve().parent
STORAGE = BASE_DIR / "storage"
ORIG_DIR, OUT_DIR = STORAGE / "originals", STORAGE / "outputs"
DB_PATH = STORAGE / "meta.db"
for d in [STORAGE, ORIG_DIR, OUT_DIR]: d.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Underwater AI Enhancer & Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pretrained YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # you can replace with yolov8s.pt for better accuracy

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            metrics TEXT,
            detections TEXT
        )
    """)
    conn.commit()
    conn.close()
init_db()

def insert_record(filename, metrics, detections):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO records (filename, timestamp, metrics, detections)
        VALUES (?, ?, ?, ?)
    """, (filename, datetime.utcnow().isoformat(), json.dumps(metrics), json.dumps(detections)))
    conn.commit()
    conn.close()

# --- Detection Function ---
def detect_objects_yolo(image):
    results = yolo_model.predict(image)
    detections = []
    annotated = results[0].plot()
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        detections.append({
            "class": yolo_model.names[int(cls)],
            "confidence": round(float(conf), 2),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
    return annotated, detections

# --- Core API ---
@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"status": "error", "message": "Invalid image"}, status_code=400)

    # Step 1 — Enhance
    enhanced = enhance_frame_cv2(img)
    metrics = compute_metrics(img, enhanced)

    # Step 2 — Detect
    annotated, detections = detect_objects_yolo(enhanced)

    # Step 3 — Save Outputs
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname_base = f"{timestamp}_{file.filename}"
    orig_path = ORIG_DIR / fname_base
    enh_path = OUT_DIR / f"enh_{fname_base}"
    det_path = OUT_DIR / f"det_{fname_base}"

    async with aiofiles.open(orig_path, 'wb') as f:
        await f.write(contents)
    cv2.imwrite(str(enh_path), enhanced)
    cv2.imwrite(str(det_path), annotated)

    insert_record(file.filename, metrics, detections)

    return JSONResponse({
        "status": "ok",
        "original": str(orig_path.name),
        "enhanced": str(enh_path.name),
        "detected": str(det_path.name),
        "metrics": metrics,
        "detections": detections
    })

@app.get("/download/{fname}")
async def download_file(fname: str):
    path = OUT_DIR / fname
    if not path.exists():
        return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)
    return FileResponse(str(path), media_type='application/octet-stream', filename=fname)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename, timestamp, metrics, detections FROM records ORDER BY id DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()

    html = """<html><head>
    <title>Underwater AI Dashboard</title>
    <style>
      body{background:#0f172a;color:#e6eef8;font-family:Inter;padding:20px}
      .card{background:#111c3d;border-radius:14px;padding:18px;margin-bottom:18px;box-shadow:0 6px 18px rgba(0,0,0,0.6);}
      h1{color:#34d399;}
      table{width:100%;border-collapse:collapse}
      th,td{border-bottom:1px solid rgba(255,255,255,0.1);padding:8px}
    </style></head><body>
    <h1>Underwater AI — Enhancement + Detection Dashboard</h1>"""
    for f, t, m, d in rows:
        m = json.loads(m)
        d = json.loads(d)
        html += f"<div class='card'><b>File:</b> {f}<br><b>Processed:</b> {t}<br>"
        html += f"<b>Metrics:</b> PSNR={m.get('psnr',0):.2f}, SSIM={m.get('ssim',0):.2f}, UIQM={m.get('uiqm_approx',0):.2f}<br>"
        html += "<b>Detections:</b><table><tr><th>Class</th><th>Confidence</th><th>BBox</th></tr>"
        for det in d:
            html += f"<tr><td>{det['class']}</td><td>{det['confidence']}</td><td>{det['bbox']}</td></tr>"
        html += "</table></div>"
    html += "</body></html>"
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
