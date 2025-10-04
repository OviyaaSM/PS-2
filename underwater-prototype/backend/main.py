# backend/main.py
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import aiofiles
import cv2
import numpy as np
import sqlite3
import json
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
STORAGE = BASE_DIR / "storage"
ORIG_DIR = STORAGE / "originals"
OUT_DIR = STORAGE / "outputs"
DB_PATH = STORAGE / "meta.db"
for d in [STORAGE, ORIG_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Underwater Enhancer - Local Service")

# Allow CORS for extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Placeholder enhancement
# -----------------------------
def enhance_frame_cv2(img):
    def enhance_frame_cv2(img):
    """
    Simple visual enhancement placeholder:
    - Increase brightness
    - Increase contrast
    """
    if img is None:
        return img
    # Convert to float32 for processing
    img_float = img.astype(np.float32)
    alpha = 1.2  # contrast factor (1.0 = original)
    beta = 20    # brightness factor (0 = original)
    enhanced = cv2.convertScaleAbs(img_float * alpha + beta)
    return enhanced


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"])

def compute_metrics(orig, enhanced):
    """
    Dummy metrics: returns zeros for now
    Replace with actual PSNR, SSIM, UIQM computations later
    """
    return {"psnr": 0, "ssim": 0, "uiqm_approx": 0}

# -----------------------------
# Database helpers
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        orig_path TEXT,
        out_path TEXT,
        filename TEXT,
        timestamp TEXT,
        metrics TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_record(orig_path, out_path, filename, metrics):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO records (orig_path, out_path, filename, timestamp, metrics)
    VALUES (?, ?, ?, ?, ?)
    """, (str(orig_path), str(out_path), filename, datetime.utcnow().isoformat(), json.dumps(metrics)))
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Upload / processing endpoint
# -----------------------------
@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    try:
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"{now}_{file.filename}"
        orig_path = ORIG_DIR / fname
        contents = await file.read()
        async with aiofiles.open(orig_path, 'wb') as f:
            await f.write(contents)

        # If image
        if is_image_file(file.filename):
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            enhanced = enhance_frame_cv2(img)
            out_fname = f"enh_{fname}"
            out_path = OUT_DIR / out_fname
            cv2.imwrite(str(out_path), enhanced)
            metrics = compute_metrics(img, enhanced)
            insert_record(orig_path, out_path, file.filename, metrics)
            return JSONResponse({
                "status": "ok",
                "type": "image",
                "out_name": out_fname,
                "out_url": f"/download/{out_fname}",
                "metrics": metrics
            })

        # Else assume video
        temp_in = str(orig_path)
        cap = cv2.VideoCapture(temp_in)
        if not cap.isOpened():
            return JSONResponse({"status":"error","message":"Cannot open video"}, status_code=400)

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fname = f"enh_{fname.rsplit('.',1)[0]}.mp4"
        out_path = OUT_DIR / out_fname
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        ret, frame = cap.read()
        if not ret:
            writer.release()
            cap.release()
            return JSONResponse({"status":"error","message":"Empty video"}, status_code=400)

        enhanced_first = enhance_frame_cv2(frame)
        writer.write(enhanced_first)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            enhanced = enhance_frame_cv2(frame)
            writer.write(enhanced)

        writer.release()
        cap.release()
        metrics = compute_metrics(frame, enhanced_first)
        insert_record(orig_path, out_path, file.filename, metrics)
        return JSONResponse({
            "status": "ok",
            "type": "video",
            "out_name": out_fname,
            "out_url": f"/download/{out_fname}",
            "metrics": metrics
        })

    except Exception as e:
        return JSONResponse({"status":"error","message":str(e)}, status_code=500)

# -----------------------------
# Download endpoint
# -----------------------------
@app.get("/download/{fname}")
async def download_file(fname: str):
    path = OUT_DIR / fname
    if not path.exists():
        return JSONResponse({"status":"error","message":"File not found"}, status_code=404)
    return FileResponse(str(path), media_type='application/octet-stream', filename=fname)

# -----------------------------
# History endpoint
# -----------------------------
@app.get("/history")
async def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, orig_path, out_path, filename, timestamp, metrics FROM records ORDER BY id DESC LIMIT 200")
    rows = c.fetchall()
    conn.close()
    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "orig_path": r[1],
            "out_path": r[2],
            "filename": r[3],
            "timestamp": r[4],
            "metrics": json.loads(r[5])
        })
    return JSONResponse({"items": items})

# -----------------------------
# Simple dashboard
# -----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html = """<html>
    <head><meta charset="utf-8"/><title>Underwater Enhancer</title></head>
    <body><h1>Dashboard</h1>
    <p>Upload files via /process endpoint. History is at /history.</p>
    </body></html>"""
    return HTMLResponse(content=html, status_code=200)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
