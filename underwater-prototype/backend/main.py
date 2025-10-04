# backend/main.py
import os
from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import aiofiles
import cv2
import numpy as np
from utils import enhance_frame_cv2, is_image_file, compute_metrics
import sqlite3
import json
from pathlib import Path
from fastapi.staticfiles import StaticFiles

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

# Database helpers
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

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """
    Accepts an image or video and returns processed output path and metrics.
    """
    try:
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"{now}_{file.filename}"
        orig_path = ORIG_DIR / fname
        contents = await file.read()
        async with aiofiles.open(orig_path, 'wb') as f:
            await f.write(contents)

        # If image
        if is_image_file(file.filename):
            # Load image using cv2
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
        # Save temp input and process frames
        temp_in = str(orig_path)
        cap = cv2.VideoCapture(temp_in)
        if not cap.isOpened():
            return JSONResponse({"status":"error","message":"Cannot open video"}, status_code=400)

        # Video writer settings (we will write mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fname = f"enh_{fname.rsplit('.',1)[0]}.mp4"
        out_path = OUT_DIR / out_fname
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # We'll compute metrics by comparing the first frame only (fast)
        ret, frame = cap.read()
        if not ret:
            writer.release()
            cap.release()
            return JSONResponse({"status":"error","message":"Empty video"}, status_code=400)

        enhanced_first = enhance_frame_cv2(frame)
        writer.write(enhanced_first)
        # process remaining frames (for speed skip every 1 frame - you can adjust)
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            enhanced = enhance_frame_cv2(frame)
            writer.write(enhanced)
            frame_count += 1

        writer.release()
        cap.release()

        metrics = compute_metrics(frame, enhanced_first)  # approximate by last vs first enhanced
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

@app.get("/download/{fname}")
async def download_file(fname: str):
    path = OUT_DIR / fname
    if not path.exists():
        return JSONResponse({"status":"error","message":"File not found"}, status_code=404)
    return FileResponse(str(path), media_type='application/octet-stream', filename=fname)

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

# Simple dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html = """
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>Underwater Enhancer - Dashboard</title>
    <style>
      body{font-family:Inter,system-ui,Segoe UI,Roboto,'Helvetica Neue',Arial;background:#0f172a;color:#e6eef8;margin:0;padding:20px}
      .card{background:linear-gradient(180deg,#0b1220, #071028);border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 6px 18px rgba(2,6,23,0.6)}
      h1{margin:0 0 12px 0}
      table{width:100%;border-collapse:collapse}
      th,td{padding:8px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.06)}
      a.btn{background:#0ea5a5;padding:6px 10px;border-radius:8px;color:#032;display:inline-block;text-decoration:none}
    </style>
    </head>
    <body>
      <h1>Underwater Enhancer — History</h1>
      <div id="content">Loading…</div>
      <script>
        async function load() {
          const res = await fetch('/history');
          const j = await res.json();
          const items = j.items;
          if(items.length===0){document.getElementById('content').innerHTML='<div class="card">No history yet.</div>';return;}
          let html = '<div class="card"><table><thead><tr><th>Time (UTC)</th><th>File</th><th>PSNR</th><th>SSIM</th><th>UIQM</th><th>Download</th></tr></thead><tbody>';
          for(const it of items){
            html += `<tr>
              <td>${it.timestamp}</td>
              <td>${it.filename}</td>
              <td>${(it.metrics.psnr||0).toFixed(2)}</td>
              <td>${(it.metrics.ssim||0).toFixed(3)}</td>
              <td>${(it.metrics.uiqm_approx||0).toFixed(2)}</td>
              <td><a class="btn" href="${it.out_path.replace('\\\\','/')}" target="_blank">Open</a></td>
            </tr>`;
          }
          html += '</tbody></table></div>';
          document.getElementById('content').innerHTML = html;
        }
        load();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
from flask import Flask, request, jsonify, send_file
from unet_model import load_trained_model, enhance_image

app = Flask(__name__)
model = load_trained_model("unet_weights.h5")   # load trained model at startup

@app.route("/enhance", methods=["POST"])
def enhance():
    file = request.files['file']
    input_path = "temp_input.jpg"
    output_path = "temp_output.jpg"
    file.save(input_path)

    enhanced_path = enhance_image(model, input_path, output_path)
    return send_file(enhanced_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)

