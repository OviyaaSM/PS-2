# Underwater Enhancer — Prototype

## 1) Start backend (required)
cd backend
python -m venv venv
# Activate venv:
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/dashboard to view history.

## 2) Load Chrome extension
Open Chrome → chrome://extensions → enable Developer Mode → Load unpacked → select the `extension/` folder.

Open the extension popup, select a CCTV image or video, click Enhance. The extension will POST to the backend and return a download link & quality metrics.

Notes:
- The backend uses a fast OpenCV-based enhancement pipeline (lightweight). Replace `enhance_frame_cv2` in backend/utils.py with your DL model inference when ready.
- For live CCTV integration later: build a local service that pulls RTSP streams and passes frames into the same enhancement function.
# PS 2
Underwater Video Enhancement
