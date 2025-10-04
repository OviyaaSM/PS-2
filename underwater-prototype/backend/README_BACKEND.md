# Underwater Enhancer - Backend

## Setup (recommended)
1. Create & activate a Python venv:
   - `python -m venv venv`
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

2. Install requirements:
   - `pip install -r requirements.txt`

3. Run:
   - `uvicorn main:app --reload --host 127.0.0.1 --port 8000`

4. Open dashboard:
   - http://127.0.0.1:8000/dashboard

API:
- POST `/process` - form file upload field name `file`. Returns JSON with `out_url` and `metrics`.
- GET `/history` - returns JSON history.
- GET `/download/{fname}` - download processed file.
