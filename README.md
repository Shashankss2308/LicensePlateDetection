# PlateVision AI — License Plate Recognition

> Real-time license plate detection powered by YOLOv8 + EasyOCR

## Project Structure
```
LPRS/
├── backend/          ← Python Flask backend
│   ├── app.py        ← Main server (YOLO + OCR + History API)
│   ├── database.py   ← SQLite history database
│   ├── wsgi.py       ← WSGI entry point for Gunicorn
│   ├── requirements.txt
│   ├── start.bat     ← Windows one-click start
│   └── platevision.db  (auto-created)
├── frontend/         ← Static web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
├── best.pt           ← Trained YOLOv8 model
└── Procfile          ← Deployment config
```

## Running Locally (Windows)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Or just double-click **`backend/start.bat`**

Then open: **http://localhost:5000**

## Deploying to Render / Railway

1. Push this entire folder to a GitHub repository
2. Create a new Web Service on [Render](https://render.com) or [Railway](https://railway.app)
3. Connect your GitHub repository
4. Set the **Start Command** to:
   ```
   cd backend && gunicorn wsgi:app -w 1 --timeout 120
   ```
5. Deploy — your site will get a public URL automatically

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect` | Upload image, returns plate text |
| `GET`  | `/api/history` | Fetch all scan history |
| `DELETE` | `/api/history/<id>` | Delete one history entry |
| `DELETE` | `/api/history` | Clear all history |

## Tech Stack
- **YOLOv8** — License plate detection
- **EasyOCR** — Text character extraction
- **Flask + Waitress/Gunicorn** — Production web server
- **SQLite** — Scan history storage
- **OpenCV + Pillow** — Image preprocessing
