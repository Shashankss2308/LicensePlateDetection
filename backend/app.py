"""
PlateVision AI — Backend v5 (Optimised)

Pipeline per request:
YOLO Detection
      ↓
Crop Plate
      ↓
PARSeq OCR API
      ↓
Detected Plate Number
      ↓
Frontend Display
"""

"""
PlateVision AI — Backend v5 (Optimised)

Pipeline per request:
YOLO Detection
      ↓
Crop Plate
      ↓
PARSeq OCR API
      ↓
Detected Plate Number
      ↓
Frontend Display
"""

import os
import re
import time
import base64
import io
import tempfile

import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image

from gradio_client import Client, handle_file

from backend.database import (
    init_db,
    save_detection,
    get_all_detections,
    delete_detection,
    clear_all_detections
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.abspath(
    os.path.join(BASE_DIR, '..', 'frontend')
)

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, '..', 'best.pt')
)

# ─────────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=''
)

init_db()

# ─────────────────────────────────────────────────────────────
# YOLO MODEL
# ─────────────────────────────────────────────────────────────
print("[PlateVision] Loading YOLO model...")
yolo = YOLO(MODEL_PATH)

# ─────────────────────────────────────────────────────────────
# PARSeq OCR API
# ─────────────────────────────────────────────────────────────
print("[PlateVision] Loading PARSeq OCR API...")

parseq_client = Client("baudm/PARSeq-OCR")

print("[PlateVision] ✓ PARSeq OCR Ready!")


# ─────────────────────────────────────────────────────────────
# PARSeq OCR FUNCTION
# ─────────────────────────────────────────────────────────────
def recognize_plate_parseq(plate_bgr):

    temp_path = None

    try:

        # Upscale for better OCR
        plate_bgr = cv2.resize(
            plate_bgr,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # Convert grayscale
        gray = cv2.cvtColor(
            plate_bgr,
            cv2.COLOR_BGR2GRAY
        )

        # CLAHE enhancement
        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        # Save temporary image
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:

            temp_path = temp.name

        cv2.imwrite(temp_path, enhanced)

        # PARSeq API call
        result = parseq_client.predict(
            model_name="parseq",
            image=handle_file(temp_path),
            api_name="/App"
        )

        if result:

            # Extract OCR text
            if isinstance(result, tuple):

                text = result[0]

            elif isinstance(result, list):

                text = result[0]

            else:

                text = str(result)

            text = str(text).upper()

            # Keep only alphanumeric
            text = re.sub(
                r'[^A-Z0-9]',
                '',
                text
            )

            # Remove unwanted words
            text = text.replace("IND", "")
            text = text.replace("LICENSE", "")
            text = text.replace("PLATE", "")
            text = text.replace("HEADERS", "")
            text = text.replace("DATA", "")

            # Limit length
            if len(text) > 10:
                text = text[:10]

            return text, 0.95

        return "", 0.0

    except Exception as e:

        print(f"[PARSeq ERROR] {e}")

        return "", 0.0

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ─────────────────────────────────────────────────────────────
# STATIC ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ─────────────────────────────────────────────────────────────
# DETECTION API
# ─────────────────────────────────────────────────────────────
@app.route('/api/detect', methods=['POST'])
def detect():

    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image provided.'
        })

    file = request.files['image']

    start_time = time.time()

    # Read image
    img_bytes = file.read()

    img_array = np.frombuffer(
        img_bytes,
        np.uint8
    )

    img_bgr = cv2.imdecode(
        img_array,
        cv2.IMREAD_COLOR
    )

    if img_bgr is None:
        return jsonify({
            'success': False,
            'error': 'Could not decode image.'
        })

    h, w = img_bgr.shape[:2]

    # ─────────────────────────────────────────
    # YOLO DETECTION
    # ─────────────────────────────────────────
    results = yolo(
        img_bgr,
        conf=0.25,
        verbose=False
    )

    if not results or len(results[0].boxes) == 0:

        return jsonify({
            'success': False,
            'error': 'No license plate detected.'
        })

    boxes = results[0].boxes

    best_idx = int(boxes.conf.argmax())

    best_box = boxes[best_idx]

    yolo_conf = float(best_box.conf[0])

    x1, y1, x2, y2 = map(
        int,
        best_box.xyxy[0].tolist()
    )

    # ─────────────────────────────────────────
    # PLATE CROP
    # ─────────────────────────────────────────
    pad_l = 6
    pad_t = 6
    pad_r = 14
    pad_b = 10

    x1p = max(0, x1 - pad_l)
    y1p = max(0, y1 - pad_t)

    x2p = min(w, x2 + pad_r)
    y2p = min(h, y2 + pad_b)

    plate_crop = img_bgr[y1p:y2p, x1p:x2p]

    # ─────────────────────────────────────────
    # PARSeq OCR
    # ─────────────────────────────────────────
    raw_text, ocr_conf = recognize_plate_parseq(
        plate_crop
    )

    if not raw_text:

        return jsonify({
            'success': False,
            'error': 'Could not recognize plate text.'
        })

    # ─────────────────────────────────────────
    # CONFIDENCE
    # ─────────────────────────────────────────
    overall_conf = round(
        (yolo_conf * 0.40 + ocr_conf * 0.60) * 100,
        1
    )

    process_time = round(
        time.time() - start_time,
        2
    )

    # ─────────────────────────────────────────
    # BOUNDING BOX
    # ─────────────────────────────────────────
    bbox = {
        'x': round((x1p / w) * 100, 1),
        'y': round((y1p / h) * 100, 1),

        'width': round(
            ((x2p - x1p) / w) * 100,
            1
        ),

        'height': round(
            ((y2p - y1p) / h) * 100,
            1
        ),
    }

    # ─────────────────────────────────────────
    # THUMBNAIL
    # ─────────────────────────────────────────
    thumbnail = None

    try:

        pil_img = Image.fromarray(
            cv2.cvtColor(
                img_bgr,
                cv2.COLOR_BGR2RGB
            )
        )

        pil_img.thumbnail(
            (160, 110),
            Image.LANCZOS
        )

        buf = io.BytesIO()

        pil_img.save(
            buf,
            format='JPEG',
            quality=70
        )

        thumbnail = (
            'data:image/jpeg;base64,' +
            base64.b64encode(
                buf.getvalue()
            ).decode()
        )

    except Exception:
        pass

    # ─────────────────────────────────────────
    # SAVE TO DATABASE
    # ─────────────────────────────────────────
    db_id = save_detection(
        raw_text,
        overall_conf,
        process_time,
        thumbnail
    )

    ts = time.strftime('%Y-%m-%d %H:%M:%S')

    print(
        f"[PlateVision] ✓ {raw_text} | "
        f"conf={overall_conf}% | "
        f"{process_time}s"
    )

    # ─────────────────────────────────────────
    # RESPONSE
    # ─────────────────────────────────────────
    return jsonify({

        'success': True,

        'id': db_id,

        'plateText': raw_text,

        'rawText': raw_text,

        'confidence': overall_conf,

        'processTime': process_time,

        'bbox': bbox,

        'timestamp': ts,

        'thumbnail': thumbnail,
    })


# ─────────────────────────────────────────────────────────────
# HISTORY API
# ─────────────────────────────────────────────────────────────
@app.route('/api/history', methods=['GET'])
def api_get_history():

    return jsonify({
        'success': True,
        'detections': get_all_detections()
    })


@app.route('/api/history/<int:det_id>', methods=['DELETE'])
def api_delete_one(det_id):

    delete_detection(det_id)

    return jsonify({
        'success': True
    })


@app.route('/api/history', methods=['DELETE'])
def api_clear_history():

    clear_all_detections()

    return jsonify({
        'success': True
    })


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    port = int(
        os.environ.get('PORT', 5000)
    )

    print("=" * 55)
    print(" PlateVision AI — PARSeq OCR Edition")
    print(f" Open in browser → http://localhost:{port}")
    print("=" * 55)

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
