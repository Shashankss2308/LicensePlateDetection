"""
PlateVision AI — SQLite History Database
"""
import sqlite3, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'platevision.db')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text   TEXT    NOT NULL,
            confidence   REAL    DEFAULT 0,
            process_time REAL    DEFAULT 0,
            timestamp    TEXT    NOT NULL,
            thumbnail    TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_detection(plate_text, confidence, process_time, thumbnail=None):
    conn = get_db()
    ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur = conn.execute(
        'INSERT INTO detections (plate_text, confidence, process_time, timestamp, thumbnail) '
        'VALUES (?, ?, ?, ?, ?)',
        (plate_text, confidence, process_time, ts, thumbnail)
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_all_detections():
    conn = get_db()
    rows = conn.execute('SELECT * FROM detections ORDER BY id DESC').fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_detection(det_id):
    conn = get_db()
    conn.execute('DELETE FROM detections WHERE id = ?', (det_id,))
    conn.commit()
    conn.close()


def clear_all_detections():
    conn = get_db()
    conn.execute('DELETE FROM detections')
    conn.commit()
    conn.close()
