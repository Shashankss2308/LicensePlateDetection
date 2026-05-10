"""
WSGI entry point for Gunicorn / waitress.
Usage (Gunicorn):   gunicorn wsgi:app -w 1 --timeout 120
Usage (waitress):   waitress-serve --port=5000 wsgi:app
"""
from app import app  # noqa: F401

if __name__ == '__main__':
    app.run()
