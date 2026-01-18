"""
WSGI entry point for Render deployment
This file is for WSGI compatibility
Render will use: gunicorn app:app
The app is defined in app.py
"""
from app import app

if __name__ == "__main__":
    app.run()
