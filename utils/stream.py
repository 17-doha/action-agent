# utils/stream.py
import time
import threading
from typing import Optional

_latest_jpeg: Optional[bytes] = None
_lock = threading.Lock()

def update_latest_jpeg(jpeg_bytes: bytes) -> None:
    """Publish newest frame (call this from your capture loop)."""
    global _latest_jpeg
    with _lock:
        _latest_jpeg = jpeg_bytes

def mjpeg_generator():
    """Yield multipart JPEG frames forever (Flask view iterates this)."""
    boundary = b"--frame\r\n"
    while True:
        with _lock:
            frame = _latest_jpeg
        if frame is not None:
            yield (boundary +
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame + b"\r\n")
        time.sleep(0.07)  # ~14 fps pacing
