import sys
import os
import cv2

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from backend.config.camera_config import (
    RTSP_STREAM_URL
)

cap = cv2.VideoCapture(
    RTSP_STREAM_URL
)

print(
    "Opened:",
    cap.isOpened()
)

ret, frame = cap.read()

print(
    "Read:",
    ret
)

if ret:

    print(
        "Shape:",
        frame.shape
    )

cap.release()
