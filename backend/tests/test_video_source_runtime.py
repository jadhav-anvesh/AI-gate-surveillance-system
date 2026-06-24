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

from backend.services.video_source import VideoSource
from backend.config.camera_config import RTSP_STREAM_URL


video_source = VideoSource(
    RTSP_STREAM_URL,
    target_fps=5
)

video_source.start()

frame = video_source.get_frame(
    timeout=10
)

if frame is not None:

    print(
        "Frame Received"
    )

    print(
        frame.shape
    )

else:

    print(
        "No Frame Received"
    )

video_source.stop()
