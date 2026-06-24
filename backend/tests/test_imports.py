import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.services.video_source import frame_grabber
from backend.services.detector import load_model

print("Imports Successful")
