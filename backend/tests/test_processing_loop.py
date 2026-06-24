import sys
import os
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from backend.core.pipeline_manager import PipelineManager
from backend.services.video_source import VideoSource
from backend.services.pipeline import VideoProcessingPipeline

from backend.config.camera_config import RTSP_STREAM_URL
from backend.config.model_config import MODEL_MAP_SET_B


manager = PipelineManager()

video_source = VideoSource(
    RTSP_STREAM_URL
)

pipeline = VideoProcessingPipeline(
    det_mode="Light (Faster)",
    model_map=MODEL_MAP_SET_B,
    frame_rate=10
)

manager.load_components(
    pipeline,
    video_source
)

video_source.start()

manager.start_processing()

time.sleep(3)

print(
    manager.get_statistics()
)

manager.stop_processing()

video_source.stop()

print(
    "Processing Loop Test Successful"
)
