import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from backend.services.pipeline import VideoProcessingPipeline
from backend.services.video_source import VideoSource
from backend.config.camera_config import RTSP_STREAM_URL


model_map = {
    "test": (
        "yolo",
        "dependencies/set_b/yolov10n-best.pt"
    )
}

pipeline = VideoProcessingPipeline(
    det_mode="test",
    model_map=model_map,
    frame_rate=10
)

video_source = VideoSource(
    RTSP_STREAM_URL,
    target_fps=5
)

video_source.start()

frame = video_source.get_frame(
    timeout=10
)

if frame is None:

    print(
        "No Frame Received"
    )

else:

    print(
        "Frame Shape:",
        frame.shape
    )

    results, detections = pipeline.detect_and_track(
        frame,
        device="cuda",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        class_ids=None
    )

    print(
        "Detections:",
        len(detections)
    )
    pipeline.initialize_density(
        frame.shape
    )

    heatmap = pipeline.process_density(
        results,
        detections,
        frame
    )

    print(
        "Heatmap Shape:",
        heatmap.shape
    )

video_source.stop()
