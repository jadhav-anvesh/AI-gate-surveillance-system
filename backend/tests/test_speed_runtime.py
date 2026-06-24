import numpy as np
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

SOURCE = np.array(
    [
        [0, 0],
        [639, 0],
        [639, 359],
        [0, 359]
    ]
)

TARGET = np.array(
    [
        [0, 0],
        [99, 0],
        [99, 99],
        [0, 99]
    ]
)

pipeline.initialize_speed_transform(
    source=SOURCE,
    target=TARGET
)
video_source.start()

for i in range(15):

    frame = video_source.get_frame(
        timeout=10
    )

    if frame is None:
        continue

    output = pipeline.process_frame(
        frame,
        device="cuda",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        class_ids=None,
        id_cls_map={
            0: "bike",
            1: "auto",
            2: "car",
            3: "truck",
            4: "bus",
            5: "other vehicle"
        }
    )

video_source.stop()

print(
    pipeline.get_speed_statistics()
)

