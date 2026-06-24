import json
import numpy as np
from fastapi.responses import Response
import cv2
from PIL import Image
import numpy as np
from backend.core.state import manager
from backend.core.pipeline_manager import CONFIG_PATH
from fastapi import APIRouter

from backend.api.schemas.pipeline import (
    PipelineStartRequest
)

from backend.services.video_source import (
    VideoSource
)

from backend.services.pipeline import (
    VideoProcessingPipeline
)

from backend.config.model_config import (
    MODEL_MAPS,
    CLASS_MAPS
)

from backend.api.schemas.config import (
    FlowConfigRequest,
    DensityConfigRequest,
    SpeedConfigRequest
)

router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"]
)

@router.get("/statistics")
def pipeline_statistics():

    stats = manager.get_statistics()
    return stats

@router.get("/status")
def pipeline_status():

    return {
        "pipeline": "ready"
    }

@router.get("/current_config")
def get_current_config():

    return manager.current_config

@router.get("/debug")
def pipeline_debug():

    pipeline = manager.get_pipeline()

    if pipeline is None:

        return {
            "pipeline_loaded": False
        }

    return {

        "pipeline_loaded": True,

        "flow_enabled":
            len(
                pipeline.line_zones
            ) > 0,

        "density_enabled":
            pipeline.density.heatmap
            is not None,

        "speed_enabled":
            pipeline.view_transformer
            is not None
    }

@router.post("/start")
def start_pipeline(request: PipelineStartRequest):
    manager.reset()
    # Save config to memory and disk
    manager.current_config = {
        "camera_type":          request.camera_type,
        "det_mode":             request.det_mode,
        "confidence_threshold": request.confidence_threshold,
        "iou_threshold":        request.iou_threshold,
        "frame_rate":           request.frame_rate,
        "classes_to_detect":    request.classes_to_detect
    }

    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(manager.current_config, f, indent=5)
        logger.info(f"Config saved: {manager.current_config}")
    except Exception as e:
        logger.exception("Config save failed")
    model_map = MODEL_MAPS[
        request.camera_type
    ]

    cls_map = {
        v: k
        for k, v in CLASS_MAPS[request.camera_type].items()
    }
    selected_class_ids = [

        CLASS_MAPS[request.camera_type][class_name]

        for class_name in request.classes_to_detect

        if class_name in CLASS_MAPS[request.camera_type]
    ]

    pipeline = VideoProcessingPipeline(
        det_mode=request.det_mode,
        model_map=model_map,
        frame_rate=request.frame_rate
    )

    video_source = VideoSource(
        source=request.rtsp_url,
        target_fps=request.frame_rate
    )

    manager.set_pipeline(
        pipeline
    )

    manager.set_class_map(
        cls_map
    )

    manager.set_detection_thresholds(
        request.confidence_threshold,
        request.iou_threshold
    )

    manager.create_session(
        request.camera_type
    )

    manager.set_video_source(
        video_source
    )

    video_source.start()
    manager.set_selected_classes(selected_class_ids)
    manager.start_processing()

    return {
        "status": "started",
        "camera_type":
            request.camera_type,
        "det_mode":
            request.det_mode
    }

@router.post("/stop")
def stop_pipeline():

    if manager.video_source is not None:

        manager.video_source.stop()

    manager.stop_processing()

    manager.close_session()

    return {
        "status": "stopped"
    }

@router.post("/config/flow")
def configure_flow(
    request: FlowConfigRequest
):

    pipeline = manager.get_pipeline()

    if pipeline is None:

        return {
            "error": "pipeline not loaded"
        }

    pipeline.initialize_flow_zones(
        request.line_params
    )

    return {
        "status": "flow configured",
        "lines":
            len(
                request.line_params
            )
    }

@router.post("/config/density")
def configure_density(
    request: DensityConfigRequest
):

    pipeline = manager.get_pipeline()

    if pipeline is None:

        return {
            "error": "pipeline not loaded"
        }

    pipeline.initialize_density(
        (
            request.height,
            request.width,
            3
        )
    )

    return {
        "status": "density configured",
        "width": request.width,
        "height": request.height
    }

@router.post("/config/speed")
def configure_speed(
    request: SpeedConfigRequest
):

    pipeline = manager.get_pipeline()

    if pipeline is None:

        return {
            "error": "pipeline not loaded"
        }

    pipeline.initialize_speed_transform(
        source=np.array(
            request.source_points
        ),
        target=np.array(
            request.target_points
        )
    )

    return {
        "status": "speed configured"
    }

@router.get("/debug/full")
def debug_full():

    pipeline = manager.get_pipeline()

    if pipeline is None:
        return {"pipeline": None}

    return {
        "pipeline_id": id(pipeline),
        "density_id": id(pipeline.density),
        "heatmap_exists":
            pipeline.density.heatmap is not None,
        "heatmap_shape":
            None if pipeline.density.heatmap is None
            else pipeline.density.heatmap.shape
    }

def resize_frame_to_360p(frame):

    image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )

    image.thumbnail((640, 360))

    resized_frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    return resized_frame

@router.get("/preview_frame")
def preview_frame():

    rtsp_url = "rtsp://arjun.badola:AB%23ai2025@10.0.102.54:554/"

    cap = cv2.VideoCapture(rtsp_url)

    success, frame = cap.read()
    frame = resize_frame_to_360p(frame)
    cap.release()

    if not success:
        return {
            "error": "Unable to capture frame"
        }

    _, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )
@router.get("/live_frame")
def live_frame():

    frame = manager.latest_processed_frame

    if frame is None:
        return {"error": "No processed frame"}

    _, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )

@router.get("/flow_frame")
def flow_frame():

    frame = manager.latest_flow_frame

    if frame is None:
        return {"error":"No flow frame"}

    _, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )
@router.get("/density_frame")
def density_frame():

    frame = manager.latest_density_frame

    if frame is None:
        return {"error":"No density frame"}

    _, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )
