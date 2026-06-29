from typing import Literal
from pydantic import BaseModel, Field


class PipelineStartRequest(BaseModel):
    rtsp_url: str = Field(min_length=1, description="RTSP stream URL")

    camera_type: Literal["Trained Model"]

    det_mode: Literal["YOLO-N", "RF-DETR"]

    frame_rate: int = Field(default=10, ge=1, le=60, description="Processing FPS")

    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    classes_to_detect: list[str] = Field(
        default_factory=list, description="Vehicle classes selected in frontend"
    )
