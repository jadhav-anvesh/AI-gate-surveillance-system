"""
Object detection service.

Provides a unified interface for running inference using
YOLO or RF-DETR models while hiding model-specific
implementation details from the rest of the pipeline.
"""

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
from rfdetr import RFDETRNano
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Model type identifiers (avoids typos in string comparisons below)
# ----------------------------------------------------------------------
MODEL_RFDETR = "rfdetr"


class DetectorService:
    """
    Wrapper around the supported object detection models.

    Supported models:
        - YOLO
        - RF-DETR
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, det_mode: str, model_map: Mapping[str, Tuple[str, str]]) -> None:
        model_type, model_path = model_map[det_mode]
        self.model_type = model_type

        if model_type == MODEL_RFDETR:
            self.model = RFDETRNano.from_checkpoint(model_path)
            # Optional optimization. Disabled because it caused
            # compatibility issues during development.
            # self.model.optimize_for_inference()
        else:
            self.model = YOLO(model_path)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def track(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float,
        device: str,
        classes: Optional[List[int]],
        max_det: int = 1000,
    ) -> Any:
        """
        Run inference on a frame.

        RF-DETR performs detection only. YOLO perform
        detection with persistent tracking via Ultralytics' `.track()`.
        """
        if self.model_type == MODEL_RFDETR:
            detections = self.model.predict(
                frame,
                threshold=conf,
                include_source_image=False,
            )
            if classes is not None:
                detections = detections[np.isin(detections.class_id, classes)]
            return detections

        # YOLO provides persistent tracking.
        return self.model.track(
            frame,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
            max_det=max_det,
            persist=True,
            verbose=False,
        )
