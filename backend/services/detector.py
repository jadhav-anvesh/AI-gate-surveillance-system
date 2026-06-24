from ultralytics import YOLO, RTDETR
from rfdetr import RFDETRNano
import numpy as np

class DetectorService:

    def __init__(self, det_mode, model_map):

        model_type = model_map[det_mode][0]
        model_path = model_map[det_mode][1]

        self.model_type = model_type

        if model_type == "rfdetr":
            self.model = RFDETRNano.from_checkpoint(model_path)
           #remove tempory
           # self.model.optimize_for_inference()

        elif model_type == "detr":
            self.model = RTDETR(model_path)

        else:
            self.model = YOLO(model_path)

    def track(
        self,
        frame,
        conf,
        iou,
        device,
        classes,
        max_det=1000
    ):

        if self.model_type == "rfdetr":
            detections = self.model.predict(
                frame,
                threshold=conf,
                include_source_image=False
            )
            if classes is not None:

                detections = detections[
                    np.isin(
                        detections.class_id,
                        classes
                    )
                ]
            return detections
        return self.model.track(
            frame,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
            max_det=max_det,
            persist=True,
            verbose=False
        )
