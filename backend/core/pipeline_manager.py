import supervision as sv
import json
import cv2
import os
import time
from threading import Thread
from datetime import datetime
from backend.database.database import (SessionLocal)
from backend.database.models import (ProcessingSession)
from backend.database.detection_service import (DetectionDatabaseService)

CONFIG_PATH = "current_config.json"

class PipelineManager:

    def __init__(self):

        self.pipeline = None
        self.selected_class_ids = None
        self.video_source = None

        self.processing_thread = None

        self.latest_statistics = {}

        self.is_running = False
        self.latest_density_frame = None

        self.id_cls_map = None
        self.latest_flow_frame = None
        self.latest_processed_frame = None
        self.current_session_id = None
        self.detection_db = (DetectionDatabaseService())
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45

        self.current_config = {}

        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    self.current_config = json.load(f)
            except Exception as e:
                import traceback

                print("\n========== FLOW FRAME ERROR ==========")
                traceback.print_exc()
                print("======================================\n")


    def set_detection_thresholds(
        self,
        confidence_threshold,
        iou_threshold
    ):

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def processing_loop(self):

        while self.is_running:

            if self.video_source is None:

                time.sleep(0.1)

                continue

            if self.pipeline is None:

                time.sleep(0.1)

                continue

            frame_data = self.video_source.get_frame()

            if frame_data is None:

                continue

            frame = frame_data["det_frame"]
            original_frame = frame_data["original_frame"]

            try:
                result = self.pipeline.process_frame(
                    frame=frame,
                    original_frame=original_frame,
                    device="cuda",
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                    class_ids=self.selected_class_ids,
                    id_cls_map=self.id_cls_map,
                    session_id=self.current_session_id
                )
            except Exception as e:

                import traceback

                print("\n========== PROCESS FRAME ERROR ==========")
                traceback.print_exc()
                print("=========================================\n")

                continue
            try:
                if isinstance(result["results"], sv.Detections):

                    detection_frame = frame.copy()

                    box_annotator = sv.BoxAnnotator()
                    label_annotator = sv.LabelAnnotator()

                    labels = []

                    if len(result["results"]) > 0:
                        tracker_ids = result["results"].tracker_id
                        if tracker_ids is None:
                            for class_name, confidence in zip(result["results"].data["class_name"], result["results"].confidence):
                                labels.append(f"{class_name} {confidence:.2f}")

                        else:

                            for tracker_id, class_name, confidence in zip(tracker_ids, result["results"].data["class_name"],result["results"].confidence):
                                labels.append(f"ID:{tracker_id} {class_name} {confidence:.2f}")

                    detection_frame = box_annotator.annotate( scene=detection_frame,detections=result["results"])

                    detection_frame = label_annotator.annotate( scene=detection_frame,detections=result["results"], labels=labels)

                else:
                    detection_frame = result["results"][0].plot()
                self.latest_processed_frame = detection_frame.copy()

                flow_frame = detection_frame.copy()
                pipeline = self.pipeline

                for lane_index, line_zone in enumerate( pipeline.line_zones ):
                    start = (int(line_zone.vector.start.x), int(line_zone.vector.start.y))
                    end = (int(line_zone.vector.end.x), int(line_zone.vector.end.y))
                    cv2.line(flow_frame, start, end, (0,255,0),3)
                    text = (
                        f"Lane {lane_index+1} "
                        f"[In]: {line_zone.out_count} "
                        f"[Out]: {line_zone.in_count}"
                    )
                    mid_x = int((start[0] + end[0]) / 2)
                    mid_y = int((start[1] + end[1]) / 2)

                    (text_w, text_h), _ = cv2.getTextSize(
                        text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        2
                    )   

                    text_x = mid_x - text_w // 2
                    text_y = mid_y - 15
                    cv2.rectangle(
                        flow_frame,
                        (text_x - 5, text_y - text_h - 5),
                        (text_x + text_w + 5, text_y + 5),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        flow_frame,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2
                    )
                self.latest_flow_frame = flow_frame
                if result.get("heatmap_overlay") is not None:
                    self.latest_density_frame = result["heatmap_overlay"]
            except Exception as e:
                logger.exception("Plot error")
            detections = result["detections"]

            for i in range(len(detections)):
                tracker_id = None

                if detections.tracker_id is not None:
                    tracker_id = int(detections.tracker_id[i])

                class_id = None

                if detections.class_id is not None:
                    class_id = int(detections.class_id[i])

                class_name = "Unknown"

                if (class_id is not None and self.id_cls_map is not None):
                    class_name = self.id_cls_map.get(class_id, f"UNKNOWN_{class_id}")

                confidence = None

                try:
                    confidence = float(detections.confidence[i])
                except:
                    pass

                x1, y1, x2, y2 = (detections.xyxy[i])
                center_x = (float(x1) + float(x2)) / 2
                center_y = (float(y1) + float(y2)) / 2

                self.detection_db.save_detection(
                    session_id=self.current_session_id,
                    frame_number=result["frame_index"],
                    track_id=tracker_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    speed=self.pipeline.speed.get_tracker_speed(tracker_id),
                    center_x=center_x,
                    center_y=center_y
                )
                self.detection_db.upsert_vehicle_summary(
                    session_id=self.current_session_id,
                    frame_number=result["frame_index"],
                    track_id=tracker_id,
                    class_name=class_name,
                    speed=self.pipeline.speed.get_tracker_speed(tracker_id),
                    center_x=center_x,
                    center_y=center_y
                )
                if tracker_id in self.pipeline.alpr_results:
                    result_data = (
                        self.pipeline.alpr_results[
                            tracker_id
                        ]
                    )

                    bbox = result_data.get("plate_bbox")

                    if bbox is not None:

                        self.detection_db.update_plate_bbox(
                            session_id=self.current_session_id,
                            track_id=tracker_id,
                            bbox=bbox
                        )

                        del self.pipeline.alpr_results[tracker_id]

            self.latest_statistics = {

                "frame_index":
                    result["frame_index"],

                "detections":
                    len(
                        result["detections"]
                    ),

                "raw_detection_records":
                    self.detection_db.get_raw_detection_count(
                    self.current_session_id
                ),

                "unique_vehicles":
                    self.detection_db.get_unique_vehicle_count(
                    self.current_session_id
                ),

                "flow_statistics":
                    result.get(
                        "flow_statistics",
                        {}
                    ),

                "density_statistics":
                    result.get(
                        "density_statistics",
                        {}
                    ),

                "speed_statistics":
                    result.get(
                        "speed_statistics",
                        {}
                    )
            }

    def load_components(
        self,
        pipeline,
        video_source
    ):

        self.pipeline = pipeline

        self.video_source = video_source

    def start_processing(self):

        if self.is_running:

            return

        self.is_running = True

        self.processing_thread = Thread(
            target=self.processing_loop,
            daemon=True
        )

        self.processing_thread.start()

    def update_statistics(
        self,
        statistics
    ):

        self.latest_statistics = statistics

    def get_statistics(self):

        return self.latest_statistics

    def set_pipeline(
        self,
        pipeline
    ):

        self.pipeline = pipeline

    def get_pipeline(self):

        return self.pipeline

    def set_video_source(
        self,
        video_source
    ):

        self.video_source = video_source

    def get_video_source(self):

        return self.video_source

    def set_class_map(
        self,
        id_cls_map
    ):

        self.id_cls_map = id_cls_map

    def get_class_map(self):

        return self.id_cls_map

    def set_selected_classes(self,class_ids):

        self.selected_class_ids = class_ids

    def stop_processing(self):

        self.is_running = False

        if self.processing_thread:

            self.processing_thread.join(
                timeout=2
            )

    def status(self):

        return {
            "running": self.is_running,
            "pipeline_loaded":
                self.pipeline is not None,

            "video_source_loaded":
                self.video_source is not None,

            "statistics_available":
                len(self.latest_statistics) > 0
        }

    def reset(self):

        self.stop_processing()

        self.pipeline = None

        self.video_source = None

        self.latest_statistics = {}

    def create_session(self, camera_type):
        db = SessionLocal()

        try:

            session = ProcessingSession(camera_type=camera_type, status="running", start_time=datetime.now())
            db.add(session)
            db.commit()
            db.refresh(session)
            self.current_session_id = session.id

        finally:
            db.close()

    def close_session(self):

        if self.current_session_id is None:
            return

        db = SessionLocal()

        try:

            session = db.query(ProcessingSession).filter(ProcessingSession.id == self.current_session_id).first()
            if session:

                session.status = "completed"
                session.end_time = datetime.now()
                db.commit()

        finally:

            db.close()
