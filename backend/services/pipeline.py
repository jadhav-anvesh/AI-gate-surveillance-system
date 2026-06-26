import supervision as sv
import cv2
import os

from utils import ViewTransformer
from backend.services.detector import DetectorService
from backend.services.tracker import TrackingService
from backend.services.flow import FlowService
from backend.services.density import DensityService
from backend.services.speed import SpeedService


class VideoProcessingPipeline:

    def __init__(
        self,
        det_mode,
        model_map,
        frame_rate
    ):
        self.alpr_results = {}
        self.detector = DetectorService(
            det_mode,
            model_map
        )

        self.tracker = TrackingService()

        self.flow = FlowService()

        self.density = DensityService()

        self.speed = SpeedService(
            frame_rate
        )

        self.frame_rate = frame_rate

        self.image_index = 0

        self.seen_tracker_ids = {}  #new

        self.line_zones = []

        self.view_transformer = None

    def detect_and_track(
        self,
        frame,
        device,
        confidence_threshold,
        iou_threshold,
        class_ids
    ):

        results = self.detector.track(
            frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=device,
            classes=class_ids,
            max_det=1000
        )

        if isinstance(results, sv.Detections):
            detections = results
        else:
            if isinstance(results, sv.Detections):
                detections = results
            else:
                detections = sv.Detections.from_ultralytics( results[0])
        detections = self.tracker.track(detections)

        return results, detections
    def initialize_flow_zones(
        self,
        line_params
    ):

        self.line_zones = []

        for line in line_params:
            start_pt = sv.Point(
                line[0],
                line[1]
            )

            end_pt = sv.Point(
                line[2],
                line[3]
            )

            line_zone = sv.LineZone(
                start_pt,
                end_pt,
                triggering_anchors=[
                    sv.Position.CENTER
                ]
            )

            self.line_zones.append(
                line_zone
            )

        self.flow.initialize_lanes(
            len(self.line_zones)
        )

    def process_flow_zones(
        self,
        detections,
        id_cls_map
    ):

        for lane_index, line_zone in enumerate(
            self.line_zones
        ):

            trigger = line_zone.trigger(
                detections=detections
            )

            self.flow.process_trigger(
                trigger,
                detections.class_id,
                id_cls_map,
                lane_index
            )

            if self.image_index % self.frame_rate == 0:

                current_sec = (
                    self.image_index
                    // self.frame_rate
                )

                in_out_count = (
                    line_zone.in_count
                    + line_zone.out_count
                )

                self.flow.update_lane_scene_count(
                    lane_index,
                    current_sec,
                    in_out_count
                )

    def update_scene_count(
        self,
        detections
    ):

        if self.image_index % self.frame_rate == 0:

            current_sec = (
                self.image_index
                // self.frame_rate
            )

            self.flow.update_vehicle_scene_count(
                current_sec,
                len(detections.class_id)
            )

    def get_flow_statistics(self):

        return {
            "vehicle_distribution":
                self.flow.get_vehicle_distribution(),

            "vehicle_scene":
                self.flow.get_vehicle_scene_data(),

            "lane_distribution":
                self.flow.get_lane_distribution(),

            "lane_scene":
                self.flow.get_lane_scene_data()
        }

    def initialize_density(
        self,
        frame_shape
    ):

        self.density.initialize_heatmap(
            frame_shape
        )

    def process_density(
        self,
        results,
        detections,
        frame
    ):

        track_ids = [
            tid
            for tid in detections.tracker_id
        ]

        self.density.update_heatmap(
            results,
            track_ids,
            frame
        )

        overlay = self.density.generate_overlay(
            frame.copy()
        )

        return overlay

    def get_density_statistics(self):

        heatmap = self.density.get_heatmap()

        track_history = {
            str(k): v
            for k, v in self.density.get_track_history().items()
        }

        last_positions = {
            str(k): v
            for k, v in self.density.get_last_positions().items()
        }

        return {
            "track_history":
                track_history,

            "last_positions":
                last_positions,

            "heatmap_available":
                heatmap is not None,

            "heatmap_shape":
                None if heatmap is None
                else list(heatmap.shape)
        }

    def initialize_speed_transform(
        self,
        source,
        target
    ):

        self.view_transformer = ViewTransformer(
            source=source,
            target=target
        )

    def process_speed(
        self,
        detections,
        id_cls_map
    ):
        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        points = self.view_transformer.transform_points(
            points=points
        ).astype(int)
        total_speed = 0
        total_count = 0

        labels = []

        for tracker_id, point in zip(
            detections.tracker_id,
            points
        ):
            _,y = point
            self.speed.update_coordinates(
                tracker_id,
                y
            )

        for tracker_id, class_id in zip(
            detections.tracker_id,
            detections.class_id
        ):

            speed = self.speed.calculate_speed(
                tracker_id
            )


            speed = self.speed.calculate_speed(
                tracker_id
            )

            if speed is None:

                labels.append(
                    "Analyzing..."
                )

                continue

            labels.append(
                f"{int(speed)} km/h"
            )

            class_name = id_cls_map.get(
                int(class_id),
                f"UNKNOWN_{class_id}"
            ).upper()

            self.speed.update_class_speed(
                class_name,
                speed
            )

            total_speed += speed
            total_count += 1

        if self.image_index % self.frame_rate == 0:

            current_sec = (
                self.image_index
                // self.frame_rate
            )

            self.speed.update_time_speed(
                current_sec,
                total_speed,
                total_count
            )

        return labels

    def get_speed_statistics(self):

        return {
            "class_speed":
                self.speed.get_class_speed_data(),
            "time_speed":
                self.speed.get_time_speed_data()
        }

    def process_frame(
        self,
        frame,
        original_frame,
        device,
        confidence_threshold,
        iou_threshold,
        class_ids,
        id_cls_map=None,
        session_id=None
    ):
        self.image_index += 1

        results, detections = self.detect_and_track(
            frame,
            device,
            confidence_threshold,
            iou_threshold,
            class_ids
        )

        # ALPR: save vehicle-box crop per unique tracker_id
        os.makedirs("temp/alpr_frames", exist_ok=True)
        if detections.tracker_id is not None:
            scale_x = original_frame.shape[1] / frame.shape[1]
            scale_y = original_frame.shape[0] / frame.shape[0]

            for i, tracker_id in enumerate(detections.tracker_id):
                key = f"{session_id}_{tracker_id}"

                if key not in self.seen_tracker_ids:
                    self.seen_tracker_ids[key] = []

                if len(self.seen_tracker_ids[key]) < 5:
                    x1, y1, x2, y2 = detections.xyxy[i]

                    # Scale coordinates from 360p space to original frame space
                    x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    pad = 20
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(original_frame.shape[1], x2 + pad), min(original_frame.shape[0], y2 + pad)
                    crop = original_frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        self.seen_tracker_ids[key].append(True)
                        frame_idx = len(self.seen_tracker_ids[key])
                        cv2.imwrite(f"temp/alpr_frames/session_{session_id}_tracker_{tracker_id}_frame{frame_idx}.jpg", crop)
                        print(f"[ALPR] Saved High_RES crop frame {frame_idx} for session_id={session_id} tracker_id={tracker_id}")

                if len(self.seen_tracker_ids[key]) == 5:
                    from alpr_api import get_best_plate
                    plate = get_best_plate(tracker_id, session_id=session_id)
                    print(f"[ALPR RESULT] session_id={session_id} tracker_id={tracker_id} → {plate}")  #new

        if id_cls_map is not None:

            if len(self.line_zones) > 0:

                self.process_flow_zones(
                    detections,
                    id_cls_map
                )

            self.update_scene_count(
                detections
            )

            speed_labels = []

            if self.view_transformer is not None:
 
                speed_labels = self.process_speed(
                    detections,
                    id_cls_map
                )

            heatmap_overlay = None

            if self.density.heatmap is not None:
 
                heatmap_overlay = self.process_density(
                    results,
                    detections,
                    frame
                )

            return {
                "results": results,
                "detections": detections,
                "speed_labels": speed_labels,
                "heatmap_overlay": heatmap_overlay,
                "flow_statistics": self.get_flow_statistics(),
                "density_statistics": self.get_density_statistics(),
                "speed_statistics": self.get_speed_statistics(),
                "frame_index": self.image_index
            }

        return {
            "results": results,
            "detections": detections,
            "frame_index": self.image_index
        }

