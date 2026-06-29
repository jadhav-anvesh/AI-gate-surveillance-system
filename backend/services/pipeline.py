"""
Video processing pipeline.

Coordinates detection, tracking, vehicle flow, density estimation,
speed estimation, and ALPR (automatic license plate recognition).

Acts as the central service used by PipelineManager.
"""

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from utils import ViewTransformer
from backend.services.detector import DetectorService
from backend.services.tracker import TrackingService
from backend.services.flow import FlowService
from backend.services.density import DensityService
from backend.services.speed import SpeedService

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Tuning constants (previously inline "magic numbers")
# ----------------------------------------------------------------------
MAX_DETECTIONS = 1000  # max detections per frame, passed to detector.track()
ALPR_CROP_DIR = "temp/alpr_frames"  # where high-res ALPR crops are written
MAX_ALPR_FRAMES = 5  # how many crops to capture per tracked vehicle
ALPR_PADDING = 20  # pixels of padding added around each crop


class VideoProcessingPipeline:
    """Central pipeline that orchestrates detection, tracking, and analytics."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, det_mode: str, model_map: Mapping[str, Any], frame_rate: int):
        self.alpr_results: Dict[int, Any] = {}

        self.detector = DetectorService(det_mode, model_map)
        self.tracker = TrackingService()
        self.flow = FlowService()
        self.density = DensityService()
        self.speed = SpeedService(frame_rate)

        self.frame_rate = frame_rate
        self.image_index = 0
        # Tracks how many ALPR crops have been captured per session+tracker.
        # Only the count matters, so this is an int, not a list.
        self.seen_tracker_ids: Dict[str, int] = {}
        self.line_zones: List[sv.LineZone] = []
        self.view_transformer: Optional[ViewTransformer] = None

    # ------------------------------------------------------------------
    # Detection & Tracking
    # ------------------------------------------------------------------

    def detect_and_track(
        self,
        frame: np.ndarray,
        device: str,
        confidence_threshold: float,
        iou_threshold: float,
        class_ids: Optional[List[int]],
    ) -> Tuple[Any, sv.Detections]:
        """Run detection followed by multi-object tracking on a single frame."""
        results = self.detector.track(
            frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=device,
            classes=class_ids,
            max_det=MAX_DETECTIONS,
        )

        if isinstance(results, sv.Detections):
            detections = results
        else:
            detections = sv.Detections.from_ultralytics(results[0])

        detections = self.tracker.track(detections)

        return results, detections

    # ------------------------------------------------------------------
    # Flow Analysis
    # ------------------------------------------------------------------

    def initialize_flow_zones(
        self, line_params: List[Tuple[float, float, float, float]]
    ) -> None:
        """Build LineZones from raw (x1, y1, x2, y2) line coordinates."""
        self.line_zones = []

        for line in line_params:
            start_pt = sv.Point(line[0], line[1])
            end_pt = sv.Point(line[2], line[3])

            line_zone = sv.LineZone(
                start_pt,
                end_pt,
                triggering_anchors=[sv.Position.CENTER],
            )
            self.line_zones.append(line_zone)

        self.flow.initialize_lanes(len(self.line_zones))

    def process_flow_zones(
        self, detections: sv.Detections, id_cls_map: Dict[int, str]
    ) -> None:
        """Trigger each line zone for the current detections and update flow stats."""
        for lane_index, line_zone in enumerate(self.line_zones):
            trigger = line_zone.trigger(detections=detections)

            self.flow.process_trigger(
                trigger,
                detections.class_id,
                id_cls_map,
                lane_index,
            )

            if self.image_index % self.frame_rate == 0:
                current_sec = self.image_index // self.frame_rate
                in_out_count = line_zone.in_count + line_zone.out_count

                self.flow.update_lane_scene_count(
                    lane_index,
                    current_sec,
                    in_out_count,
                )

    def update_scene_count(self, detections: sv.Detections) -> None:
        """Record the vehicle count for the current second, once per second."""
        if self.image_index % self.frame_rate == 0:
            current_sec = self.image_index // self.frame_rate
            self.flow.update_vehicle_scene_count(current_sec, len(detections.class_id))

    def get_flow_statistics(self) -> Dict[str, Any]:
        """Return aggregated vehicle/lane flow statistics."""
        return {
            "vehicle_distribution": self.flow.get_vehicle_distribution(),
            "vehicle_scene": self.flow.get_vehicle_scene_data(),
            "lane_distribution": self.flow.get_lane_distribution(),
            "lane_scene": self.flow.get_lane_scene_data(),
        }

    # ------------------------------------------------------------------
    # Density Analysis
    # ------------------------------------------------------------------

    def initialize_density(self, frame_shape: Tuple[int, ...]) -> None:
        """Allocate the heatmap buffer for the given frame shape."""
        self.density.initialize_heatmap(frame_shape)

    def process_density(
        self,
        results: Any,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Update the density heatmap and return a frame overlay."""
        # tracker_id can be None (e.g. no tracks confirmed yet) — guard
        # instead of letting list() raise on a None iterable.
        track_ids = [] if detections.tracker_id is None else list(detections.tracker_id)

        self.density.update_heatmap(results, track_ids, frame)
        overlay = self.density.generate_overlay(frame.copy())

        return overlay

    def get_density_statistics(self) -> Dict[str, Any]:
        """Return track history, last positions, and heatmap metadata."""
        heatmap = self.density.get_heatmap()

        track_history = {str(k): v for k, v in self.density.get_track_history().items()}
        last_positions = {
            str(k): v for k, v in self.density.get_last_positions().items()
        }

        return {
            "track_history": track_history,
            "last_positions": last_positions,
            "heatmap_available": heatmap is not None,
            "heatmap_shape": None if heatmap is None else list(heatmap.shape),
        }

    # ------------------------------------------------------------------
    # Speed Analysis
    # ------------------------------------------------------------------

    def initialize_speed_transform(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """Set up the perspective transform used to estimate real-world speed."""
        self.view_transformer = ViewTransformer(source=source, target=target)

    def process_speed(
        self, detections: sv.Detections, id_cls_map: Dict[int, str]
    ) -> List[str]:
        """Estimate per-vehicle speed and return display labels."""
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = self.view_transformer.transform_points(points=points).astype(int)

        total_speed = 0
        total_count = 0
        labels: List[str] = []

        for tracker_id, point in zip(detections.tracker_id, points):
            _, y = point
            self.speed.update_coordinates(tracker_id, y)

        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            speed = self.speed.calculate_speed(tracker_id)

            if speed is None:
                labels.append("Analyzing...")
                continue

            labels.append(f"{int(speed)} km/h")

            class_name = id_cls_map.get(int(class_id), f"UNKNOWN_{class_id}").upper()
            self.speed.update_class_speed(class_name, speed)

            total_speed += speed
            total_count += 1

        if self.image_index % self.frame_rate == 0:
            current_sec = self.image_index // self.frame_rate
            self.speed.update_time_speed(current_sec, total_speed, total_count)

        return labels

    def get_speed_statistics(self) -> Dict[str, Any]:
        """Return per-class and per-second speed statistics."""
        return {
            "class_speed": self.speed.get_class_speed_data(),
            "time_speed": self.speed.get_time_speed_data(),
        }

    # ------------------------------------------------------------------
    # ALPR (License Plate Capture)
    # ------------------------------------------------------------------

    def _extract_scaled_crop(
        self,
        xyxy: np.ndarray,
        original_frame: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        """Scale a box from inference resolution to the original frame, then pad-crop it."""
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        x1 = max(0, x1 - ALPR_PADDING)
        y1 = max(0, y1 - ALPR_PADDING)
        x2 = min(original_frame.shape[1], x2 + ALPR_PADDING)
        y2 = min(original_frame.shape[0], y2 + ALPR_PADDING)

        return original_frame[y1:y2, x1:x2]

    def _run_alpr_lookup(self, tracker_id: int, session_id: Optional[str]) -> None:
        """Run ALPR and cache the result for PipelineManager."""
        from alpr_api import get_best_plate

        result = get_best_plate(tracker_id, session_id=session_id)
        if result is not None:
            self.alpr_results[tracker_id] = result
        logger.info(
            "ALPR result | session=%s tracker=%s result=%s",
            session_id,
            tracker_id,
            result,
        )

    def _process_alpr(
        self,
        frame: np.ndarray,
        original_frame: np.ndarray,
        detections: sv.Detections,
        session_id: Optional[str],
    ) -> None:
        """
        Capture up to MAX_ALPR_FRAMES high-res crops per tracked vehicle
        and trigger the ALPR lookup once enough crops are collected.
        """
        if detections.tracker_id is None:
            return

        os.makedirs(ALPR_CROP_DIR, exist_ok=True)

        scale_x = original_frame.shape[1] / frame.shape[1]
        scale_y = original_frame.shape[0] / frame.shape[0]

        for i, tracker_id in enumerate(detections.tracker_id):
            key = f"{session_id}_{tracker_id}"
            captured = self.seen_tracker_ids.get(key, 0)

            if captured >= MAX_ALPR_FRAMES:
                continue

            crop = self._extract_scaled_crop(
                detections.xyxy[i], original_frame, scale_x, scale_y
            )
            if crop.size == 0:
                continue

            frame_idx = captured + 1
            self.seen_tracker_ids[key] = frame_idx

            crop_path = f"{ALPR_CROP_DIR}/session_{session_id}_tracker_{tracker_id}_frame{frame_idx}.jpg"
            cv2.imwrite(crop_path, crop)

            logger.info(
                "ALPR crop saved | session=%s tracker=%s frame=%s",
                session_id,
                tracker_id,
                frame_idx,
            )

            if frame_idx == MAX_ALPR_FRAMES:
                self._run_alpr_lookup(tracker_id, session_id)

    # ------------------------------------------------------------------
    # Frame Processing
    # ------------------------------------------------------------------

    def _process_analytics(
        self,
        frame: np.ndarray,
        results: Any,
        detections: sv.Detections,
        id_cls_map: Dict[int, str],
    ) -> Dict[str, Any]:
        """Run flow, speed, and density analytics for the current frame."""
        if len(self.line_zones) > 0:
            self.process_flow_zones(detections, id_cls_map)

        self.update_scene_count(detections)

        speed_labels: List[str] = []
        if self.view_transformer is not None:
            speed_labels = self.process_speed(detections, id_cls_map)

        heatmap_overlay = None
        if self.density.heatmap is not None:
            heatmap_overlay = self.process_density(results, detections, frame)

        return {
            "speed_labels": speed_labels,
            "heatmap_overlay": heatmap_overlay,
            "flow_statistics": self.get_flow_statistics(),
            "density_statistics": self.get_density_statistics(),
            "speed_statistics": self.get_speed_statistics(),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        original_frame: np.ndarray,
        device: str,
        confidence_threshold: float,
        iou_threshold: float,
        class_ids: Optional[List[int]],
        id_cls_map: Optional[Dict[int, str]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run one full pipeline pass over a frame: detection, tracking,
        ALPR capture, and (if id_cls_map is provided) flow/speed/density
        analytics. Returns a dict consumed by the caller/UI layer.
        """
        self.image_index += 1

        results, detections = self.detect_and_track(
            frame, device, confidence_threshold, iou_threshold, class_ids
        )

        self._process_alpr(frame, original_frame, detections, session_id)

        if id_cls_map is not None:
            analytics = self._process_analytics(frame, results, detections, id_cls_map)
            return {
                "results": results,
                "detections": detections,
                "frame_index": self.image_index,
                **analytics,
            }

        return {
            "results": results,
            "detections": detections,
            "frame_index": self.image_index,
        }
