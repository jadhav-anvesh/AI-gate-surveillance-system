"""
Density analysis service.

Maintains a vehicle density heatmap and per-track position
history, and produces the heatmap overlay used by the
analytics dashboard.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Tuning constants
# ----------------------------------------------------------------------
MAX_TRACK_HISTORY = 1200  # max stored positions per track before trimming oldest
MIN_MOVEMENT_DISTANCE = 5  # pixels a vehicle must move before a heatmap cell increments


class DensityService:
    """Maintains the traffic density heatmap and per-track position history."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self):
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.last_positions: Dict[int, Tuple[float, float]] = {}
        self.heatmap: Optional[np.ndarray] = None

    def initialize_heatmap(self, frame_shape: Tuple[int, ...]) -> None:
        """Allocate a zeroed heatmap buffer matching the given frame shape."""
        self.heatmap = np.zeros(frame_shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_heatmap(self) -> Optional[np.ndarray]:
        """Return the current heatmap buffer, or None if not yet initialized."""
        return self.heatmap

    def get_track_history(self) -> Dict[int, List[Tuple[float, float]]]:
        """Return recorded positions for every tracked vehicle."""
        return self.track_history

    def get_last_positions(self) -> Dict[int, Tuple[float, float]]:
        """Return the most recent known position for every tracked vehicle."""
        return self.last_positions

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """Return the Euclidean distance between two points."""
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    def _extract_boxes(self, results: Any):
        """
        Normalize detector output into (x_center, y_center, width, height) boxes.

        Handles both supervision-style results (`results.xyxy`) and raw
        ultralytics results (`results[0].boxes.xywh`). If more detector
        backends are added later, normalize their box format here too.
        """
        if hasattr(results, "xyxy"):
            boxes = []
            for x1, y1, x2, y2 in results.xyxy:
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2
                boxes.append((x_center, y_center, width, height))
            return boxes

        return results[0].boxes.xywh.cpu()

    # ------------------------------------------------------------------
    # Heatmap Processing
    # ------------------------------------------------------------------

    def update_heatmap(
        self, results: Any, track_ids: List[int], frame: np.ndarray
    ) -> None:
        """Update the cumulative density heatmap using the current tracked detections."""
        boxes = self._extract_boxes(results)

        # Cached once per call instead of re-reading on every loop iteration.
        heatmap_height, heatmap_width = self.heatmap.shape[0], self.heatmap.shape[1]

        for box, track_id in zip(boxes, track_ids):
            x_center, y_center, width, height = box
            current_position = (float(x_center), float(y_center))

            top_left_x = max(0, int(x_center - width / 2))
            top_left_y = max(0, int(y_center - height / 2))
            bottom_right_x = min(heatmap_width, int(x_center + width / 2))
            bottom_right_y = min(heatmap_height, int(y_center + height / 2))

            track = self.track_history[track_id]
            track.append(current_position)
            if len(track) > MAX_TRACK_HISTORY:
                track.pop(0)

            last_position = self.last_positions.get(track_id)
            if (
                last_position
                and self.calculate_distance(last_position, current_position)
                > MIN_MOVEMENT_DISTANCE
            ):
                self.heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

            self.last_positions[track_id] = current_position

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def generate_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Blend the current heatmap onto a frame and return the overlay."""
        if self.heatmap is None:
            # Nothing to overlay yet — return the frame untouched instead
            # of crashing on GaussianBlur(None, ...).
            return frame

        heatmap_blurred = cv2.GaussianBlur(self.heatmap, (15, 15), 0)
        heatmap_norm = cv2.normalize(
            heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        if frame.shape != heatmap_color.shape:
            heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))

        overlay = cv2.addWeighted(frame, 0.3, heatmap_color, 0.7, 0)
        return overlay
