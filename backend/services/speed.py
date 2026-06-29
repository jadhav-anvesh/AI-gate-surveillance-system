"""
Speed estimation service.

Tracks per-vehicle vertical position history (in transformed,
real-world coordinates) and derives speed in km/h, plus
per-class and per-second speed aggregates for the dashboard.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# Tuning constants
# ----------------------------------------------------------------------
MS_TO_KMH = 3.6  # conversion factor: meters/second -> km/h
MIN_SAMPLES_RATIO = (
    0.5  # need at least (frame_rate * this) samples before estimating speed
)


class SpeedService:
    """Estimates per-vehicle speed and aggregates it by class and by second."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, frame_rate: int) -> None:
        self.frame_rate = frame_rate
        # Rolling window of y-coordinates per tracker, one slot per frame.
        self.coordinates_for_speed = defaultdict(lambda: deque(maxlen=frame_rate))
        # class_name -> [sum_of_speeds, sample_count]
        self.vehicle_speed_class_map: Dict[str, List[float]] = {}
        # second -> average speed across all vehicles that second
        self.vehicle_speed_time_map: Dict[int, float] = {}
        # tracker_id -> most recently calculated speed
        self.current_tracker_speed: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Coordinate Tracking
    # ------------------------------------------------------------------

    def update_coordinates(self, tracker_id: int, y_coordinate: float) -> None:
        """Record the latest transformed y-coordinate for a tracked vehicle."""
        self.coordinates_for_speed[tracker_id].append(y_coordinate)

    # ------------------------------------------------------------------
    # Speed Calculation
    # ------------------------------------------------------------------

    def calculate_speed(self, tracker_id: int) -> Optional[float]:
        """
        Estimate a vehicle's speed (km/h) from its recent y-coordinate
        history. Returns None until enough samples have been collected.
        """
        history = self.coordinates_for_speed[tracker_id]

        if len(history) < self.frame_rate * MIN_SAMPLES_RATIO:
            return None

        coordinate_newest = history[-1]
        coordinate_oldest = history[0]
        distance = abs(coordinate_newest - coordinate_oldest)

        # Time spanned by the buffer, assuming one sample per frame.
        elapsed_seconds = len(history) / self.frame_rate

        speed = distance / elapsed_seconds * MS_TO_KMH
        self.current_tracker_speed[tracker_id] = speed
        return speed

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def update_class_speed(self, class_name: str, speed: float) -> None:
        """Accumulate speed sum/count for a vehicle class (true running average)."""
        if class_name not in self.vehicle_speed_class_map:
            self.vehicle_speed_class_map[class_name] = [speed, 1]
        else:
            self.vehicle_speed_class_map[class_name][0] += speed
            self.vehicle_speed_class_map[class_name][1] += 1

    def update_time_speed(
        self, current_second: int, total_speed: float, total_count: int
    ) -> None:
        """Record the average speed across all vehicles for a given second."""
        if total_count > 0:
            self.vehicle_speed_time_map[current_second] = total_speed / total_count
        else:
            self.vehicle_speed_time_map[current_second] = 0

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_class_speed_data(self) -> Dict[str, List[float]]:
        """Return {class_name: [sum_of_speeds, sample_count]} for every class seen."""
        return self.vehicle_speed_class_map

    def get_time_speed_data(self) -> Dict[int, float]:
        """Return average speed keyed by second."""
        return self.vehicle_speed_time_map

    def get_tracker_speed(self, tracker_id: int) -> Optional[float]:
        """Return the most recently calculated speed for a tracker, if any."""
        return self.current_tracker_speed.get(tracker_id)
