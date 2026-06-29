"""
Flow analysis service.

Tracks overall and per-lane vehicle counts, plus time-series
"vehicles in scene" data used by the analytics dashboard.
"""

from typing import Any, Dict, Iterable, List, Tuple


class FlowService:
    """Tracks vehicle counts and per-lane crossing statistics."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Total vehicle counts, keyed by class name.
        self.vehicle_distribution_map: Dict[str, int] = {}
        # Vehicles seen in scene, keyed by second.
        self.vehicle_in_scene_map: Dict[int, int] = {}
        # Per-lane vehicle counts, one dict per lane.
        self.lane_wise_vehicle_distribution_maps: List[Dict[str, int]] = []
        # Per-lane time series, one dict per lane.
        self.lane_wise_vehicle_in_scene_maps: List[Dict[int, int]] = []

    def initialize_lanes(self, num_lanes: int) -> None:
        """Allocate empty per-lane count/time-series maps."""
        self.lane_wise_vehicle_distribution_maps = [{} for _ in range(num_lanes)]
        self.lane_wise_vehicle_in_scene_maps = [{} for _ in range(num_lanes)]

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def update_vehicle_count(self, vehicle_class: str, lane_index: int) -> None:
        """Increment the total and per-lane count for a vehicle class."""
        self.vehicle_distribution_map[vehicle_class] = (
            self.vehicle_distribution_map.get(vehicle_class, 0) + 1
        )

        lane_map = self.lane_wise_vehicle_distribution_maps[lane_index]
        lane_map[vehicle_class] = lane_map.get(vehicle_class, 0) + 1

    def update_vehicle_scene_count(
        self, current_second: int, vehicle_count: int
    ) -> None:
        """Record the total number of vehicles in scene for a given second."""
        self.vehicle_in_scene_map[current_second] = vehicle_count

    def update_lane_scene_count(
        self,
        lane_index: int,
        current_second: int,
        current_total_count: int,
    ) -> None:
        """Record the per-second crossing delta for a lane."""
        lane_map = self.lane_wise_vehicle_in_scene_maps[lane_index]
        previous_second = current_second - 1

        if previous_second in lane_map:
            lane_map[current_second] = current_total_count - lane_map[previous_second]
        else:
            lane_map[current_second] = current_total_count

    # ------------------------------------------------------------------
    # Trigger Processing
    # ------------------------------------------------------------------

    def process_trigger(
        self,
        trigger: Tuple[Any, Any],
        class_ids: Iterable[int],
        id_cls_map: Dict[int, str],
        lane_index: int,
    ) -> None:
        """Update counts for every detection that crossed a line zone this frame."""
        crossed_in, crossed_out = trigger

        for is_in, is_out, cls_id in zip(crossed_in, crossed_out, class_ids):
            if is_in or is_out:
                # Falls back instead of raising if cls_id is missing from
                # the map, so an unmapped class can't crash the pipeline.
                vehicle_class = id_cls_map.get(cls_id, f"UNKNOWN_{cls_id}").upper()
                self.update_vehicle_count(vehicle_class, lane_index)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_vehicle_distribution(self) -> Dict[str, int]:
        """Return total vehicle counts by class."""
        return self.vehicle_distribution_map

    def get_vehicle_scene_data(self) -> Dict[int, int]:
        """Return vehicles-in-scene counts keyed by second."""
        return self.vehicle_in_scene_map

    def get_lane_distribution(self) -> List[Dict[str, int]]:
        """Return per-lane vehicle counts by class."""
        return self.lane_wise_vehicle_distribution_maps

    def get_lane_scene_data(self) -> List[Dict[int, int]]:
        """Return per-lane vehicles-in-scene time series."""
        return self.lane_wise_vehicle_in_scene_maps
