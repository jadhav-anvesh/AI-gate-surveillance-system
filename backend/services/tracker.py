"""
Object tracking service.

Thin wrapper around supervision's ByteTrack, assigning
persistent tracker IDs to detections across frames.
"""

import supervision as sv


class TrackingService:
    """Assigns and maintains persistent tracker IDs for detections."""

    def __init__(self) -> None:
        self.tracker = sv.ByteTrack()
        # Defensive reset in case the underlying tracker carries any
        # state over from a previous instantiation.
        self.tracker.reset()

    def track(self, detections: sv.Detections) -> sv.Detections:
        """Assign tracker IDs to the given frame's detections."""
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Clear all tracking history (e.g. when starting a new session)."""
        self.tracker.reset()
