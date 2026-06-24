import supervision as sv


class TrackingService:

    def __init__(self):

        # Create ByteTrack object
        self.tracker = sv.ByteTrack()

        # Clear tracker state
        self.tracker.reset()

    def track(self, detections):

        # Assign tracker IDs
        return self.tracker.update_with_detections(
            detections
        )

    def reset(self):

        # Reset tracking history
        self.tracker.reset()
