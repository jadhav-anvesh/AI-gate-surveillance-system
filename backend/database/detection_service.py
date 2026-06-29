"""
Database service layer.

Provides helper methods for storing and retrieving detections,
vehicle summaries, and related metadata used by the Gate
Surveillance pipeline.

"""

import json
import logging
from datetime import datetime
from typing import Optional, Sequence

from backend.database.database import SessionLocal
from backend.database.models import DetectionRecord, VehicleSummary

logger = logging.getLogger(__name__)


class DetectionDatabaseService:
    """Handles DB access for detections, vehicle summaries, and stats."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_vehicle(db, session_id, track_id) -> Optional[VehicleSummary]:
        """Fetch the VehicleSummary row for a session/track pair."""
        return (
            db.query(VehicleSummary)
            .filter(
                VehicleSummary.session_id == session_id,
                VehicleSummary.track_id == track_id,
            )
            .first()
        )

    # ------------------------------------------------------------------
    # Detection Operations
    # ------------------------------------------------------------------

    def save_detection(
        self,
        session_id: int,
        frame_number: int,
        track_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        speed: Optional[float],
        center_x: float,
        center_y: float,
    ) -> None:
        """Save a single raw detection produced by the pipeline."""
        db = SessionLocal()
        try:
            record = DetectionRecord(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                frame_number=frame_number,
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                speed=speed,
                center_x=center_x,
                center_y=center_y,
            )
            db.add(record)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Failed to save detection (session_id=%s, track_id=%s, frame=%s)",
                session_id,
                track_id,
                frame_number,
            )
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Vehicle Summary Operations
    # ------------------------------------------------------------------

    def upsert_vehicle_summary(
        self,
        session_id: int,
        frame_number: int,
        track_id: int,
        class_name: str,
        speed: Optional[float],
        center_x: float,
        center_y: float,
    ) -> None:
        """Create or update the rolling summary for a tracked vehicle."""
        db = SessionLocal()
        try:
            vehicle = self._get_vehicle(db, session_id, track_id)

            if vehicle is None:
                vehicle = VehicleSummary(
                    session_id=session_id,
                    track_id=track_id,
                    class_name=class_name,
                    first_frame=frame_number,
                    last_frame=frame_number,
                    avg_speed=speed or 0,
                    max_speed=speed or 0,
                    last_center_x=center_x,
                    last_center_y=center_y,
                )
                db.add(vehicle)
            else:
                vehicle.last_frame = frame_number
                if vehicle.class_name is None:
                    vehicle.class_name = class_name
                if vehicle.first_frame is None:
                    vehicle.first_frame = frame_number

                if speed is not None:
                    # NOTE: simple two-point average, not a true running
                    # average over many samples. Add a speed_sample_count
                    # column to VehicleSummary if you need exact averages.
                    vehicle.avg_speed = (vehicle.avg_speed + speed) / 2
                    vehicle.max_speed = max(vehicle.max_speed, speed)

                vehicle.last_center_x = center_x
                vehicle.last_center_y = center_y

            db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Failed to upsert vehicle summary (session_id=%s, track_id=%s)",
                session_id,
                track_id,
            )
            raise
        finally:
            db.close()

    def update_plate_bbox(
        self,
        session_id: int,
        track_id: int,
        bbox: Sequence[float],
    ) -> None:
        """Store the license-plate bounding box (x1, y1, x2, y2) for a vehicle."""
        db = SessionLocal()
        try:
            vehicle = self._get_vehicle(db, session_id, track_id)
            if vehicle:
                x1, y1, x2, y2 = bbox
                vehicle.plate_bbox = json.dumps(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                )
                db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Failed to update plate bbox (session_id=%s, track_id=%s)",
                session_id,
                track_id,
            )
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_raw_detection_count(self, session_id: int) -> int:
        """Return the number of detections for a session."""
        db = SessionLocal()
        try:
            return (
                db.query(DetectionRecord)
                .filter(DetectionRecord.session_id == session_id)
                .count()
            )
        except Exception:
            logger.exception(
                "Failed to get raw detection count (session_id=%s)", session_id
            )
            return 0
        finally:
            db.close()

    def get_unique_vehicle_count(self, session_id: int) -> int:
        """Return the number of unique tracked vehicles for a session."""
        db = SessionLocal()
        try:
            return (
                db.query(VehicleSummary)
                .filter(VehicleSummary.session_id == session_id)
                .count()
            )
        except Exception:
            logger.exception(
                "Failed to get unique vehicle count (session_id=%s)", session_id
            )
            return 0
        finally:
            db.close()
