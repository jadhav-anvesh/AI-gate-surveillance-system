from datetime import datetime

from backend.database.database import SessionLocal

from backend.database.models import (
    DetectionRecord,
    VehicleSummary
)
import json

class DetectionDatabaseService:

    def save_detection(
        self,
        session_id,
        frame_number,
        track_id,
        class_id,
        class_name,
        confidence,
        speed,
        center_x,
        center_y
    ):

        db = SessionLocal()

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
            center_y=center_y
        )

        db.add(record)

        db.commit()

        db.close()

    def upsert_vehicle_summary(
        self,
        session_id,
        frame_number,
        track_id,
        class_name,
        speed,
        center_x,
        center_y
    ):
        db = SessionLocal()

        vehicle = (
            db.query(VehicleSummary)
            .filter(
                VehicleSummary.session_id == session_id,
                VehicleSummary.track_id == track_id
            )
            .first()
        )

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
                last_center_y=center_y
            )

            db.add(vehicle)
        else:

            vehicle.last_frame = frame_number
            if vehicle.class_name is None:
                vehicle.class_name = class_name

            if vehicle.first_frame is None:
                vehicle.first_frame = frame_number
            if speed is not None:

                vehicle.avg_speed = (
                    vehicle.avg_speed + speed
                ) / 2

                vehicle.max_speed = max(
                    vehicle.max_speed,
                    speed
                )

            vehicle.last_center_x = center_x
            vehicle.last_center_y = center_y

        db.commit()
        db.close()
    def update_plate_bbox(
        self,
        session_id,
        track_id,
        bbox
    ):

        db = SessionLocal()

        vehicle = (
            db.query(VehicleSummary)
            .filter(
                VehicleSummary.session_id == session_id,
                VehicleSummary.track_id == track_id
            )
            .first()
        )
      
        if vehicle:

            vehicle.plate_bbox = json.dumps({
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3]
            })

            db.commit()


        db.close()
    def get_raw_detection_count(self, session_id):

        db = SessionLocal()

        count = (
            db.query(DetectionRecord)
            .filter(
                DetectionRecord.session_id == session_id
            )
            .count()
        )

        db.close()

        return count


    def get_unique_vehicle_count(self, session_id):

        db = SessionLocal()

        count = (
            db.query(VehicleSummary)
            .filter(
                VehicleSummary.session_id == session_id
            )
            .count()
        )

        db.close()

        return count
