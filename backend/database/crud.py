from datetime import datetime

from backend.database.database import SessionLocal
from backend.database.models import DetectionRecord


def save_detection(
    session_id,
    frame_number,
    track_id,
    class_id,
    class_name,
    speed,
    center_x,
    center_y
):

    db = SessionLocal()

    try:

        record = DetectionRecord(
            session_id=session_id,
            timestamp=datetime.now(),
            frame_number=frame_number,
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            speed=speed,
            center_x=center_x,
            center_y=center_y
        )

        db.add(record)

        db.commit()

    finally:

        db.close()
