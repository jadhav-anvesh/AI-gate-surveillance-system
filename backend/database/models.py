from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey
)
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime
)
from sqlalchemy import String
from backend.database.database import Base


class ProcessingSession(Base):

    __tablename__ = "processing_sessions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    camera_type = Column(
        String
    )

    status = Column(
        String
    )

    start_time = Column(
        DateTime
    )

    end_time = Column(
        DateTime,
        nullable=True
    )


class DetectionRecord(Base):

    __tablename__ = "detection_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    session_id = Column(
        Integer,
        ForeignKey(
            "processing_sessions.id"
        )
    )

    timestamp = Column(
        DateTime
    )

    frame_number = Column(
        Integer
    )

    track_id = Column(
        Integer
    )

    class_id = Column(
        Integer
    )

    class_name = Column(
        String
    )

    confidence = Column(
        Float
    )

    speed = Column(
        Float,
        nullable=True
    )

    center_x = Column(
        Float
    )

    center_y = Column(
        Float
    )

class VehicleSummary(Base):

    __tablename__ = "vehicle_summary"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    session_id = Column(
        Integer,
        ForeignKey(
            "processing_sessions.id"
        )
    )
    track_id = Column(
        Integer
    )

    class_name = Column(
        String
    )

    first_frame = Column(
        Integer
    )

    last_frame = Column(
        Integer
    )

    avg_speed = Column(
        Float,
        default=0
    )

    max_speed = Column(
        Float,
        default=0
    )

    last_center_x = Column(
        Float
    )

    last_center_y = Column(
        Float
    )
    plate_bbox = Column(
        String,
        nullable=True
    )
    plate_text = Column(
        String,
        nullable=True
    )
