"""
SQLAlchemy ORM models.

Defines the database schema used by the Gate Surveillance backend.

Models:
    - ProcessingSession
    - DetectionRecord
    - VehicleSummary
"""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from backend.database.database import Base


# ============================================================
# Processing Session
# ============================================================


class ProcessingSession(Base):
    """Stores metadata for a single pipeline execution."""

    __tablename__ = "processing_sessions"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Session metadata
    camera_type = Column(String, nullable=False)
    status = Column(String, nullable=False)

    # Lifecycle timestamps
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)


# ============================================================
# Raw Detection Records
# ============================================================


class DetectionRecord(Base):
    """Stores every object detected in every processed frame."""

    __tablename__ = "detection_records"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Relationships
    session_id = Column(
        Integer,
        ForeignKey("processing_sessions.id"),
        nullable=False,
        index=True,
    )

    # Tracking information
    timestamp = Column(DateTime, nullable=False)
    frame_number = Column(Integer, nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)

    # Detection information
    class_id = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    speed = Column(Float, nullable=True)

    # Geometry
    center_x = Column(Float, nullable=False)
    center_y = Column(Float, nullable=False)


# ============================================================
# Vehicle Summary
# ============================================================


class VehicleSummary(Base):
    """Stores aggregated information for each tracked vehicle."""

    __tablename__ = "vehicle_summary"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Relationships
    session_id = Column(
        Integer,
        ForeignKey("processing_sessions.id"),
        nullable=False,
        index=True,
    )
    track_id = Column(Integer, nullable=False, index=True)

    # Detection information
    class_name = Column(String, nullable=False)

    # Tracking information
    first_frame = Column(Integer, nullable=False)
    last_frame = Column(Integer, nullable=False)

    # Speed statistics
    avg_speed = Column(Float, default=0.0)
    max_speed = Column(Float, default=0.0)

    # Geometry
    last_center_x = Column(Float, nullable=False)
    last_center_y = Column(Float, nullable=False)

    # License plate data
    plate_bbox = Column(String, nullable=True)
    plate_text = Column(String, nullable=True)
