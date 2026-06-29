"""
Database API Routes

This module exposes REST endpoints for:
1. Database statistics
2. Processing session information
3. Detection records
4. Vehicle summaries
"""

# ============================================================================
# Imports
# ============================================================================

from fastapi import APIRouter

from backend.database.database import SessionLocal

from backend.database.models import (
    DetectionRecord,
    ProcessingSession,
    VehicleSummary,
)

# ============================================================================
# Router
# ============================================================================

# All endpoints in this module are prefixed with `/database`.
router = APIRouter(prefix="/database", tags=["Database"])


# ============================================================================
# Database Statistics
# ============================================================================


@router.get("/statistics")
def get_database_statistics():
    """
    Returns aggregate statistics across the entire database.

    Includes:
        • Total sessions
        • Total detection records
        • Total tracked vehicles
    """
    db = SessionLocal()

    try:
        # Count records stored in each database table.
        return {
            "total_sessions": db.query(ProcessingSession).count(),
            "total_raw_detection_records": db.query(DetectionRecord).count(),
            "total_unique_vehicles": db.query(VehicleSummary).count(),
        }

    finally:
        # Ensure the database connection is closed
        # even if an exception occurs.
        db.close()


# ============================================================================
# Session Endpoints
# ============================================================================


@router.get("/sessions")
def get_sessions():
    """
    Returns all processing sessions stored in the database.

    Each session represents one execution of the
    video processing pipeline.
    """
    # Create a new SQLAlchemy session for executing
    # database queries within this request.
    db = SessionLocal()

    try:
        # Retrieve every processing session.
        sessions = db.query(ProcessingSession).all()

        # Format database records into a JSON response.
        return [
            {
                "id": s.id,
                "camera_type": s.camera_type,
                "status": s.status,
                "start_time": str(s.start_time),
                "end_time": str(s.end_time),
            }
            for s in sessions
        ]

    finally:
        db.close()


@router.get("/session/{session_id}")
def get_session_details(session_id: int):
    """
    Returns detailed analytics for a single processing session.

    Includes:
        • Session metadata
        • Detection statistics
        • Vehicle counts
        • Average speed
    """
    db = SessionLocal()

    try:
        # Retrieve the requested processing session.
        session = (
            db.query(ProcessingSession)
            .filter(ProcessingSession.id == session_id)
            .first()
        )

        if session is None:
            return {"error": "Session not found"}

        # Retrieve all raw detections belonging to this session.
        detections = (
            db.query(DetectionRecord)
            .filter(DetectionRecord.session_id == session_id)
            .all()
        )

        # Retrieve one summary record for every tracked vehicle.
        vehicle_summary = (
            db.query(VehicleSummary)
            .filter(VehicleSummary.session_id == session_id)
            .all()
        )

        # Extract valid speed measurements for averaging.
        speeds = [d.speed for d in detections if d.speed is not None]

        # Aggregate the number of tracked vehicles for
        # each detected vehicle class.
        vehicle_counts = {}
        for d in vehicle_summary:
            vehicle_counts[d.class_name] = vehicle_counts.get(d.class_name, 0) + 1

        # Combine session metadata and analytics into
        # a single response object.
        return {
            "session_id": session.id,
            "camera_type": session.camera_type,
            "status": session.status,
            "start_time": str(session.start_time),
            "end_time": str(session.end_time),
            "total_raw_detection_records": len(detections),
            "total_unique_vehicles": len(vehicle_summary),
            "average_speed": (sum(speeds) / len(speeds) if len(speeds) > 0 else None),
            "vehicle_counts": vehicle_counts,
        }

    finally:
        db.close()


# ============================================================================
# Detection Records
# ============================================================================


@router.get("/detections")
def get_detections(limit: int = 50):
    """
    Returns the most recent detection records.

    Parameters:
        limit:
            Maximum number of records to return.
    """
    db = SessionLocal()

    try:
        # Retrieve the latest detection records,
        # ordered by insertion time.
        detections = (
            db.query(DetectionRecord)
            .order_by(DetectionRecord.id.desc())
            .limit(limit)
            .all()
        )

        # Format database records into a JSON response.
        return [
            {
                "id": d.id,
                "session_id": d.session_id,
                "frame_number": d.frame_number,
                "track_id": d.track_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "speed": d.speed,
                "center_x": d.center_x,
                "center_y": d.center_y,
            }
            for d in detections
        ]

    finally:
        db.close()


# ============================================================================
# Vehicle Summary
# ============================================================================


@router.get("/vehicle_summary")
def get_vehicle_summary():
    """
    Returns one summary entry for every tracked vehicle.

    Each record contains the vehicle's lifetime
    statistics collected during a session.
    """
    db = SessionLocal()

    try:
        # Retrieve one summary record per tracked vehicle.
        # Each record contains aggregated statistics collected
        # throughout the vehicle's lifetime in the session.
        vehicles = db.query(VehicleSummary).order_by(VehicleSummary.id.desc()).all()

        # Format database records into a JSON response.
        return [
            {
                "track_id": v.track_id,
                "session_id": v.session_id,
                "class_name": v.class_name,
                "first_frame": v.first_frame,
                "last_frame": v.last_frame,
                "avg_speed": v.avg_speed,
                "max_speed": v.max_speed,
                "last_center_x": v.last_center_x,
                "last_center_y": v.last_center_y,
                "plate_bbox": v.plate_bbox,
                "plate_text": v.plate_text,
            }
            for v in vehicles
        ]

    finally:
        db.close()
