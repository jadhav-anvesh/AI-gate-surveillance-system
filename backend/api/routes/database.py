from fastapi import APIRouter

from backend.database.database import (SessionLocal)

from backend.database.models import (ProcessingSession,DetectionRecord,VehicleSummary)
#This router assigns a prefix to all the routes defined using it
router = APIRouter(
    prefix="/database",
    tags=["Database"]
)

@router.get("/sessions")
def get_sessions():
    #creates database connection
    db = SessionLocal()
    #run try block no matter what happens, run finally block this keeps data from leaking connections
    try:

        sessions = db.query(ProcessingSession).all()

        return [
            {
                "id": s.id,
                "camera_type": s.camera_type,
                "status": s.status,
                "start_time": str(s.start_time),
                "end_time": str(s.end_time)
            }
            for s in sessions
        ]

    finally:

        db.close()

@router.get("/statistics")
def get_database_statistics():

    db = SessionLocal()

    try:

        return {

            "total_sessions":
                db.query(
                    ProcessingSession
                ).count(),

            "total_raw_detection_records":
                db.query(
                    DetectionRecord
                ).count(),

            "total_unique_vehicles":
                db.query(
                    VehicleSummary
                ).count()
        }

    finally:

        db.close()

@router.get("/session/{session_id}")
def get_session_details(
    session_id: int
):

    db = SessionLocal()

    try:

        session = db.query(
            ProcessingSession
        ).filter(
            ProcessingSession.id == session_id
        ).first()

        if session is None:

            return {
                "error": "Session not found"
            }

        detections = db.query(
            DetectionRecord
        ).filter(
            DetectionRecord.session_id == session_id
        ).all()
        vehicle_summary = db.query(
            VehicleSummary
        ).filter(
            VehicleSummary.session_id == session_id
        ).all()
        speeds = [
            d.speed
            for d in detections
            if d.speed is not None
        ]

        vehicle_counts = {}

        for d in vehicle_summary:

            vehicle_counts[
                d.class_name
            ] = (
                vehicle_counts.get(
                    d.class_name,
                    0
                ) + 1
            )

        return {

            "session_id":
                session.id,

            "camera_type":
                session.camera_type,

            "status":
                session.status,

            "start_time":
                str(session.start_time),

            "end_time":
                str(session.end_time),

            "total_raw_detection_records":
                len(detections),

            "total_unique_vehicles":
                len(vehicle_summary),

            "average_speed":
                (
                    sum(speeds)
                    / len(speeds)
                )
                if len(speeds) > 0
                else None,

            "vehicle_counts":
                vehicle_counts
        }

    finally:

        db.close()

@router.get("/detections")
def get_detections(
    limit: int = 50
):

    db = SessionLocal()

    try:

        detections = (
            db.query(
                DetectionRecord
            )
            .order_by(
                DetectionRecord.id.desc()
            )
            .limit(limit)
            .all()
        )

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
                "center_y": d.center_y
            }

            for d in detections
        ]

    finally:

        db.close()
@router.get("/vehicle_summary")
def get_vehicle_summary():

    db = SessionLocal()

    try:

        vehicles = (
            db.query(
                VehicleSummary
            )
            .order_by(
                VehicleSummary.id.desc()
            )
            .all()
        )

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
                "plate_text": v.plate_text
            }

            for v in vehicles
        ]

    finally:

        db.close()
