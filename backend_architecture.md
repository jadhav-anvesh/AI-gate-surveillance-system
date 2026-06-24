# Gate Surveillance Backend Extraction Plan

## Goal

Move all processing logic from Streamlit into backend services.

Backend Responsibilities:
- Frame acquisition
- Model loading
- Detection
- Tracking
- Vehicle counting
- Flow estimation
- Density analysis
- Heatmap generation
- Speed estimation

Frontend Responsibilities:
- Display images
- Display charts
- Display analytics results

---

## Current Processing Pipeline

Video/RTSP
    ↓
frame_grabber()
    ↓
model.track()
    ↓
ByteTrack
    ↓
Flow Analysis
    ↓
Density Analysis
    ↓
Heatmap Generation
    ↓
Speed Analysis
    ↓
Dashboard Visualization

---

## Backend Extraction Status

[X] Frame Grabber
[X] Model Loading
[X] Detection
[X] Tracking
[X] Flow Analysis
[X] Density Analysis
[X] Speed Analysis
[X] pipeline
[ ] Heatmap Generation
[ ] API Layer
[ ] Database Layer

## Component Mapping

| Component                 | Current Location | Backend | Frontend |
|---------------------------|------------------|---------|----------|
| frame_grabber             | analysis.py      | YES     | NO       |
| load_model                | analysis.py      | YES     | NO       |
| YOLO/RT-DETR Detection    | analysis.py      | YES     | NO       |
| ByteTrack Tracking        | analysis.py      | YES     | NO       |
| Flow Counting             | analysis.py      | YES     | NO       |
| Heatmap Generation        | utils.py         | YES     | NO       |
| Speed Analysis            | analysis.py      | YES     | NO       |
| Plotly Charts             | analysis.py      | NO      | YES      |
| Streamlit Widgets         | analysis.py      | NO      | YES      |
| Session State UI Controls | analysis.py      | NO      | YES      |

## Backend State Variables

vehicle_distribution_map

vehicle_in_scene_map

lane_wise_vehicle_distribution_maps

lane_wise_vehicle_in_scene_maps

vehicle_speed_class_map

vehicle_speed_time_map

track_history

last_positions

heatmap

coordinates_for_speed

## Backend Dependencies

- ultralytics
- supervision
- opencv-python
- numpy
- torch
- utils.py
- parameters.py

## Frontend Dependencies

- streamlit
- plotly

## Planned Backend Services

DetectorService
- load_model()
- detect()

TrackingService
- initialize_tracker()
- track()

FlowService
- update()
- get_statistics()

DensityService
- update()
- get_heatmap()

SpeedService
- update()
- get_statistics()

VideoProcessingPipeline
- process_frame()
- process_video()

## Known Issues

- supervision.ByteTrack is deprecated since v0.28.0
- Current implementation still works
- Migration may be required before supervision v0.30.0
