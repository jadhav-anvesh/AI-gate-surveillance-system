# Backend Architecture

## AI-Powered Gate Surveillance and Vehicle Analytics System

---

# Project Goal

The objective of this project is to build a modular backend architecture for an AI-powered gate surveillance system capable of processing CCTV, RTSP, and video streams in real time.

The backend is responsible for the complete computer vision pipeline, while the frontend focuses only on visualization and user interaction.

The system supports multiple detection models, vehicle tracking, traffic analytics, and database-backed monitoring.

---

# Overall Architecture

```text
Video File / RTSP Stream
            │
            ▼
      Video Source
            │
            ▼
 Vehicle Detection
 (YOLOv10n / RF-DETR)
            │
            ▼
 Multi-Object Tracking
     (ByteTrack)
            │
            ▼
 ┌─────────────────────────────┐
 │      Analytics Layer        │
 │                             │
 │ Vehicle Counting            │
 │ Vehicle Flow Analysis       │
 │ Vehicle Density Analysis    │
 │ Vehicle Speed Estimation    │
 └─────────────────────────────┘
            │
            ▼
 Database Storage (SQLite)
            │
            ▼
 FastAPI REST API
            │
            ▼
 Streamlit Dashboard
```

---

# Backend Responsibilities

The backend performs all computational tasks.

* Video acquisition
* Frame preprocessing
* Model loading
* Vehicle detection
* Multi-object tracking
* Vehicle counting
* Entry/Exit flow estimation
* Density estimation
* Speed estimation
* Analytics computation
* Database management
* REST API endpoints
* Configuration management

---

# Frontend Responsibilities

The Streamlit dashboard is responsible only for visualization.

* Display processed video
* Display dashboard analytics
* Plot charts
* Display heatmaps
* View historical records
* Configure runtime settings
* Start and stop processing

---

# Backend Folder Structure

```text
backend/
│
├── api/
│   ├── main.py
│   ├── routes/
│   └── schemas/
│
├── config/
│   ├── camera_config.py
│   └── model_config.py
│
├── core/
│   ├── pipeline_manager.py
│   └── state.py
│
├── database/
│   ├── database.py
│   ├── models.py
│   └── detection_service.py
│
├── services/
│   ├── detector.py
│   ├── tracker.py
│   ├── pipeline.py
│   ├── flow.py
│   ├── density.py
│   ├── speed.py
│   └── video_source.py
│
└── tests/
```

---

# Processing Pipeline

```text
Video Source
      │
      ▼
Frame Capture
      │
      ▼
Resize Frame
      │
      ▼
Vehicle Detection
(YOLOv10n / RF-DETR)
      │
      ▼
ByteTrack Tracking
      │
      ▼
Analytics Update
│
├── Vehicle Counting
├── Flow Estimation
├── Density Estimation
└── Speed Estimation
      │
      ▼
Database Update
      │
      ▼
Visualization Frame
      │
      ▼
FastAPI Response
      │
      ▼
Streamlit Dashboard
```

---

# Backend Services

## 1. Video Source Service

**File**

```text
backend/services/video_source.py
```

### Responsibilities

* Read frames from RTSP streams
* Read local video files
* Frame resizing
* Frame validation
* Stream management

---

## 2. Detector Service

**File**

```text
backend/services/detector.py
```

### Responsibilities

* Load detection model
* Switch between YOLOv10n and RF-DETR
* Perform object detection
* Return bounding boxes, classes, and confidence scores

Supported models:

* YOLOv10n
* RF-DETR

---

## 3. Tracker Service

**File**

```text
backend/services/tracker.py
```

### Responsibilities

* Initialize ByteTrack
* Associate detections across frames
* Assign persistent tracker IDs
* Maintain track history

---

## 4. Flow Service

**File**

```text
backend/services/flow.py
```

### Responsibilities

* Detect line crossings
* Count entry and exit vehicles
* Maintain per-class statistics
* Generate traffic flow analytics

---

## 5. Density Service

**File**

```text
backend/services/density.py
```

### Responsibilities

* Compute traffic density
* Generate density heatmaps
* Maintain occupancy statistics
* Per-region density estimation

---

## 6. Speed Service

**File**

```text
backend/services/speed.py
```

### Responsibilities

* Estimate vehicle speed
* Perspective-aware calculations
* Per-class speed statistics
* Time-wise speed analytics

---

## 7. Pipeline Service

**File**

```text
backend/services/pipeline.py
```

### Responsibilities

* Coordinate complete processing pipeline
* Invoke detector
* Invoke tracker
* Update analytics services
* Generate visualization frames
* Return processed results

---

# API Layer

Located in:

```text
backend/api/
```

The API layer exposes REST endpoints for communication between the backend and frontend.

Main endpoints include:

* Pipeline control
* System status
* Database access
* Runtime configuration

---

# Core Layer

Located in:

```text
backend/core/
```

### pipeline_manager.py

Responsible for:

* Starting pipeline
* Stopping pipeline
* Managing processing threads
* Runtime state management

### state.py

Maintains application-wide runtime state including:

* Active pipeline
* Running status
* Current configuration
* Shared objects

---

# Database Layer

Located in:

```text
backend/database/
```

Responsibilities:

* Store detection results
* Store tracker information
* Store analytics records
* Historical data retrieval
* SQLite database management

---

# Configuration Layer

Located in:

```text
backend/config/
```

Configuration files include:

* Camera configuration
* Model configuration
* Detection parameters
* Runtime options

---

# Backend State Variables

The backend maintains runtime state for analytics and visualization.

Major state variables include:

* Vehicle distribution
* Vehicles currently in scene
* Lane-wise vehicle counts
* Lane-wise occupancy
* Vehicle speed statistics
* Time-wise speed history
* Track history
* Previous tracker positions
* Density maps
* Heatmap matrices

---

# Technologies Used

## Backend

* Python
* FastAPI

## Computer Vision

* OpenCV
* Ultralytics YOLOv10
* RF-DETR

## Tracking

* ByteTrack

## Database

* SQLite

## Frontend

* Streamlit

## Visualization

* Plotly
* Matplotlib

---

# Current System Status

| Module                   | Status         |
| ------------------------ | -------------- |
| Video Source             | ✅ Completed    |
| Detection                | ✅ Completed    |
| Tracking                 | ✅ Completed    |
| Flow Analysis            | ✅ Completed    |
| Density Analysis         | ✅ Completed    |
| Speed Estimation         | ✅ Completed    |
| Database Integration     | ✅ Completed    |
| FastAPI Backend          | ✅ Completed    |
| Streamlit Dashboard      | ✅ Completed    |
| Configuration Management | ✅ Completed    |
| ALPR Integration         | 🚧 In Progress |

---

# Future Improvements

* Complete ALPR integration
* Multi-camera processing
* Vehicle re-identification
* PostgreSQL support
* Docker deployment
* Cloud deployment
* WebSocket-based live streaming
* Traffic congestion prediction
* Anomaly detection
* Distributed processing support

---

# Design Philosophy

The project follows a modular service-oriented architecture where each component has a single responsibility.

Benefits include:

* Easy maintenance
* Better scalability
* Independent testing
* Clear separation of concerns
* Support for multiple detection models
* Easy integration of new analytics modules
* Simplified future deployment and extension
