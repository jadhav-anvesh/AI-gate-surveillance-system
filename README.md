# AI-Powered Gate Surveillance and Vehicle Analytics System

## Overview

This project is a real-time Gate Surveillance and Vehicle Analytics System developed as part of the Summer Research Internship Program (SRIP).

The system performs real-time vehicle detection, tracking, counting, flow analysis, density estimation, speed estimation, and dashboard visualization from CCTV/RTSP camera streams.

The architecture was redesigned from a monolithic Streamlit application into a modular FastAPI backend with independent services for detection, tracking, analytics, and database management.

---

## Key Features

### Vehicle Detection

* YOLOv10n-based vehicle detection
* RF-DETR integration for improved detection performance
* Configurable confidence and IoU thresholds
* Multi-class vehicle detection

### Multi-Object Tracking

* Persistent vehicle tracking using tracker IDs
* Vehicle trajectory maintenance
* Real-time object association across frames

### Vehicle Flow Estimation

* Entry and Exit counting
* Line-crossing analytics
* Vehicle movement statistics

### Vehicle Density Analysis

* Density heatmap generation
* Traffic concentration analysis
* Spatial vehicle distribution visualization

### Vehicle Speed Estimation

* Real-time speed estimation
* Speed statistics and analytics
* Perspective-aware calculations

### Dashboard Analytics

* Vehicle type distribution
* Vehicle count statistics
* Speed analysis
* Density visualization
* Flow analytics
* Historical database records

### RTSP/CCTV Support

* Live camera stream processing
* RTSP stream integration
* Real-time analytics pipeline

---

## System Architecture

RTSP/CCTV Stream

↓

Vehicle Detection (YOLOv10 / RF-DETR)

↓

Multi-Object Tracking

↓

Analytics Layer

* Vehicle Flow
* Density Analysis
* Speed Estimation

↓

Database Storage

↓

Streamlit Dashboard

---

## Backend Architecture

The backend is organized into modular services:

### API Layer

Handles all REST endpoints.

```text
backend/api/
├── routes/
├── schemas/
└── main.py
```

### Core Layer

Responsible for pipeline orchestration and state management.

```text
backend/core/
├── pipeline_manager.py
└── state.py
```

### Services Layer

Contains all analytics modules.

```text
backend/services/
├── detector.py
├── tracker.py
├── flow.py
├── density.py
├── speed.py
├── video_source.py
└── pipeline.py
```

### Database Layer

Database operations and models.

```text
backend/database/
├── database.py
├── models.py
├── crud.py
└── detection_service.py
```

---

## Technologies Used

### Backend

* FastAPI
* Python

### Computer Vision

* OpenCV
* YOLOv10
* RF-DETR

### Tracking

* ByteTrack

### Frontend

* Streamlit

### Database

* SQLite

### Visualization

* Matplotlib
* Plotly

---

## Project Structure

```text
.
├── backend/
│   ├── api/
│   ├── config/
│   ├── core/
│   ├── database/
│   ├── services/
│   └── tests/
│
├── database_dashboard.py
├── alpr_api.py
├── backend_architecture.md
├── current_config.json
└── README.md
```

---

## Implemented Analytics

### Vehicle Counting

Counts vehicles entering and exiting monitored zones.

### Vehicle Flow Analysis

Tracks traffic movement patterns across defined lines.

### Density Analysis

Generates heatmaps showing traffic concentration.

### Speed Estimation

Computes approximate vehicle speed using calibrated transformations.

### Vehicle Distribution

Provides statistics for different vehicle categories.

---

## ALPR Module Status

Automatic License Plate Recognition (ALPR) integration is currently under development.

Current approach:

1. Vehicle detection assigns tracker IDs.
2. Crops from newly detected vehicles are saved.
3. ALPR operates asynchronously on saved crops.
4. Extracted plate numbers are linked back to tracker IDs.
5. Database records are updated with recognized license plates.

The ALPR module is experimental and is not part of the production pipeline yet.

---

## Performance Highlights

* Real-time CCTV stream processing
* Modular backend architecture
* Multi-model detection support
* End-to-end analytics pipeline
* RTSP stream compatibility
* Database-backed analytics

---

## Research Contributions

### Model Training

* Fine-tuned YOLOv10n on the Indian Driving Dataset (IDD)
* Evaluated detection performance on Indian traffic scenarios

### System Engineering

* Migrated a monolithic application into modular backend services
* Developed scalable analytics modules
* Integrated detection, tracking, analytics, and visualization pipelines

---

## Future Work

### ALPR Completion

* Full license plate recognition integration
* Plate-to-vehicle association
* Searchable vehicle records

### Multi-Camera Support

* Simultaneous camera processing
* Cross-camera tracking

### Advanced Analytics

* Vehicle dwell time
* Congestion prediction
* Anomaly detection

### Cloud Deployment

* Docker deployment
* Scalable backend services
* Centralized monitoring

---

## Important Notes

This repository intentionally excludes:

* Trained model weights
* Datasets
* Runtime artifacts
* Temporary files
* Database files

Model weights must be placed in the expected model directories before running the system.

---

## Author

Anvesh Jadhav

Summer Research Internship Program (SRIP)

AI-Powered Gate Surveillance and Vehicle Analytics System
