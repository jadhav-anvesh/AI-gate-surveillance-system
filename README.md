# AI-Powered Gate Surveillance and Vehicle Analytics System

A real-time AI-powered gate surveillance and vehicle analytics system for intelligent traffic monitoring using CCTV or RTSP camera streams. The system combines deep learning-based object detection, multi-object tracking, traffic analytics, and an interactive dashboard to provide comprehensive insights into vehicle movement.

---

# 🚀 Project Overview

This project was developed as part of the **Summer Research Internship Program (SRIP)**.

The application processes live CCTV or RTSP streams and performs real-time vehicle analytics, including:

* 🚗 Vehicle Detection
* 🎯 Multi-Object Tracking
* 🔢 Vehicle Counting
* ↔️ Entry/Exit Flow Analysis
* 📊 Vehicle Density Estimation
* 🚦 Vehicle Speed Estimation
* 📈 Interactive Analytics Dashboard
* 🔍 Automatic License Plate Recognition (ALPR) *(Under Development)*

The system has been redesigned from a **monolithic Streamlit application** into a **modular FastAPI backend**, making it scalable, maintainable, and easier to extend.

---

# ✨ Key Features

## Vehicle Detection

* YOLOv10n-based vehicle detection
* RF-DETR integration
* Configurable confidence threshold
* Configurable IoU threshold
* Multi-class vehicle detection

Supported classes include:

* Car
* Bus
* Truck
* Motorcycle
* Auto-rickshaw
* Bicycle
* Caravan
* Person
* Rider
* Traffic light
* Traffic sign
* Trailer
* Train
* Vehicle fallback
* Animal

---

## Multi-Object Tracking

* Persistent tracker IDs
* Vehicle trajectory tracking
* Frame-to-frame association
* Real-time tracking visualization

---

## Vehicle Flow Estimation

* Entry counting
* Exit counting
* Line-crossing analytics
* Traffic movement statistics

---

## Vehicle Density Analysis

* Density heatmap generation
* Traffic concentration visualization
* Spatial distribution analysis

---

## Vehicle Speed Estimation

* Perspective-aware speed estimation
* Average speed statistics
* Per-class speed analytics
* Real-time speed computation

---

## Dashboard Analytics

Interactive Streamlit dashboard providing:

* Vehicle counts
* Vehicle distribution
* Speed statistics
* Density heatmaps
* Flow analytics
* Historical database records
* Live visualization

---

## Video Source Support

* RTSP camera streams
* CCTV feeds
* Local video files

---

# 🏗️ System Pipeline

```text
RTSP / CCTV / Video File
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
Analytics Layer
 ├── Vehicle Counting
 ├── Flow Estimation
 ├── Density Analysis
 └── Speed Estimation
            │
            ▼
SQLite Database
            │
            ▼
FastAPI Backend
            │
            ▼
Streamlit Dashboard
```

---

# 🏛️ Backend Architecture

## API Layer

Handles REST API endpoints.

```text
backend/api/
├── main.py
├── routes/
└── schemas/
```

---

## Core Layer

Responsible for orchestration and application state.

```text
backend/core/
├── pipeline_manager.py
└── state.py
```

---

## Services Layer

Contains the core analytics modules.

```text
backend/services/
├── detector.py
├── tracker.py
├── pipeline.py
├── flow.py
├── density.py
├── speed.py
├── video_source.py
└── visualization.py
```

---

## Database Layer

Responsible for data storage and retrieval.

```text
backend/database/
├── database.py
├── models.py
├── crud.py
└── detection_service.py
```

---

# 📂 Project Structure

```text
.
├── backend/
│   ├── api/
│   ├── config/
│   ├── core/
│   ├── database/
│   ├── models/
│   └── services/
│   
│
├── database_dashboard.py
├── alpr_api.py
├── backend_architecture.md
├── current_config.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ Requirements

* Python 3.11
* CUDA-enabled GPU (Recommended)
* OpenCV
* FastAPI
* Streamlit
* SQLite
* Ultralytics
* RF-DETR
* ByteTrack

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/jadhav-anvesh/AI-gate-surveillance-system.git

cd AI-gate-surveillance-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📥 Model Setup

This repository **does not include trained model weights**.

Download or place the required model weights into the expected model directories before running the project.

Supported detection models:

* YOLOv10n
* RF-DETR

---

# ⚙️ Configuration

Runtime settings are controlled using:

```text
current_config.json
```

Examples of configurable parameters:

* Detection model
* Confidence threshold
* IoU threshold
* Video source
* Analytics modules
* Stream settings

---

# ▶️ Running the Backend

Start the FastAPI server:

```bash
uvicorn backend.api.main:app 
```

FastAPI documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

# 📊 Running the Dashboard

Launch the Streamlit dashboard:

```bash
python -m streamlit run --server.fileWatcherType none database_dashboard.py
```

Dashboard URL:

```text
http://localhost:8502
```

---

# 🚀 How to Use

1. Start the FastAPI backend.
2. Launch the Streamlit dashboard.
3. Select the preferred detection model (YOLOv10n or RF-DETR).
4. Configure the input source (RTSP stream or video file).
5. Start the processing pipeline.
6. Monitor detections and analytics in real time.
7. View historical records from the dashboard database.

---

# 📈 Dashboard Features

The dashboard provides:

* Live detection visualization
* Vehicle type distribution
* Vehicle count statistics
* Speed analysis
* Vehicle density heatmaps
* Entry/Exit flow analytics
* Historical database records
* Tracker visualization

---

# 🤖 Supported Detection Models

## YOLOv10n

* Lightweight
* Faster inference
* Optimized for real-time applications

---

## RF-DETR

* Transformer-based detector
* Better localization accuracy
* Improved performance in crowded scenes

---

# 🚘 Automatic License Plate Recognition (ALPR)

The project includes an asynchronous ALPR pipeline.

Workflow:

1. Detect vehicle
2. Assign tracker ID
3. Crop detected vehicle
4. Detect license plate
5. Recognize license plate text
6. Associate plate with tracker ID
7. Update the database

The ALPR module is currently under active development and is not enabled in the default processing pipeline.

---

# ⚡ Performance Highlights

* Real-time inference
* Modular FastAPI architecture
* Multi-model detection support
* ByteTrack integration
* SQLite-backed analytics
* Interactive Streamlit dashboard
* RTSP camera support
* End-to-end traffic analytics pipeline

---

# 🔬 Research Contributions

## Model Development

* Fine-tuned YOLOv10n on the Indian Driving Dataset (IDD)
* Evaluated detection performance on Indian traffic scenarios

## System Engineering

* Migrated a monolithic Streamlit application into a modular FastAPI architecture
* Developed reusable analytics services
* Integrated detection, tracking, analytics, and visualization into a unified pipeline

---

# 🚧 Future Work

* Complete ALPR integration
* Multi-camera support
* Vehicle re-identification
* PostgreSQL support
* Docker deployment
* Cloud deployment
* Traffic congestion prediction
* Anomaly detection
* Vehicle dwell-time estimation

---

# 📸 Screenshots

You can add screenshots here for better visualization.

```text
images/
├── dashboard.png
├── yolo_detection.png
├── rfdetr_detection.png
├── density_heatmap.png
├── flow_analysis.png
└── speed_estimation.png
```

Example:

```markdown
![Dashboard](images/dashboard.png)

![YOLO Detection](images/yolo_detection.png)

![RF-DETR Detection](images/rfdetr_detection.png)

![Density Heatmap](images/density_heatmap.png)

![Flow Analysis](images/flow_analysis.png)

![Speed Estimation](images/speed_estimation.png)
```

---

# 📝 Important Notes

This repository intentionally excludes:

* Trained model weights
* Datasets
* Runtime artifacts
* Temporary files
* SQLite database files
* Logs and cache files

Please download the required model weights separately before running the project.

---

# 👨‍💻 Author

**Anvesh Jadhav**

Summer Research Internship Program (SRIP)

AI-Powered Gate Surveillance and Vehicle Analytics System
