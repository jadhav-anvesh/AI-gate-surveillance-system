"""
Gate Surveillance — backend dashboard.

Streamlit frontend for configuring, starting/stopping, and
monitoring the detection pipeline, plus browsing the stored
session/detection/vehicle data.
"""

import logging
from typing import Any, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS
# ==========================================

BACKEND_URL = "http://127.0.0.1:8000"

# Timeouts (seconds). Frame polling uses a shorter timeout since it's
# called on every auto-refresh tick; everything else gets a longer one
# so a slow backend doesn't hang the UI indefinitely.
DEFAULT_TIMEOUT = 5.0
FRAME_FETCH_TIMEOUT = 2.0

CLASS_OPTIONS = [
    "animal",
    "autorickshaw",
    "bicycle",
    "bus",
    "car",
    "caravan",
    "motorcycle",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "trailer",
    "train",
    "truck",
    "vehicle fallback",
]

MODEL_SET_OPTIONS = ["Trained Model"]
MODE_OPTIONS = ["YOLO-N", "RF-DETR"]
DEVICE_OPTIONS = ["GPU"]

MAX_LINES = 10
DEFAULT_LINE = {"x1": 0, "y1": 100, "x2": 640, "y2": 180}

# Calibration defaults for the speed-estimation region. Sourced from
# whatever the camera was last set up against — change these if the
# camera position changes.
DEFAULT_SOURCE_POINTS = [
    [344, 83],  # top-left
    [516, 99],  # top-right
    [557, 411],  # bottom-right
    [-236, 218],  # bottom-left
]
DEFAULT_ROAD_WIDTH = 19
DEFAULT_ROAD_LENGTH = 19

DEFAULT_FRAME_RATE = 10
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480

# NOTE: credentials are embedded directly in this URL. Fine for a local
# SRIP demo, but if this file is ever committed/shared more broadly,
# move this to an environment variable instead.
RTSP_URL = "rtsp://arjun.badola:AB%23ai2025@10.0.102.54:554/"

# ==========================================
# BACKEND HTTP HELPERS
# ==========================================
# Two layers:
#   _get_raw / _post_raw  -> return the raw requests.Response, may raise
#   backend_get / backend_post -> raw + .json(), may raise (callers that
#       already wrap their call site in try/except rely on this)
#   fetch_jpeg            -> always returns Optional[np.ndarray], never raises
# ==========================================


def _get_raw(
    path: str, params: Optional[dict] = None, timeout: float = DEFAULT_TIMEOUT
) -> requests.Response:
    return requests.get(f"{BACKEND_URL}{path}", params=params, timeout=timeout)


def _post_raw(
    path: str, json_payload: Optional[dict] = None, timeout: float = DEFAULT_TIMEOUT
) -> requests.Response:
    return requests.post(f"{BACKEND_URL}{path}", json=json_payload, timeout=timeout)


def backend_get(
    path: str, params: Optional[dict] = None, timeout: float = DEFAULT_TIMEOUT
) -> Any:
    """GET a backend endpoint and return parsed JSON. Raises on failure."""
    return _get_raw(path, params=params, timeout=timeout).json()


def backend_post(
    path: str, json_payload: Optional[dict] = None, timeout: float = DEFAULT_TIMEOUT
) -> Any:
    """POST to a backend endpoint and return parsed JSON. Raises on failure."""
    return _post_raw(path, json_payload=json_payload, timeout=timeout).json()


def fetch_jpeg(
    endpoint: str, timeout: float = FRAME_FETCH_TIMEOUT
) -> Optional[np.ndarray]:
    """Fetch a JPEG frame from a `/pipeline/<endpoint>` route and decode it to BGR."""
    try:
        response = _get_raw(f"/pipeline/{endpoint}", timeout=timeout)
        if response.headers.get("content-type") == "image/jpeg":
            arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.debug("Failed to fetch frame from %s: %s", endpoint, e)
    return None


# ==========================================
# BACKEND CONFIG LOADER
# ==========================================


def load_config_from_backend() -> dict:
    """Fetch the previously saved pipeline configuration, if any."""
    try:
        response = _get_raw("/pipeline/current_config", timeout=FRAME_FETCH_TIMEOUT)
        logger.info("Config fetch status=%s", response.status_code)
        if response.status_code == 200:
            return response.json()
    except Exception:
        logger.exception("Failed to load config from backend")
    return {}


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Gate Surveillance Backend Dashboard", layout="wide")

# ==========================================
# SESSION STATE DEFAULTS
# ==========================================


def ensure_session_defaults() -> None:
    """Populate session_state with sane defaults on first run."""
    st.session_state.setdefault("model_set", "Trained Model")
    st.session_state.setdefault("mode", "YOLO-N")
    st.session_state.setdefault("confidence_threshold", 0.30)
    st.session_state.setdefault("iou_threshold", 0.50)

    st.session_state.setdefault("device", "GPU")
    st.session_state.setdefault("classes_to_detect", CLASS_OPTIONS)
    st.session_state.setdefault("pipeline_started", False)

    st.session_state.setdefault("num_lines", 1)
    for i in range(MAX_LINES):
        st.session_state.setdefault(f"x1_{i}", DEFAULT_LINE["x1"])
        st.session_state.setdefault(f"y1_{i}", DEFAULT_LINE["y1"])
        st.session_state.setdefault(f"x2_{i}", DEFAULT_LINE["x2"])
        st.session_state.setdefault(f"y2_{i}", DEFAULT_LINE["y2"])

    (top_left, top_right, bottom_right, bottom_left) = DEFAULT_SOURCE_POINTS
    st.session_state.setdefault("x_top_left", top_left[0])
    st.session_state.setdefault("y_top_left", top_left[1])
    st.session_state.setdefault("x_top_right", top_right[0])
    st.session_state.setdefault("y_top_right", top_right[1])
    st.session_state.setdefault("x_bottom_right", bottom_right[0])
    st.session_state.setdefault("y_bottom_right", bottom_right[1])
    st.session_state.setdefault("x_bottom_left", bottom_left[0])
    st.session_state.setdefault("y_bottom_left", bottom_left[1])

    st.session_state.setdefault("road_width", DEFAULT_ROAD_WIDTH)
    st.session_state.setdefault("road_length", DEFAULT_ROAD_LENGTH)


ensure_session_defaults()

# ==========================================
# SESSION STATE INIT (load saved config once)
# ==========================================

if "session_state_loaded" not in st.session_state:
    st.session_state.session_state_loaded = True

    saved = load_config_from_backend()

    st.session_state.model_set = saved.get("camera_type", "Trained Model")
    st.session_state.mode = saved.get("det_mode", "YOLO-N")
    st.session_state.confidence_threshold = saved.get("confidence_threshold", 0.30)
    st.session_state.iou_threshold = saved.get("iou_threshold", 0.50)

    st.session_state.device = saved.get("device", "GPU")
    st.session_state.classes_to_detect = saved.get("classes_to_detect", CLASS_OPTIONS)

    st.session_state.pipeline_started = False

    st.session_state.num_lines = saved.get("num_lines", 1)

    saved_lines = saved.get("line_params", [])

    for i in range(MAX_LINES):
        if i < len(saved_lines):
            st.session_state[f"x1_{i}"] = saved_lines[i][0]
            st.session_state[f"y1_{i}"] = saved_lines[i][1]
            st.session_state[f"x2_{i}"] = saved_lines[i][2]
            st.session_state[f"y2_{i}"] = saved_lines[i][3]
        else:
            st.session_state[f"x1_{i}"] = DEFAULT_LINE["x1"]
            st.session_state[f"y1_{i}"] = DEFAULT_LINE["y1"]
            st.session_state[f"x2_{i}"] = DEFAULT_LINE["x2"]
            st.session_state[f"y2_{i}"] = DEFAULT_LINE["y2"]

    saved_source = saved.get("source_points", [])

    def _sp(idx, coord, fallback):
        try:
            return saved_source[idx][coord]
        except (IndexError, TypeError, KeyError):
            return fallback

    st.session_state.x_top_left = _sp(0, 0, DEFAULT_SOURCE_POINTS[0][0])
    st.session_state.y_top_left = _sp(0, 1, DEFAULT_SOURCE_POINTS[0][1])
    st.session_state.x_top_right = _sp(1, 0, DEFAULT_SOURCE_POINTS[1][0])
    st.session_state.y_top_right = _sp(1, 1, DEFAULT_SOURCE_POINTS[1][1])
    st.session_state.x_bottom_right = _sp(2, 0, DEFAULT_SOURCE_POINTS[2][0])
    st.session_state.y_bottom_right = _sp(2, 1, DEFAULT_SOURCE_POINTS[2][1])
    st.session_state.x_bottom_left = _sp(3, 0, DEFAULT_SOURCE_POINTS[3][0])
    st.session_state.y_bottom_left = _sp(3, 1, DEFAULT_SOURCE_POINTS[3][1])

    saved_target = saved.get("target_points", [])

    def _tp(idx, coord, fallback):
        try:
            return saved_target[idx][coord]
        except (IndexError, TypeError, KeyError):
            return fallback

    st.session_state.road_width = _tp(1, 0, DEFAULT_ROAD_WIDTH)
    st.session_state.road_length = _tp(2, 1, DEFAULT_ROAD_LENGTH)

# ==========================================
# PIPELINE API HELPERS
# ==========================================


def get_pipeline_status() -> dict:
    return backend_get("/pipeline/status")


def start_pipeline(payload: dict) -> dict:
    return backend_post("/pipeline/start", payload)


def stop_pipeline() -> dict:
    return backend_post("/pipeline/stop")


def get_statistics() -> dict:
    try:
        return backend_get("/pipeline/statistics")
    except Exception as e:
        logger.exception("Failed to fetch statistics")
        return {"error": str(e)}


def configure_speed(payload: dict) -> dict:
    return backend_post("/pipeline/config/speed", payload)


def configure_density(payload: dict) -> dict:
    return backend_post("/pipeline/config/density", payload)


def configure_flow(payload: dict) -> dict:
    return backend_post("/pipeline/config/flow", payload)


def get_preview_frame() -> Optional[np.ndarray]:
    """Fetch a raw preview JPEG frame from the backend (no annotations)."""
    return fetch_jpeg("preview_frame")


def get_live_frame() -> Optional[np.ndarray]:
    """Fetch the latest processed frame with bounding boxes, class labels, and tracker IDs."""
    return fetch_jpeg("live_frame")


def get_flow_frame() -> Optional[np.ndarray]:
    """Fetch the latest flow frame with detection line and in/out counters."""
    return fetch_jpeg("flow_frame")


def get_density_frame() -> Optional[np.ndarray]:
    """Fetch the latest density heatmap overlay frame."""
    return fetch_jpeg("density_frame")


# ==========================================
# DATABASE API HELPERS
# ==========================================


def get_database_statistics() -> dict:
    """Fetch aggregate database statistics (sessions, records, vehicles)."""
    return backend_get("/database/statistics")


def get_sessions() -> List[dict]:
    """Fetch the list of processing sessions."""
    return backend_get("/database/sessions")


def get_session_details(session_id: Any) -> dict:
    """Fetch details for a single processing session."""
    return backend_get(f"/database/session/{session_id}")


def get_vehicle_summary() -> List[dict]:
    """Fetch the per-vehicle summary table."""
    return backend_get("/database/vehicle_summary")


def get_detections(limit: int = 50) -> List[dict]:
    """Fetch the most recent raw detection records."""
    return backend_get("/database/detections", params={"limit": limit})


def get_debug_info() -> dict:
    """Fetch full backend debug/diagnostic info."""
    return backend_get("/pipeline/debug/full")


# ==========================================
# CHART HELPERS
# ==========================================


def render_vehicle_distribution_chart(flow_stats: dict) -> None:
    vd = flow_stats.get("vehicle_distribution", {})
    if vd:
        df = pd.DataFrame({"Type": list(vd.keys()), "Count": list(vd.values())})
        fig = px.pie(
            df, values="Count", names="Type", title="Vehicle Distribution", hole=0.3
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Vehicle Distribution — no data yet")


def render_vehicle_scene_chart(flow_stats: dict) -> None:
    vs = flow_stats.get("vehicle_scene", {})
    if vs:
        df = pd.DataFrame({"Time": list(vs.keys()), "Count": list(vs.values())})
        fig = px.line(df, x="Time", y="Count", title="Vehicles In Scene", markers=True)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Vehicles In Scene — no data yet")


def render_class_speed_chart(speed_stats: dict) -> None:
    cs = speed_stats.get("class_speed", {})
    if cs:
        rows = []
        for vtype, val in cs.items():
            if isinstance(val, (list, tuple)) and len(val) == 2 and val[1] > 0:
                avg = val[0] / val[1]
            elif isinstance(val, (int, float)):
                avg = val
            else:
                avg = 0
            rows.append({"Type": vtype, "Avg. Speed": avg})
        df = pd.DataFrame(rows)
        fig = px.bar_polar(
            df,
            r="Avg. Speed",
            theta="Type",
            color="Type",
            title="Average Speed by Vehicle Type",
            labels={"Avg. Speed": "Speed (km/h)"},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Class Speed — no data yet")


def render_time_speed_chart(speed_stats: dict) -> None:
    ts = speed_stats.get("time_speed", {})
    if ts:
        df = pd.DataFrame({"Time": list(ts.keys()), "Avg. Speed": list(ts.values())})
        fig = px.area(
            df,
            x="Time",
            y="Avg. Speed",
            title="Average Speed Over Time",
            labels={"Time": "Time (s)", "Avg. Speed": "Avg. Speed (km/h)"},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Time Speed — no data yet")


# ==========================================
# SESSION RESET HELPER
# ==========================================


def get_reset_keys() -> List[str]:
    """Session-state keys cleared when the user clicks 'Reset Settings'."""
    keys = [
        "model_set",
        "mode",
        "confidence_threshold",
        "iou_threshold",
        "device",
        "classes_to_detect",
        "num_lines",
        "road_width",
        "road_length",
        "x_top_left",
        "y_top_left",
        "x_top_right",
        "y_top_right",
        "x_bottom_right",
        "y_bottom_right",
        "x_bottom_left",
        "y_bottom_left",
    ]
    for i in range(MAX_LINES):
        keys.extend([f"x1_{i}", f"y1_{i}", f"x2_{i}", f"y2_{i}"])
    return keys


def reset_session_settings() -> None:
    """Clear all configurable session-state keys back to their defaults."""
    for key in get_reset_keys():
        st.session_state.pop(key, None)


# ==========================================
# TITLE + SESSION STATE DEBUG
# ==========================================

st.title("Gate Surveillance Backend Dashboard")
st.write("SESSION ID:", id(st.session_state))

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Pipeline Control",
        "Analysis Dashboard",
        "Live Analytics",
        "Flow Analytics",
        "Speed Analytics",
        "Density Analytics",
        "Database",
        "Vehicle Summary",
        "Detections",
        "Debug",
    ],
)

# =====================================================
# PIPELINE CONTROL
# =====================================================

if page == "Pipeline Control":
    st.header("Pipeline Control")

    # --- Model Settings ---
    with st.expander("Model Settings", expanded=True):
        st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="confidence_threshold",
        )
        st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="iou_threshold",
        )
        st.selectbox("Model Set", MODEL_SET_OPTIONS, key="model_set")
        st.selectbox("Mode", MODE_OPTIONS, key="mode")
        st.selectbox("Device", DEVICE_OPTIONS, key="device")

    # --- Detector Settings ---
    with st.expander("Detector Settings", expanded=True):
        st.write("Configure Detection Dashboard")
        st.multiselect("Classes To Detect", CLASS_OPTIONS, key="classes_to_detect")

    # --- Flow Estimation Settings ---
    with st.expander("Flow Estimation Settings", expanded=False):
        st.info(
            "Select the number of detection lines and configure "
            "the lines to count the objects crossing the line."
        )

        left, right = st.columns([1, 1])

        with left:
            st.number_input(
                "Number of detection Lines",
                min_value=1,
                max_value=MAX_LINES,
                step=1,
                key="num_lines",
            )

            line_params = []

            for i in range(st.session_state.num_lines):
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.number_input("X1", key=f"x1_{i}")
                with c2:
                    st.number_input("Y1", key=f"y1_{i}")
                with c3:
                    st.number_input("X2", key=f"x2_{i}")
                with c4:
                    st.number_input("Y2", key=f"y2_{i}")

                line_params.append(
                    [
                        st.session_state[f"x1_{i}"],
                        st.session_state[f"y1_{i}"],
                        st.session_state[f"x2_{i}"],
                        st.session_state[f"y2_{i}"],
                    ]
                )

        with right:
            frame = get_preview_frame()

            if frame is not None:
                for i in range(st.session_state.num_lines):
                    cv2.line(
                        frame,
                        (
                            int(st.session_state[f"x1_{i}"]),
                            int(st.session_state[f"y1_{i}"]),
                        ),
                        (
                            int(st.session_state[f"x2_{i}"]),
                            int(st.session_state[f"y2_{i}"]),
                        ),
                        (0, 255, 0),
                        3,
                    )

                    cv2.putText(
                        frame,
                        f"L{i + 1}",
                        (
                            int(st.session_state[f"x1_{i}"]),
                            int(st.session_state[f"y1_{i}"]) - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

            else:
                st.warning("Preview frame unavailable.")

    # --- Speed Estimation Settings ---
    with st.expander("Speed Estimation Settings", expanded=False):
        st.info(
            "Select the speed estimation region and configure "
            "the region to estimate the speed of the objects."
        )

        left, right = st.columns([1, 1])

        with left:
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)

            with r1c1:
                st.slider("X Top Left", -1000, 1000, key="x_top_left")
            with r1c2:
                st.slider("Y Top Left", -1000, 1000, key="y_top_left")
            with r1c3:
                st.slider("X Top Right", -1000, 1000, key="x_top_right")
            with r1c4:
                st.slider("Y Top Right", -1000, 1000, key="y_top_right")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)

            with r2c1:
                st.slider("X Bottom Left", -1000, 1000, key="x_bottom_left")
            with r2c2:
                st.slider("Y Bottom Left", -1000, 1000, key="y_bottom_left")
            with r2c3:
                st.slider("X Bottom Right", -1000, 1000, key="x_bottom_right")
            with r2c4:
                st.slider("Y Bottom Right", -1000, 1000, key="y_bottom_right")

            r3c1, r3c2 = st.columns(2)

            with r3c1:
                st.slider("Breadth of Road in Meters", 1, 100, key="road_width")
            with r3c2:
                st.slider("Length of Road in Meters", 1, 100, key="road_length")

        with right:
            frame = get_preview_frame()

            if frame is not None:
                pts = np.array(
                    [
                        [
                            st.session_state["x_top_left"],
                            st.session_state["y_top_left"],
                        ],
                        [
                            st.session_state["x_top_right"],
                            st.session_state["y_top_right"],
                        ],
                        [
                            st.session_state["x_bottom_right"],
                            st.session_state["y_bottom_right"],
                        ],
                        [
                            st.session_state["x_bottom_left"],
                            st.session_state["y_bottom_left"],
                        ],
                    ],
                    dtype=np.int32,
                )

                cv2.polylines(frame, [pts], True, (255, 0, 255), 3)

                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

            else:
                st.warning("Preview frame unavailable.")

    # --- Session Control ---
    with st.expander("Session Control", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Start Session"):
                payload = {
                    "rtsp_url": RTSP_URL,
                    "camera_type": st.session_state.model_set,
                    "det_mode": st.session_state.mode,
                    "frame_rate": DEFAULT_FRAME_RATE,
                    "confidence_threshold": st.session_state.confidence_threshold,
                    "iou_threshold": st.session_state.iou_threshold,
                    "device": st.session_state.device,
                    "classes_to_detect": st.session_state.classes_to_detect,
                    "num_lines": st.session_state.num_lines,
                    "line_params": line_params,
                    "source_points": [
                        [st.session_state.x_top_left, st.session_state.y_top_left],
                        [st.session_state.x_top_right, st.session_state.y_top_right],
                        [
                            st.session_state.x_bottom_right,
                            st.session_state.y_bottom_right,
                        ],
                        [
                            st.session_state.x_bottom_left,
                            st.session_state.y_bottom_left,
                        ],
                    ],
                    "target_points": [
                        [0, 0],
                        [st.session_state.road_width, 0],
                        [st.session_state.road_width, st.session_state.road_length],
                        [0, st.session_state.road_length],
                    ],
                }

                speed_payload = {
                    "source_points": payload["source_points"],
                    "target_points": payload["target_points"],
                }

                flow_payload = {"line_params": line_params}

                st.json(payload)
                result = start_pipeline(payload)
                if result.get("status") == "started":
                    st.session_state.pipeline_started = True
                st.write(result)

                st.write("Speed Payload:")
                st.json(speed_payload)
                st.write(configure_speed(speed_payload))

                st.write(
                    configure_density(
                        {
                            "width": DEFAULT_FRAME_WIDTH,
                            "height": DEFAULT_FRAME_HEIGHT,
                        }
                    )
                )

                st.write("Flow Payload:")
                st.json(flow_payload)
                st.write(configure_flow(flow_payload))

        with col2:
            if st.button("Stop Session"):
                result = stop_pipeline()
                if result.get("status") == "stopped":
                    st.session_state.pipeline_started = False
                st.write(result)

        with col3:
            if st.button("Reset Settings"):
                reset_session_settings()
                st.rerun()

    # --- Pipeline Status ---
    with st.expander("Pipeline Status", expanded=True):
        try:
            status = get_pipeline_status()
            if status.get("pipeline") == "ready":
                st.success("Pipeline Ready")
            else:
                st.warning("Pipeline Not Ready")
        except Exception as e:
            st.error(str(e))

# =====================================================
# ANALYSIS DASHBOARD
# =====================================================

if page == "Analysis Dashboard":
    st.header("Analysis Dashboard")

    st_autorefresh(interval=1000, key="analysis_refresh")

    try:
        status = get_pipeline_status()
        if status.get("pipeline") == "ready":
            st.success("Pipeline Ready")
        else:
            st.warning("Pipeline Not Ready — start it from Pipeline Control.")
    except Exception as e:
        st.error(f"Backend connection failed: {e}")

    stats = get_statistics()
    flow_stats = stats.get("flow_statistics", {})
    speed_stats = stats.get("speed_statistics", {})
    density_stats = stats.get("density_statistics", {})

    frame_index = stats.get("frame_index", 0)
    st.info(f"Frame Index: {frame_index}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Detections", stats.get("detections", 0))

    with col2:
        st.metric("Raw Detection Records", stats.get("raw_detection_records", 0))

    with col3:
        st.metric("Unique Vehicles", stats.get("unique_vehicles", 0))

    # -------------------------------------------------------
    # VEHICLE FLOW ESTIMATION — side by side
    # -------------------------------------------------------
    st.subheader("Vehicle Flow Estimation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Live Detection**")
        st.caption("Bounding boxes, tracker IDs, and class labels.")

        live_frame = get_live_frame()

        if live_frame is not None:
            st.image(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info(
                "No live frame available — start the pipeline from Pipeline Control."
            )

    with col2:
        st.markdown("**Flow Estimation**")
        st.caption("Detection line with live In / Out counters.")

        flow_frame = get_flow_frame()

        if flow_frame is not None:
            st.image(cv2.cvtColor(flow_frame, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info(
                "No flow frame available — start the pipeline from Pipeline Control."
            )

    # -------------------------------------------------------
    # FLOW CHARTS
    # -------------------------------------------------------
    with st.expander("Vehicle Flow Charts", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            render_vehicle_distribution_chart(flow_stats)
        with col2:
            render_vehicle_scene_chart(flow_stats)

    # -------------------------------------------------------
    # SPEED CHARTS
    # -------------------------------------------------------
    with st.expander("Vehicle Speed Charts", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            render_class_speed_chart(speed_stats)
        with col2:
            render_time_speed_chart(speed_stats)

    # -------------------------------------------------------
    # DENSITY SUMMARY
    # -------------------------------------------------------
    with st.expander("Vehicle Density", expanded=False):
        col1, col2 = st.columns(2)
        col1.metric("Heatmap Available", density_stats.get("heatmap_available", False))
        col2.write(f"Heatmap Shape: {density_stats.get('heatmap_shape', None)}")

    # -------------------------------------------------------
    # DIAGNOSTICS
    # -------------------------------------------------------
    with st.expander("Diagnostics — Flow vs Speed Classes", expanded=False):
        flow_classes = set(flow_stats.get("vehicle_distribution", {}).keys())
        speed_classes = set(speed_stats.get("class_speed", {}).keys())

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Flow classes** (vehicle_distribution)")
            st.write(sorted(flow_classes) if flow_classes else "— none —")
        with col2:
            st.write("**Speed classes** (class_speed)")
            st.write(sorted(speed_classes) if speed_classes else "— none —")

        only_in_speed = speed_classes - flow_classes
        only_in_flow = flow_classes - speed_classes

        if only_in_speed:
            st.warning(
                "Classes only in speed_statistics (no matching flow crossing yet): "
                f"{sorted(only_in_speed)}"
            )
        if only_in_flow:
            st.warning(
                "Classes only in flow_statistics (no speed estimate yet): "
                f"{sorted(only_in_flow)}"
            )
        if not only_in_speed and not only_in_flow and flow_classes:
            st.success("Flow and Speed class sets match.")

        st.write("Raw flow_statistics:")
        st.json(flow_stats)
        st.write("Raw speed_statistics:")
        st.json(speed_stats)

# =====================================================
# LIVE ANALYTICS
# =====================================================

if page == "Live Analytics":
    st.header("Live Analytics")

    st_autorefresh(interval=1000, key="live_analytics_refresh")

    stats = get_statistics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Frame Index", stats.get("frame_index", 0))
    col2.metric("Current Detections", stats.get("detections", 0))
    col3.metric("Raw Detection Records", stats.get("raw_detection_records", 0))
    col4.metric("Unique Vehicles", stats.get("unique_vehicles", 0))

# =====================================================
# FLOW ANALYTICS
# =====================================================

if page == "Flow Analytics":
    st.header("Flow Analytics")

    st_autorefresh(interval=1000, key="flow_analytics_refresh")

    flow_stats = get_statistics().get("flow_statistics", {})

    col1, col2 = st.columns(2)
    with col1:
        render_vehicle_distribution_chart(flow_stats)
    with col2:
        render_vehicle_scene_chart(flow_stats)

    with st.expander("Raw Data"):
        st.json(flow_stats)

# =====================================================
# SPEED ANALYTICS
# =====================================================

if page == "Speed Analytics":
    st.header("Speed Analytics")

    st_autorefresh(interval=1000, key="speed_analytics_refresh")

    speed_stats = get_statistics().get("speed_statistics", {})

    col1, col2 = st.columns(2)
    with col1:
        render_class_speed_chart(speed_stats)
    with col2:
        render_time_speed_chart(speed_stats)

    with st.expander("Raw Data"):
        st.json(speed_stats)

# =====================================================
# DENSITY ANALYTICS
# =====================================================

if page == "Density Analytics":
    st.header("Density Analytics")

    st_autorefresh(interval=1000, key="density_analytics_refresh")

    density_stats = get_statistics().get("density_statistics", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Live Camera View**")
        st.caption("Current camera frame.")

        live_frame = get_live_frame()

        if live_frame is not None:
            st.image(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info(
                "No live frame available — start the pipeline from Pipeline Control."
            )

    with col2:
        st.markdown("**Density Heatmap**")
        st.caption("Vehicle density overlay.")

        density_frame = get_density_frame()

        if density_frame is not None:
            st.image(cv2.cvtColor(density_frame, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info(
                "No density frame available — start the pipeline from Pipeline Control."
            )

    with st.expander("Density Statistics", expanded=False):
        col1, col2 = st.columns(2)
        col1.metric("Heatmap Available", density_stats.get("heatmap_available", False))
        col2.write(f"Heatmap Shape: {density_stats.get('heatmap_shape', None)}")

    with st.expander("Raw Data"):
        st.json(density_stats)

# =====================================================
# DATABASE
# =====================================================

if page == "Database":
    st.header("Database Statistics")
    try:
        stats = get_database_statistics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sessions", stats["total_sessions"])
        col2.metric("Raw Detection Records", stats["total_raw_detection_records"])
        col3.metric("Unique Vehicles", stats["total_unique_vehicles"])
    except Exception as e:
        st.error(f"Backend connection failed: {e}")

    st.header("Sessions")
    sessions: List[dict] = []
    try:
        sessions = get_sessions()
        st.dataframe(pd.DataFrame(sessions), width="stretch")
    except Exception as e:
        st.error(f"Failed to load sessions: {e}")

    st.header("Session Details")
    try:
        # Reuses the `sessions` list fetched above instead of fetching it again.
        session_ids = [s["id"] for s in sessions]
        if session_ids:
            selected = st.selectbox("Select Session", session_ids)
            details = get_session_details(selected)
            st.subheader(f"Session {selected}")
            st.write("Camera Type:", details["camera_type"])
            st.write("Status:", details["status"])
            st.write("Raw Detection Records:", details["total_raw_detection_records"])
            st.write("Unique Vehicles:", details["total_unique_vehicles"])
            st.write("Average Speed:", details["average_speed"])
            st.write("Vehicle Counts:")
            st.json(details["vehicle_counts"])
    except Exception as e:
        st.error(f"Failed to load session details: {e}")

# =====================================================
# VEHICLE SUMMARY
# =====================================================

if page == "Vehicle Summary":
    st.header("Vehicle Summary")

    try:
        vehicles = get_vehicle_summary()
        df = pd.DataFrame(vehicles)
        st.write(f"Unique Vehicles: {len(df)}")
        st.dataframe(df, width="stretch")
    except Exception as e:
        st.error(f"Failed to load vehicle summary: {e}")

# =====================================================
# DETECTIONS
# =====================================================

if page == "Detections":
    st.header("Latest Detections")
    try:
        detections = get_detections(limit=50)
        st.dataframe(pd.DataFrame(detections), width="stretch")
    except Exception as e:
        st.error(f"Failed to load detections: {e}")

# =====================================================
# DEBUG
# =====================================================

if page == "Debug":
    st.header("Debug Information")
    try:
        st.json(get_debug_info())
    except Exception as e:
        st.error(str(e))
