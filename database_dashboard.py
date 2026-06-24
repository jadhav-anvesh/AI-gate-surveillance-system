import cv2
import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

BACKEND_URL = "http://127.0.0.1:8000"

# ==========================================
# BACKEND CONFIG LOADER
# ==========================================

def load_config_from_backend():

    print("\n========== LOADING CONFIG ==========")

    try:

        response = requests.get(
            f"{BACKEND_URL}/pipeline/current_config",
            timeout=2
        )

        print("STATUS:", response.status_code)
        print("DATA:",   response.text)

        if response.status_code == 200:
            return response.json()

    except Exception as e:

        print("CONFIG ERROR:", e)

    return {}

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Gate Surveillance Backend Dashboard", layout="wide")

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
    "vehicle fallback"
]
# ==========================================
# UNCONDITIONAL SESSION STATE DEFAULTS
# ==========================================

def ensure_session_defaults():

    st.session_state.setdefault("model_set", "Trained Model")
    st.session_state.setdefault("mode", "YOLO-N")
    st.session_state.setdefault("confidence_threshold", 0.30)
    st.session_state.setdefault("iou_threshold", 0.50)

    st.session_state.setdefault("device", "GPU")
    st.session_state.setdefault("classes_to_detect", CLASS_OPTIONS)
    st.session_state.setdefault("pipeline_started", False)

    st.session_state.setdefault("num_lines", 1)
    for i in range(10):
        st.session_state.setdefault(f"x1_{i}", 0)
        st.session_state.setdefault(f"y1_{i}", 100)
        st.session_state.setdefault(f"x2_{i}", 640)
        st.session_state.setdefault(f"y2_{i}", 180)

    st.session_state.setdefault("x_top_left", 344)
    st.session_state.setdefault("y_top_left", 83)
    st.session_state.setdefault("x_top_right", 516)
    st.session_state.setdefault("y_top_right", 99)
    st.session_state.setdefault("x_bottom_right", 557)
    st.session_state.setdefault("y_bottom_right", 411)
    st.session_state.setdefault("x_bottom_left", -236)
    st.session_state.setdefault("y_bottom_left", 218)

    st.session_state.setdefault("road_width", 19)
    st.session_state.setdefault("road_length", 19)

ensure_session_defaults()

# ==========================================
# SESSION STATE INIT
# ==========================================

if "session_state_loaded" not in st.session_state:

    st.session_state.session_state_loaded = True

    saved = load_config_from_backend()

    st.session_state.model_set            = saved.get("camera_type",           "Trained Model")
    st.session_state.mode                 = saved.get("det_mode",               "YOLO-N")
    st.session_state.confidence_threshold = saved.get("confidence_threshold",   0.30)
    st.session_state.iou_threshold        = saved.get("iou_threshold",          0.50)

    st.session_state.device               = saved.get("device",                 "GPU")
    st.session_state.classes_to_detect    = saved.get("classes_to_detect",      CLASS_OPTIONS)

    st.session_state.pipeline_started = False

    st.session_state.num_lines = saved.get("num_lines", 1)

    default_lines = [
        {"x1": 0, "y1": 100, "x2": 640, "y2": 180}
    ] * 10

    saved_lines = saved.get("line_params", [])

    for i in range(10):
        if i < len(saved_lines):
            st.session_state[f"x1_{i}"] = saved_lines[i][0]
            st.session_state[f"y1_{i}"] = saved_lines[i][1]
            st.session_state[f"x2_{i}"] = saved_lines[i][2]
            st.session_state[f"y2_{i}"] = saved_lines[i][3]
        else:
            st.session_state[f"x1_{i}"] = default_lines[i]["x1"]
            st.session_state[f"y1_{i}"] = default_lines[i]["y1"]
            st.session_state[f"x2_{i}"] = default_lines[i]["x2"]
            st.session_state[f"y2_{i}"] = default_lines[i]["y2"]

    saved_source = saved.get("source_points", [])

    def _sp(idx, coord, fallback):
        try:
            return saved_source[idx][coord]
        except (IndexError, TypeError, KeyError):
            return fallback

    st.session_state.x_top_left     = _sp(0, 0,  344)
    st.session_state.y_top_left     = _sp(0, 1,   83)
    st.session_state.x_top_right    = _sp(1, 0,  516)
    st.session_state.y_top_right    = _sp(1, 1,   99)
    st.session_state.x_bottom_right = _sp(2, 0,  557)
    st.session_state.y_bottom_right = _sp(2, 1,  411)
    st.session_state.x_bottom_left  = _sp(3, 0, -236)
    st.session_state.y_bottom_left  = _sp(3, 1,  218)

    saved_target = saved.get("target_points", [])

    def _tp(idx, coord, fallback):
        try:
            return saved_target[idx][coord]
        except (IndexError, TypeError, KeyError):
            return fallback

    st.session_state.road_width  = _tp(1, 0, 19)
    st.session_state.road_length = _tp(2, 1, 19)

# ==========================================
# API HELPERS
# ==========================================

def get_pipeline_status():
    return requests.get(f"{BACKEND_URL}/pipeline/status").json()

def start_pipeline(payload):
    return requests.post(f"{BACKEND_URL}/pipeline/start", json=payload).json()

def stop_pipeline():
    return requests.post(f"{BACKEND_URL}/pipeline/stop").json()

def get_statistics():
    try:
        return requests.get(f"{BACKEND_URL}/pipeline/statistics").json()
    except Exception as e:
        return {"error": str(e)}

def configure_speed(payload):
    return requests.post(f"{BACKEND_URL}/pipeline/config/speed", json=payload).json()

def configure_density(payload):
    return requests.post(f"{BACKEND_URL}/pipeline/config/density", json=payload).json()

def configure_flow(payload):
    return requests.post(f"{BACKEND_URL}/pipeline/config/flow", json=payload).json()

def get_preview_frame():
    """Fetch a raw preview JPEG frame from the backend (no annotations)."""
    try:
        r = requests.get(f"{BACKEND_URL}/pipeline/preview_frame", timeout=2)
        if r.headers.get("content-type") == "image/jpeg":
            arr = np.asarray(bytearray(r.content), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None

def get_live_frame():
    """Fetch the latest processed frame with bounding boxes, class labels, and tracker IDs."""
    try:
        r = requests.get(f"{BACKEND_URL}/pipeline/live_frame", timeout=2)
        if r.headers.get("content-type") == "image/jpeg":
            arr = np.asarray(bytearray(r.content), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None

def get_flow_frame():
    """Fetch the latest flow frame with detection line and in/out counters."""
    try:
        r = requests.get(f"{BACKEND_URL}/pipeline/flow_frame", timeout=2)
        if r.headers.get("content-type") == "image/jpeg":
            arr = np.asarray(bytearray(r.content), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None

def get_density_frame():
    """Fetch the latest density heatmap overlay frame."""
    try:
        r = requests.get(f"{BACKEND_URL}/pipeline/density_frame", timeout=2)
        if r.headers.get("content-type") == "image/jpeg":
            arr = np.asarray(bytearray(r.content), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None

# ==========================================
# CHART HELPERS
# ==========================================

def render_vehicle_distribution_chart(flow_stats):
    vd = flow_stats.get("vehicle_distribution", {})
    if vd:
        df = pd.DataFrame({"Type": list(vd.keys()), "Count": list(vd.values())})
        fig = px.pie(df, values="Count", names="Type", title="Vehicle Distribution", hole=0.3)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Vehicle Distribution — no data yet")

def render_vehicle_scene_chart(flow_stats):
    vs = flow_stats.get("vehicle_scene", {})
    if vs:
        df = pd.DataFrame({"Time": list(vs.keys()), "Count": list(vs.values())})
        fig = px.line(df, x="Time", y="Count", title="Vehicles In Scene", markers=True)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Vehicles In Scene — no data yet")

def render_class_speed_chart(speed_stats):
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
            df, r="Avg. Speed", theta="Type", color="Type",
            title="Average Speed by Vehicle Type",
            labels={"Avg. Speed": "Speed (km/h)"}
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Class Speed — no data yet")

def render_time_speed_chart(speed_stats):
    ts = speed_stats.get("time_speed", {})
    if ts:
        df = pd.DataFrame({"Time": list(ts.keys()), "Avg. Speed": list(ts.values())})
        fig = px.area(
            df, x="Time", y="Avg. Speed",
            title="Average Speed Over Time",
            labels={"Time": "Time (s)", "Avg. Speed": "Avg. Speed (km/h)"}
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Time Speed — no data yet")

# ==========================================
# TITLE + SESSION STATE DEBUG
# ==========================================

st.title("Gate Surveillance Backend Dashboard")
st.write("SESSION ID:", id(st.session_state))

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

page = st.sidebar.radio("Navigation", [
    "Pipeline Control",
    "Analysis Dashboard",
    "Live Analytics",
    "Flow Analytics",
    "Speed Analytics",
    "Density Analytics",
    "Database",
    "Vehicle Summary",
    "Detections",
    "Debug"
])

# =====================================================
# PIPELINE CONTROL
# =====================================================

if page == "Pipeline Control":

    st.header("Pipeline Control")

    MODEL_SET_OPTIONS = ["Trained Model"]
    MODE_OPTIONS      = ["YOLO-N", "RF-DETR"]
    DEVICE_OPTIONS    = ["GPU"]
    CLASS_OPTIONS = ["animal", "autorickshaw", "bicycle", "bus", "car", "caravan", "motorcycle", "person", "rider", "traffic light", "traffic sign", "trailer", "train", "truck", "vehicle fallback"]

    # --- Model Settings ---
    with st.expander("Model Settings", expanded=True):

        st.slider(
            "Confidence Threshold",
            min_value=0.0, max_value=1.0, step=0.01,
            key="confidence_threshold"
        )
        st.slider(
            "IoU Threshold",
            min_value=0.0, max_value=1.0, step=0.01,
            key="iou_threshold"
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
                max_value=10,
                step=1,
                key="num_lines"
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

                line_params.append([
                    st.session_state[f"x1_{i}"],
                    st.session_state[f"y1_{i}"],
                    st.session_state[f"x2_{i}"],
                    st.session_state[f"y2_{i}"]
                ])

        with right:

            try:

                frame_response = requests.get(f"{BACKEND_URL}/pipeline/preview_frame")

                if frame_response.headers.get("content-type") == "image/jpeg":

                    frame_bytes = np.asarray(bytearray(frame_response.content), dtype=np.uint8)
                    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

                    for i in range(st.session_state.num_lines):

                        cv2.line(
                            frame,
                            (int(st.session_state[f"x1_{i}"]), int(st.session_state[f"y1_{i}"])),
                            (int(st.session_state[f"x2_{i}"]), int(st.session_state[f"y2_{i}"])),
                            (0, 255, 0),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"L{i+1}",
                            (int(st.session_state[f"x1_{i}"]), int(st.session_state[f"y1_{i}"]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

                else:
                    st.warning(frame_response.text)

            except Exception as e:
                st.warning(f"Frame unavailable: {e}")

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

            try:

                frame_response = requests.get(f"{BACKEND_URL}/pipeline/preview_frame")

                if frame_response.headers.get("content-type") == "image/jpeg":

                    frame_bytes = np.asarray(bytearray(frame_response.content), dtype=np.uint8)
                    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

                    pts = np.array([
                        [st.session_state["x_top_left"],     st.session_state["y_top_left"]],
                        [st.session_state["x_top_right"],    st.session_state["y_top_right"]],
                        [st.session_state["x_bottom_right"], st.session_state["y_bottom_right"]],
                        [st.session_state["x_bottom_left"],  st.session_state["y_bottom_left"]]
                    ], dtype=np.int32)

                    cv2.polylines(frame, [pts], True, (255, 0, 255), 3)

                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

                else:
                    st.warning(frame_response.text)

            except Exception as e:
                st.warning(f"Frame unavailable: {e}")

    # --- Session Control ---
    with st.expander("Session Control", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Start Session"):

                payload = {
                    "rtsp_url":             "rtsp://arjun.badola:AB%23ai2025@10.0.102.54:554/",
                    "camera_type":          st.session_state.model_set,
                    "det_mode":             st.session_state.mode,
                    "frame_rate":           10,
                    "confidence_threshold": st.session_state.confidence_threshold,
                    "iou_threshold":        st.session_state.iou_threshold,
                    "device":               st.session_state.device,
                    "classes_to_detect":    st.session_state.classes_to_detect,
                    "num_lines":            st.session_state.num_lines,
                    "line_params":          line_params,
                    "source_points": [
                        [st.session_state.x_top_left,     st.session_state.y_top_left],
                        [st.session_state.x_top_right,    st.session_state.y_top_right],
                        [st.session_state.x_bottom_right, st.session_state.y_bottom_right],
                        [st.session_state.x_bottom_left,  st.session_state.y_bottom_left]
                    ],
                    "target_points": [
                        [0, 0],
                        [st.session_state.road_width, 0],
                        [st.session_state.road_width, st.session_state.road_length],
                        [0, st.session_state.road_length]
                    ]
                }

                speed_payload = {
                    "source_points": payload["source_points"],
                    "target_points": payload["target_points"]
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

                st.write(configure_density({"width": 640, "height": 480}))

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

                keys_to_delete = [

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
                    "y_bottom_left"
                ]

                for i in range(10):

                    keys_to_delete.extend([

                        f"x1_{i}",
                        f"y1_{i}",
                        f"x2_{i}",
                        f"y2_{i}"
                    ])

                for key in keys_to_delete:

                    if key in st.session_state:
                        del st.session_state[key]

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

    # Auto-refresh every 1 second
    st_autorefresh(interval=1000, key="analysis_refresh")

    try:
        status = get_pipeline_status()
        if status.get("pipeline") == "ready":
            st.success("Pipeline Ready")
        else:
            st.warning("Pipeline Not Ready — start it from Pipeline Control.")
    except Exception as e:
        st.error(f"Backend connection failed: {e}")

    stats         = get_statistics()
    flow_stats    = stats.get("flow_statistics", {})
    speed_stats   = stats.get("speed_statistics", {})
    density_stats = stats.get("density_statistics", {})

    frame_index = stats.get("frame_index", 0)
    st.info(f"Frame Index: {frame_index}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Detections",
            stats.get("detections", 0)
        )

    with col2:
        st.metric(
            "Raw Detection Records",
            stats.get("raw_detection_records", 0)
        ) 

    with col3:
        st.metric(
            "Unique Vehicles",
            stats.get("unique_vehicles", 0)
        )
    # -------------------------------------------------------
    # VEHICLE FLOW ESTIMATION — side by side
    # Left: Live Detection (bounding boxes from live_frame)
    # Right: Flow Estimation (plain preview_frame — no bbox)
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
            st.info("No live frame available — start the pipeline from Pipeline Control.")

    with col2:
        st.markdown("**Flow Estimation**")
        st.caption("Detection line with live In / Out counters.")

        flow_frame = get_flow_frame()

        if flow_frame is not None:
            st.image(cv2.cvtColor(flow_frame, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info("No flow frame available — start the pipeline from Pipeline Control.")

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

        flow_classes  = set(flow_stats.get("vehicle_distribution", {}).keys())
        speed_classes = set(speed_stats.get("class_speed", {}).keys())

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Flow classes** (vehicle_distribution)")
            st.write(sorted(flow_classes) if flow_classes else "— none —")
        with col2:
            st.write("**Speed classes** (class_speed)")
            st.write(sorted(speed_classes) if speed_classes else "— none —")

        only_in_speed = speed_classes - flow_classes
        only_in_flow  = flow_classes - speed_classes

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

    # Auto-refresh every 1 second
    st_autorefresh(interval=1000, key="live_analytics_refresh")

    stats = get_statistics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Frame Index",
        stats.get("frame_index", 0)
    )

    col2.metric(
        "Current Detections",
        stats.get("detections", 0)
    )

    col3.metric(
        "Raw Detection Records",
        stats.get("raw_detection_records", 0)
    )
   
    col4.metric(
        "Unique Vehicles",
        stats.get("unique_vehicles", 0)
    )
# =====================================================
# FLOW ANALYTICS
# =====================================================

if page == "Flow Analytics":

    st.header("Flow Analytics")

    # Auto-refresh every 1 second
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

    # Auto-refresh every 1 second
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

    # Auto-refresh every 1 second
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
            st.info("No live frame available — start the pipeline from Pipeline Control.")

    with col2:
        st.markdown("**Density Heatmap**")
        st.caption("Vehicle density overlay.")

        density_frame = get_density_frame()

        if density_frame is not None:
            st.image(cv2.cvtColor(density_frame, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info("No density frame available — start the pipeline from Pipeline Control.")

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
        stats = requests.get(f"{BACKEND_URL}/database/statistics").json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sessions",   stats["total_sessions"])
        col2.metric("Raw Detection Records", stats["total_raw_detection_records"])
        col3.metric("Unique Vehicles", stats["total_unique_vehicles"])

    except Exception as e:
        st.error(f"Backend connection failed: {e}")

    st.header("Sessions")
    try:
        sessions = requests.get(f"{BACKEND_URL}/database/sessions").json()
        st.dataframe(pd.DataFrame(sessions), width="stretch")
    except Exception as e:
        st.error(f"Failed to load sessions: {e}")

    st.header("Session Details")
    try:
        sessions    = requests.get(f"{BACKEND_URL}/database/sessions").json()
        session_ids = [s["id"] for s in sessions]
        if session_ids:
            selected = st.selectbox("Select Session", session_ids)
            details  = requests.get(f"{BACKEND_URL}/database/session/{selected}").json()
            st.subheader(f"Session {selected}")
            st.write("Camera Type:",      details["camera_type"])
            st.write("Status:",           details["status"])
            st.write("Raw Detection Records:", details["total_raw_detection_records"])
            st.write("Unique Vehicles:", details["total_unique_vehicles"])
            st.write("Average Speed:",    details["average_speed"])
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

        vehicles = requests.get(
            f"{BACKEND_URL}/database/vehicle_summary"
        ).json()

        df = pd.DataFrame(
            vehicles
        )

        st.write(
            f"Unique Vehicles: {len(df)}"
        )

        st.dataframe(
            df,
            width="stretch"
        )

    except Exception as e:

        st.error(
            f"Failed to load vehicle summary: {e}"
        )




# =====================================================
# DETECTIONS
# =====================================================

if page == "Detections":

    st.header("Latest Detections")
    try:
        detections = requests.get(f"{BACKEND_URL}/database/detections?limit=50").json()
        st.dataframe(pd.DataFrame(detections), width="stretch")
    except Exception as e:
        st.error(f"Failed to load detections: {e}")

# =====================================================
# DEBUG
# =====================================================

if page == "Debug":

    st.header("Debug Information")
    try:
        st.json(requests.get(f"{BACKEND_URL}/pipeline/debug/full").json())
    except Exception as e:
        st.error(str(e))
