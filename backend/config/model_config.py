
MODEL_MAP_SET_A = {
    "YOLO-N": ("yolo", "backend/models/yolov10n_best.pt"),
    "RF-DETR": (
        "rfdetr",
        "backend/models/rfdetr_best.pth"
    ),
}

SET_A_CLS_MAP = {
    "animal": 0,
    "autorickshaw": 1,
    "bicycle": 2,
    "bus": 3,
    "car": 4,
    "caravan": 5,
    "motorcycle": 6,
    "person": 7,
    "rider": 8,
    "traffic light": 9,
    "traffic sign": 10,
    "trailer": 11,
    "train": 12,
    "truck": 13,
    "vehicle fallback": 14,
}

MODEL_SETS = {
    "Trained Model": (MODEL_MAP_SET_A, SET_A_CLS_MAP)
}

MODEL_MAPS = {
    "Trained Model": MODEL_MAP_SET_A
}

CLASS_MAPS = {
    "Trained Model": SET_A_CLS_MAP
}
