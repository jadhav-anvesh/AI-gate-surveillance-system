import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Load models
plate_model = YOLO("/mnt/zoneA/projects/gate/archive/streamlit_app_upstream/streamlit_app_og/backend/models/best.pt")
reader = easyocr.Reader(['en'])

# Load y
image_path = "/home/project/Storage/zoneA/summer_project/tanmai/test2.png"
img = cv2.imread(image_path)

if img is None:
    print("Could not read image. Check the path.")
else:
    results = plate_model(img)

    found = False
    for result in results:
        for box in result.boxes:
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            print(f"Plate detected at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")

            plate_crop = img[y1:y2, x1:x2]
            cv2.imwrite("test_plate_crop.jpg", plate_crop)

            # Version A: Color, just upscaled
            v1 = cv2.resize(plate_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            print("V1 (color upscale):", reader.readtext(v1, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

            # Version B: Grayscale + CLAHE, no threshold
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            print("V2 (CLAHE only):", reader.readtext(enhanced, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

            # Version C: CLAHE + bilateral filter + Otsu threshold
            denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print("V3 (threshold):", reader.readtext(thresh, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

    if not found:
        print("No plate detected in this image.")
