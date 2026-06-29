


import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

plate_model = YOLO("/mnt/zoneA/projects/gate/archive/streamlit_app_upstream/streamlit_app_og/backend/models/best.pt")
reader = easyocr.Reader(['en'])

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

            allow = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            # V1: Color, upscaled
            v1 = cv2.resize(plate_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            print("V1 (color upscale):", reader.readtext(v1, detail=0, allowlist=allow))

            # V2: Grayscale + CLAHE
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            print("V2 (CLAHE only):", reader.readtext(enhanced, detail=0, allowlist=allow))

            # V3: CLAHE + bilateral filter + Otsu threshold
            denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print("V3 (threshold):", reader.readtext(thresh, detail=0, allowlist=allow))

            # V4: Padded crop + upscale
            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = x2 + pad, y2 + pad
            plate_crop_padded = img[y1p:y2p, x1p:x2p]
            v4 = cv2.resize(plate_crop_padded, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            print("V4 (padded + upscale):", reader.readtext(v4, detail=0, allowlist=allow))

            # V5: Sharpened
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(plate_crop_padded, -1, kernel)
            sharpened_big = cv2.resize(sharpened, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            print("V5 (sharpened):", reader.readtext(sharpened_big, detail=0, allowlist=allow))

    if not found:
        print("No plate detected in this image.")


from paddleocr import PaddleOCR
paddle_reader = PaddleOCR(use_textline_orientation=True, lang='en')

result = paddle_reader.predict(plate_crop_padded)
for res in result:
    print("PaddleOCR result:",res['rec_texts'])
