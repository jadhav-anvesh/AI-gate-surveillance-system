import cv2
import os
import glob
from collections import Counter
from ultralytics import YOLO
import easyocr
from backend.database.detection_service import DetectionDatabaseService

db_service = DetectionDatabaseService()


# Load models once
plate_model = YOLO("/mnt/zoneA/projects/gate/archive/streamlit_app_upstream/streamlit_app_og/backend/models/best.pt")
reader = easyocr.Reader(['en'])

def read_plate_from_image(img, tracker_id, frame_number, session_id):
    results = plate_model(img)
    print(f"[ALPR IMG] shape={img.shape}")
    print(f"[ALPR RESULTS] count={len(results)}")
    boxed_img = img.copy()
    found_plate = False
    text = None
    plate_bbox = None

    for result in results:
        print(f"[ALPR BOXES] {len(result.boxes)}")
        if len(result.boxes) == 0:
            print("[ALPR BOXES] NO PLATES FOUND")
        for box in result.boxes:
            found_plate = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_bbox = [x1, y1, x2, y2]

            print(f"[ALPR] Plate detected at coordinates ({x1}, {y1}, {x2}, {y2}) for tracker_id={tracker_id} frame_number={frame_number}")

            plate_crop = img[y1:y2, x1:x2]
            ocr_text = reader.readtext(plate_crop, detail=0)
            cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if ocr_text and text is None:
                text = "".join(ocr_text).upper().replace(" ", "")

    if not found_plate:
        return "NO_PLATE", None, boxed_img, None
    elif text is None:
        return "NO_TEXT", None, boxed_img, plate_bbox
    else:
        return "TEXT", text, boxed_img, plate_bbox


def get_best_plate(tracker_id, session_id=None):
    frames = sorted(glob.glob(f"temp/alpr_frames/session_{session_id}_tracker_{tracker_id}_frame*.jpg"))
    readings = []
    best_bbox = None

    for frame_path in frames:
        frame_number = os.path.basename(frame_path).split("frame")[-1].split(".")[0]
        img = cv2.imread(frame_path)
        if img is None:
            continue

        status, text, boxed_img, bbox = read_plate_from_image(img, tracker_id, frame_number, session_id)

        if status == "NO_PLATE":
            print(f"[ALPR] No number plate detected for tracker_id={tracker_id} frame_number={frame_number}")
            #os.remove(frame_path)<------------------------------added comment for verification(Anvesh)

        elif status == "NO_TEXT":
            print(f"[ALPR] Plate detected but no text detected for tracker_id={tracker_id} frame_number={frame_number}")
            cv2.imwrite(frame_path, boxed_img)
            best_bbox = bbox

        elif status == "TEXT":
            print(f"[ALPR] tracker_id={tracker_id} frame_number={frame_number} → {text}")
            readings.append(text)
            cv2.imwrite(frame_path, boxed_img)
            best_bbox = bbox
    print(
        f"[DEBUG] tracker={tracker_id} "
        f"session_id={session_id} "
        f"best_bbox={best_bbox}"
    )
    if best_bbox is not None and session_id is not None:
        db_service.update_plate_bbox(session_id, tracker_id, best_bbox)
        print(f"[ALPR] Saved plate bbox to DB for tracker_id={tracker_id}: {best_bbox}")

    #if not readings:<--------------------------------changed
     #   return None<------------------------------changed
#=============================================================added
    if not readings and best_bbox is None:
        return None

    best = None

    if readings:
        best = Counter(readings).most_common(1)[0][0]

    return {
        "plate_text": best,
        "plate_bbox": best_bbox
    }
#======================================================================
    best = Counter(readings).most_common(1)[0][0]
    print(f"[ALPR RESULT] Best plate for tracker_id={tracker_id} → {best}")
    return best
