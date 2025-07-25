import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# === CONFIG ===
VIDEO_SOURCE = r"C:\Users\Sanya\OneDrive\Document\projects\YOLOv8BDD100k\dataset\indiaNightTime.mp4"  # or "rtsp://..." or local video file path
VEHICLE_MODEL_PATH = r"C:\Users\Sanya\OneDrive\Document\projects\YOLOv8BDD100k\runs\detect\train3\weights\best.pt"
PLATE_MODEL_PATH = r"C:\Users\Sanya\OneDrive\Document\projects\YOLOv8BDD100k\LiscencePlateRuns\detect\train2\weights\best.pt"

# === INIT MODELS & OCR ===
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# === FUNCTIONS ===
def preprocess_frame(frame, target_size=(640,640)):
    # Resize + pad
    h, w = frame.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (new_w, new_h))
    delta_w, delta_h = target_size[1]-new_w, target_size[0]-new_h
    top, bottom = delta_h//2, delta_h - delta_h//2
    left, right = delta_w//2, delta_w - delta_w//2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    # Brightness/contrast
    adjusted = cv2.convertScaleAbs(padded, alpha=1.5, beta=10)
    return adjusted

# === MAIN LOOP ===
cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of stream or cannot read frame.")
        break

    frame = preprocess_frame(frame)

    # --- Vehicle detection ---
    vehicle_results = vehicle_model(frame)[0]
    for box in vehicle_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = vehicle_model.names[cls_id]
        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # --- Plate detection inside vehicle ---
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0: continue  # skip if invalid crop
        plate_results = plate_model(vehicle_crop)[0]
        for pbox in plate_results.boxes:
            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            p_conf = float(pbox.conf[0])
            # adjust coords to original frame
            fx1, fy1, fx2, fy2 = x1+px1, y1+py1, x1+px2, y1+py2
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0,0,255), 2)
            p_label = f"Plate {p_conf:.2f}"
            cv2.putText(frame, p_label, (fx1, fy1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # --- OCR ---
            plate_crop = frame[fy1:fy2, fx1:fx2]
            if plate_crop.size == 0: continue
            ocr_result = ocr_reader.readtext(plate_crop)
            for (_, text, _) in ocr_result:
                cv2.putText(frame, text, (fx1, fy2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
