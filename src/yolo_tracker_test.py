import cv2
from ultralytics import YOLO

MODEL_PATH = "models/yolov8n-face.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("YOLOv8-Face + ByteTrack started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Intelligent Face Tracker - YOLO + ByteTrack",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("YOLOv8-Face + ByteTrack stopped.")