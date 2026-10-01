import cv2

from face_detector import FaceDetector
from face_tracker import FaceTracker


MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

detector = FaceDetector(MODEL_PATH)
tracker = FaceTracker(max_distance=100)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Face tracking test started.")
print("Move your face around.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    faces = detector.detect(frame)

    tracked_faces = tracker.update(faces)

    for face in tracked_faces:
        x, y, w, h = [int(value) for value in face[:4]]
        confidence = face[4]
        face_id = int(face[15])

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Person ID: {face_id}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Intelligent Face Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tracking test finished.")
