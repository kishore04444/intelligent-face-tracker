import cv2
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ultralytics import YOLO

from face_recognizer import FaceRecognizer
from visitor_manager import VisitorManager
from logger import log_event
from image_logger import save_face_image


# ============================================================
# CONFIGURATION
# ============================================================

with open("config.json", "r") as file:
    config = json.load(file)


YOLO_MODEL_PATH = "models/yolov8n-face.pt"
YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"


# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

YOLO_CONFIDENCE = max(
    0.65,
    config["detection"]["confidence"]
)

CONFIG_SKIP_FRAMES = config["detection"]["skip_frames"]

SKIP_FRAMES = min(
    CONFIG_SKIP_FRAMES,
    2
)


# ------------------------------------------------------------
# Recognition
# ------------------------------------------------------------

SIMILARITY_THRESHOLD = (
    config["recognition"]["similarity_threshold"]
)


# ------------------------------------------------------------
# Tracking
# ------------------------------------------------------------

MAX_MISSING_FRAMES = (
    config["tracking"]["max_missing_frames"]
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

SHOW_WINDOW = config["output"]["show_window"]
SAVE_IMAGES = config["output"]["save_images"]


# ============================================================
# EXTRA FILTER SETTINGS
# ============================================================

MIN_FACE_WIDTH = 45
MIN_FACE_HEIGHT = 45

STABLE_TRACK_COUNT = 2

MAX_RECOGNITION_ATTEMPTS = 2


# Keep the box visible during short detection gaps.
# This is ONLY for display.
# It does NOT control EXIT detection.
MAX_DISPLAY_MISSING_CYCLES = 5


YUNET_SCORE_THRESHOLD = 0.70

FACE_VALIDATION_IOU = 0.20

DUPLICATE_IOU_THRESHOLD = 0.30


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two boxes.

    Box format:
        (x1, y1, x2, y2)
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def center_distance(box_a, box_b):
    """
    Calculate center-point distance between two boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2

    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    return math.sqrt(
        (acx - bcx) ** 2
        + (acy - bcy) ** 2
    )


def boxes_are_duplicate(box_a, box_b):
    """
    Determine whether two detections are probably
    the same physical face.
    """

    iou = calculate_iou(
        box_a,
        box_b
    )

    if iou >= DUPLICATE_IOU_THRESHOLD:
        return True

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    aw = ax2 - ax1
    ah = ay2 - ay1

    bw = bx2 - bx1
    bh = by2 - by1

    minimum_size = min(
        aw,
        ah,
        bw,
        bh
    )

    if minimum_size <= 0:
        return False

    distance = center_distance(
        box_a,
        box_b
    )

    if distance < minimum_size * 0.45:
        return True

    return False


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLOv8-Face...")

yolo = YOLO(
    YOLO_MODEL_PATH
)


# ============================================================
# LOAD YUNET
# ============================================================

print("Loading YuNet face validator...")

yunet = cv2.FaceDetectorYN.create(
    YUNET_MODEL_PATH,
    "",
    (320, 320),
    YUNET_SCORE_THRESHOLD,
    0.30,
    5000
)


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("Loading InsightFace...")

recognizer = FaceRecognizer()


# ============================================================
# LOAD VISITOR MANAGER
# ============================================================

print("Loading Visitor Manager...")

visitor_manager = VisitorManager(
    similarity_threshold=SIMILARITY_THRESHOLD
)


print("All AI modules loaded.")


# ============================================================
# OPEN CAMERA
# ============================================================

source = config["input"]["source"]

if isinstance(source, str) and source.isdigit():
    source = int(source)


cap = cv2.VideoCapture(
    source
)


if not cap.isOpened():

    print(
        "ERROR: Could not open video source."
    )

    exit()


print(
    "Video source opened successfully."
)

print(
    "Press Q to quit."
)


# ============================================================
# TRACK STATE
# ============================================================

frame_count = 0

last_boxes = {}

last_face_crops = {}

track_seen_count = defaultdict(int)

missing_frames = defaultdict(int)

recognition_attempts = defaultdict(int)

recognition_jobs = {}

failed_tracks = set()

active_tracks = set()

expired_tracks = set()

unique_visitors = set()


# ============================================================
# BACKGROUND RECOGNITION
# ============================================================

executor = ThreadPoolExecutor(
    max_workers=1
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "WARNING: Frame read failed."
        )

        continue


    frame_count += 1


    # ========================================================
    # RUN YOLO + YUNET
    # ========================================================

    if frame_count % (
        SKIP_FRAMES + 1
    ) == 0:


        # ====================================================
        # YUNET FACE DETECTION
        # ====================================================

        yunet.setInputSize(
            (
                frame.shape[1],
                frame.shape[0]
            )
        )


        _, yunet_faces = yunet.detect(
            frame
        )


        valid_yunet_boxes = []


        if yunet_faces is not None:

            for face in yunet_faces:

                fx = int(face[0])
                fy = int(face[1])

                fw = int(face[2])
                fh = int(face[3])

                fx1 = max(
                    0,
                    fx
                )

                fy1 = max(
                    0,
                    fy
                )

                fx2 = min(
                    frame.shape[1],
                    fx + fw
                )

                fy2 = min(
                    frame.shape[0],
                    fy + fh
                )


                if (
                    fx2 > fx1
                    and fy2 > fy1
                ):

                    valid_yunet_boxes.append(
                        (
                            fx1,
                            fy1,
                            fx2,
                            fy2
                        )
                    )


        # ====================================================
        # YOLO + BYTETRACK
        # ====================================================

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONFIDENCE,
            iou=0.50,
            imgsz=640,
            verbose=False
        )


        result = results[0]

        candidates = []


        if result.boxes is not None:

            boxes = result.boxes


            for i in range(
                len(boxes)
            ):


                # --------------------------------------------
                # TRACK ID
                # --------------------------------------------

                if boxes.id is None:
                    continue


                track_id = int(
                    boxes.id[i].item()
                )


                # --------------------------------------------
                # CONFIDENCE
                # --------------------------------------------

                confidence = 0.0


                if boxes.conf is not None:

                    confidence = float(
                        boxes.conf[i].item()
                    )


                if (
                    confidence
                    < YOLO_CONFIDENCE
                ):
                    continue


                # --------------------------------------------
                # BOX
                # --------------------------------------------

                x1, y1, x2, y2 = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )


                x1 = max(
                    0,
                    int(x1)
                )

                y1 = max(
                    0,
                    int(y1)
                )

                x2 = min(
                    frame.shape[1],
                    int(x2)
                )

                y2 = min(
                    frame.shape[0],
                    int(y2)
                )


                if (
                    x2 <= x1
                    or y2 <= y1
                ):
                    continue


                width = x2 - x1
                height = y2 - y1


                # --------------------------------------------
                # SIZE FILTER
                # --------------------------------------------

                if (
                    width
                    < MIN_FACE_WIDTH
                ):
                    continue


                if (
                    height
                    < MIN_FACE_HEIGHT
                ):
                    continue


                yolo_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )


                # =================================================
                # YUNET VALIDATION
                # =================================================

                face_confirmed = False


                for yunet_box in (
                    valid_yunet_boxes
                ):

                    overlap = calculate_iou(
                        yolo_box,
                        yunet_box
                    )


                    # --------------------------------------------
                    # Direct IoU match
                    # --------------------------------------------

                    if (
                        overlap
                        >= FACE_VALIDATION_IOU
                    ):

                        face_confirmed = True
                        break


                    # --------------------------------------------
                    # Center-point match
                    # --------------------------------------------

                    cx = (
                        x1 + x2
                    ) / 2

                    cy = (
                        y1 + y2
                    ) / 2


                    ux1, uy1, ux2, uy2 = (
                        yunet_box
                    )


                    padding_x = (
                        ux2 - ux1
                    ) * 0.20

                    padding_y = (
                        uy2 - uy1
                    ) * 0.20


                    if (
                        ux1 - padding_x
                        <= cx
                        <= ux2 + padding_x
                        and
                        uy1 - padding_y
                        <= cy
                        <= uy2 + padding_y
                    ):

                        face_confirmed = True
                        break


                # --------------------------------------------
                # Reject if YuNet cannot confirm face
                # --------------------------------------------

                if not face_confirmed:

                    continue


                # =================================================
                # CANDIDATE
                # =================================================

                already_known = (
                    track_id
                    in visitor_manager.track_to_visitor
                )


                candidates.append(
                    {
                        "track_id": track_id,
                        "bbox": yolo_box,
                        "confidence": confidence,
                        "known": already_known
                    }
                )


        # ====================================================
        # REMOVE DUPLICATE TRACK BOXES
        # ====================================================

        candidates.sort(
            key=lambda item: (
                item["known"],
                item["confidence"]
            ),
            reverse=True
        )


        filtered_candidates = []


        for candidate in candidates:

            duplicate = False


            for kept in (
                filtered_candidates
            ):

                if boxes_are_duplicate(
                    candidate["bbox"],
                    kept["bbox"]
                ):

                    duplicate = True
                    break


            if not duplicate:

                filtered_candidates.append(
                    candidate
                )


        # ====================================================
        # PROCESS VALID TRACKS
        # ====================================================

        current_tracks = set()


        for candidate in (
            filtered_candidates
        ):

            track_id = (
                candidate["track_id"]
            )

            x1, y1, x2, y2 = (
                candidate["bbox"]
            )


            current_tracks.add(
                track_id
            )


            track_seen_count[
                track_id
            ] += 1


            missing_frames[
                track_id
            ] = 0


            # --------------------------------------------
            # SAVE BOX
            # --------------------------------------------

            last_boxes[
                track_id
            ] = (
                x1,
                y1,
                x2,
                y2
            )


            # --------------------------------------------
            # SAVE FACE CROP
            # --------------------------------------------

            face_crop = frame[
                y1:y2,
                x1:x2
            ]


            if face_crop.size > 0:

                last_face_crops[
                    track_id
                ] = face_crop.copy()


            # =================================================
            # START RECOGNITION
            # =================================================

            already_identified = (
                track_id
                in visitor_manager.track_to_visitor
            )


            recognition_running = (
                track_id
                in recognition_jobs
            )


            recognition_failed = (
                track_id
                in failed_tracks
            )


            if (
                not already_identified
                and not recognition_running
                and not recognition_failed
                and track_seen_count[
                    track_id
                ] >= STABLE_TRACK_COUNT
                and recognition_attempts[
                    track_id
                ] < MAX_RECOGNITION_ATTEMPTS
            ):

                recognition_frame = (
                    frame.copy()
                )


                recognition_jobs[
                    track_id
                ] = executor.submit(
                    recognizer.get_embedding,
                    recognition_frame,
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )


                recognition_attempts[
                    track_id
                ] += 1


                log_event(
                    f"RECOGNITION_STARTED | "
                    f"track_id={track_id} | "
                    f"attempt="
                    f"{recognition_attempts[track_id]}"
                )


        # ====================================================
        # CHECK RECOGNITION JOBS
        # ====================================================

        for track_id, future in list(
            recognition_jobs.items()
        ):

            if not future.done():
                continue


            del recognition_jobs[
                track_id
            ]


            # --------------------------------------------
            # Ignore expired tracks
            # --------------------------------------------

            if (
                track_id
                in expired_tracks
            ):
                continue


            if (
                missing_frames.get(
                    track_id,
                    0
                )
                > 1
            ):
                continue


            try:

                embedding = (
                    future.result()
                )

            except Exception as error:

                log_event(
                    f"RECOGNITION_ERROR | "
                    f"track_id={track_id} | "
                    f"error={error}"
                )

                embedding = None


            # =================================================
            # FAILED
            # =================================================

            if embedding is None:

                log_event(
                    f"EMBEDDING_FAILED | "
                    f"track_id={track_id} | "
                    f"attempt="
                    f"{recognition_attempts[track_id]}"
                )


                if (
                    recognition_attempts[
                        track_id
                    ]
                    >= MAX_RECOGNITION_ATTEMPTS
                ):

                    failed_tracks.add(
                        track_id
                    )

                    last_boxes.pop(
                        track_id,
                        None
                    )


                continue


            # =================================================
            # SUCCESS
            # =================================================

            log_event(
                f"EMBEDDING_GENERATED | "
                f"track_id={track_id}"
            )


            visitor_id, is_new = (
                visitor_manager.identify(
                    track_id,
                    embedding
                )
            )


            # Only add a valid visitor ID.
            if visitor_id is not None:

                unique_visitors.add(
                    visitor_id
                )


            # =================================================
            # NEW VISITOR
            # =================================================

            if is_new:

                image_path = None


                if SAVE_IMAGES:

                    face_crop = (
                        last_face_crops.get(
                            track_id
                        )
                    )


                    if face_crop is not None:

                        image_path = (
                            save_face_image(
                                face_crop,
                                "ENTRY"
                            )
                        )


                visitor_manager.db.add_event(
                    visitor_id,
                    "ENTRY",
                    image_path
                )


                log_event(
                    f"ENTRY | "
                    f"visitor_id={visitor_id} | "
                    f"track_id={track_id}"
                )


            # =================================================
            # EXISTING VISITOR
            # =================================================

            else:

                log_event(
                    f"VISITOR_RECOGNIZED | "
                    f"visitor_id={visitor_id} | "
                    f"track_id={track_id}"
                )


        # ====================================================
        # HANDLE MISSING TRACKS
        # ====================================================

        for old_track in list(
            active_tracks
        ):

            if (
                old_track
                not in current_tracks
            ):

                missing_frames[
                    old_track
                ] += 1


                # --------------------------------------------
                # EXIT
                # --------------------------------------------

                if (
                    missing_frames[
                        old_track
                    ]
                    >= MAX_MISSING_FRAMES
                ):

                    visitor_id = (
                        visitor_manager.remove_track(
                            old_track
                        )
                    )


                    if visitor_id is not None:

                        image_path = None


                        if SAVE_IMAGES:

                            face_crop = (
                                last_face_crops.get(
                                    old_track
                                )
                            )


                            if face_crop is not None:

                                image_path = (
                                    save_face_image(
                                        face_crop,
                                        "EXIT"
                                    )
                                )


                        visitor_manager.db.add_event(
                            visitor_id,
                            "EXIT",
                            image_path
                        )


                        log_event(
                            f"EXIT | "
                            f"visitor_id={visitor_id} | "
                            f"track_id={old_track}"
                        )


                    # ----------------------------------------
                    # CLEAN STATE
                    # ----------------------------------------

                    expired_tracks.add(
                        old_track
                    )


                    missing_frames.pop(
                        old_track,
                        None
                    )

                    track_seen_count.pop(
                        old_track,
                        None
                    )

                    recognition_attempts.pop(
                        old_track,
                        None
                    )

                    last_boxes.pop(
                        old_track,
                        None
                    )

                    last_face_crops.pop(
                        old_track,
                        None
                    )

                    recognition_jobs.pop(
                        old_track,
                        None
                    )

                    failed_tracks.discard(
                        old_track
                    )


        # ====================================================
        # IMPORTANT:
        # KEEP TEMPORARILY MISSING TRACKS ACTIVE
        # UNTIL EXIT TIMEOUT IS REACHED.
        # ====================================================

        active_tracks = (
            current_tracks
            | {
                track_id
                for track_id in active_tracks
                if (
                    track_id
                    not in expired_tracks
                    and missing_frames.get(
                        track_id,
                        0
                    ) < MAX_MISSING_FRAMES
                )
            }
        )


    # ========================================================
    # DRAW ONLY VALID TRACKS
    # ========================================================

    for track_id, bbox in list(
        last_boxes.items()
    ):


        # --------------------------------------------
        # Failed recognition
        # --------------------------------------------

        if (
            track_id
            in failed_tracks
        ):
            continue


        # --------------------------------------------
        # Don't display an expired track
        # --------------------------------------------

        if (
            track_id
            in expired_tracks
        ):
            continue


        # --------------------------------------------
        # Stale track
        # --------------------------------------------

        if (
            missing_frames.get(
                track_id,
                0
            )
            > MAX_DISPLAY_MISSING_CYCLES
        ):
            continue


        # --------------------------------------------
        # IDENTIFIED PERSON
        # --------------------------------------------

        if (
            track_id
            in visitor_manager.track_to_visitor
        ):

            visitor_id = (
                visitor_manager.track_to_visitor[
                    track_id
                ]
            )

            label = (
                f"Person {visitor_id}"
            )


        # --------------------------------------------
        # RECOGNITION
        # --------------------------------------------

        elif (
            track_id
            in recognition_jobs
        ):

            label = (
                "Recognizing..."
            )


        # --------------------------------------------
        # NOTHING TO SHOW
        # --------------------------------------------

        else:

            continue


        x1, y1, x2, y2 = bbox


        # --------------------------------------------
        # DRAW BOX
        # --------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # --------------------------------------------
        # DRAW LABEL
        # --------------------------------------------

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    y1 - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ========================================================
    # UNIQUE VISITOR COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Unique Visitors: {len(unique_visitors)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ========================================================
    # FRAME NUMBER
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    if SHOW_WINDOW:

        cv2.imshow(
            "Intelligent Face Tracker",
            frame
        )


        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

executor.shutdown(
    wait=False,
    cancel_futures=True
)

visitor_manager.close()


print()
print(
    "==================================="
)

print(
    "Intelligent Face Tracker stopped."
)

print(
    f"Unique Visitors: "
    f"{len(unique_visitors)}"
)

print(
    "==================================="
)