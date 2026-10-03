import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
    )
)


# ============================================================
# IMPORTS
# ============================================================

import cv2
import json
import math

from collections import defaultdict

from concurrent.futures import (
    ThreadPoolExecutor
)

from ultralytics import YOLO

from face_recognizer import (
    FaceRecognizer
)

from visitor_manager import (
    VisitorManager
)

from logger import (
    log_event
)

from image_logger import (
    save_face_image
)

from dashboard_server import (
    DashboardServer
)


# ============================================================
# CONFIGURATION
# ============================================================

with open(
    "config.json",
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)


# ============================================================
# MODEL PATHS
# ============================================================

YOLO_MODEL_PATH = (
    "models/yolov8n-face.pt"
)

YUNET_MODEL_PATH = (
    "models/face_detection_yunet_2023mar.onnx"
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

YOLO_CONFIDENCE = max(
    0.65,
    config[
        "detection"
    ][
        "confidence"
    ]
)


CONFIG_SKIP_FRAMES = (
    config[
        "detection"
    ][
        "skip_frames"
    ]
)


SKIP_FRAMES = min(
    CONFIG_SKIP_FRAMES,
    2
)


# ============================================================
# RECOGNITION SETTINGS
# ============================================================

SIMILARITY_THRESHOLD = (
    config[
        "recognition"
    ][
        "similarity_threshold"
    ]
)


# ============================================================
# TRACKING SETTINGS
# ============================================================

MAX_MISSING_FRAMES = (
    config[
        "tracking"
    ][
        "max_missing_frames"
    ]
)


# ============================================================
# OUTPUT SETTINGS
# ============================================================

SHOW_WINDOW = (
    config[
        "output"
    ][
        "show_window"
    ]
)


SAVE_IMAGES = (
    config[
        "output"
    ][
        "save_images"
    ]
)


# ============================================================
# FACE FILTER SETTINGS
# ============================================================

MIN_FACE_WIDTH = 45

MIN_FACE_HEIGHT = 45


# ============================================================
# IMPORTANT RECOGNITION SETTINGS
#
# 1 stable detection is enough to start recognition.
#
# This makes Person 2 start recognition immediately instead
# of waiting for several detection cycles.
# ============================================================

STABLE_TRACK_COUNT = 1


# ============================================================
# IMPORTANT:
#
# VisitorManager requires multiple consistent observations
# before registering a completely new face.
#
# Therefore this MUST be greater than 2.
# ============================================================

MAX_RECOGNITION_ATTEMPTS = 4


# ============================================================
# DISPLAY SETTINGS
# ============================================================

MAX_DISPLAY_MISSING_CYCLES = 5


# ============================================================
# YUNET SETTINGS
# ============================================================

YUNET_SCORE_THRESHOLD = 0.70

FACE_VALIDATION_IOU = 0.20

DUPLICATE_IOU_THRESHOLD = 0.30


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = (
        box_a
    )

    bx1, by1, bx2, by2 = (
        box_b
    )


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


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
        *
        intersection_height
    )


    area_a = (
        max(
            0,
            ax2 - ax1
        )
        *
        max(
            0,
            ay2 - ay1
        )
    )


    area_b = (
        max(
            0,
            bx2 - bx1
        )
        *
        max(
            0,
            by2 - by1
        )
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:

        return 0.0


    return (
        intersection
        /
        union
    )


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = (
        box_a
    )

    bx1, by1, bx2, by2 = (
        box_b
    )


    acx = (
        ax1 + ax2
    ) / 2

    acy = (
        ay1 + ay2
    ) / 2


    bcx = (
        bx1 + bx2
    ) / 2

    bcy = (
        by1 + by2
    ) / 2


    return math.sqrt(
        (
            acx - bcx
        ) ** 2
        +
        (
            acy - bcy
        ) ** 2
    )


# ============================================================
# DUPLICATE BOX CHECK
# ============================================================

def boxes_are_duplicate(
    box_a,
    box_b
):

    iou = calculate_iou(
        box_a,
        box_b
    )


    if (
        iou
        >= DUPLICATE_IOU_THRESHOLD
    ):

        return True


    ax1, ay1, ax2, ay2 = (
        box_a
    )

    bx1, by1, bx2, by2 = (
        box_b
    )


    aw = (
        ax2 - ax1
    )

    ah = (
        ay2 - ay1
    )

    bw = (
        bx2 - bx1
    )

    bh = (
        by2 - by1
    )


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


    return (
        distance
        <
        minimum_size * 0.45
    )


# ============================================================
# LOAD YOLO
# ============================================================

print(
    "Loading YOLOv8-Face..."
)


yolo = YOLO(
    YOLO_MODEL_PATH
)


# ============================================================
# LOAD YUNET
# ============================================================

print(
    "Loading YuNet face validator..."
)


yunet = (
    cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        YUNET_SCORE_THRESHOLD,
        0.30,
        5000
    )
)


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print(
    "Loading InsightFace..."
)


recognizer = (
    FaceRecognizer()
)


# ============================================================
# LOAD VISITOR MANAGER
# ============================================================

print(
    "Loading Visitor Manager..."
)


visitor_manager = (
    VisitorManager(
        similarity_threshold=(
            SIMILARITY_THRESHOLD
        )
    )
)


print(
    "All AI modules loaded."
)


# ============================================================
# OPEN VIDEO SOURCE
# ============================================================

source = (
    config[
        "input"
    ][
        "source"
    ]
)


if (
    isinstance(
        source,
        str
    )
    and
    source.isdigit()
):

    source = int(
        source
    )


cap = cv2.VideoCapture(
    source
)


if not cap.isOpened():

    print(
        "ERROR: "
        "Could not open video source."
    )

    visitor_manager.close()

    raise SystemExit(1)


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


track_seen_count = (
    defaultdict(int)
)


missing_frames = (
    defaultdict(int)
)


recognition_attempts = (
    defaultdict(int)
)


recognition_jobs = {}


failed_tracks = set()


active_tracks = set()


expired_tracks = set()


unique_visitors = set()


# ============================================================
# ENTRY TRACK STATE
#
# This prevents:
#
# Person 1
# Person 1
# Person 1
# Person 1
#
# from generating multiple ENTRY events.
#
# But after Person 1 exits, the track is removed.
# When Person 1 comes back with a new ByteTrack ID,
# a new ENTRY will be generated.
# ============================================================

entry_logged_tracks = set()


# ============================================================
# BACKGROUND RECOGNITION
#
# Two workers allow multiple faces to enter the recognition
# queue without forcing Person 2 to wait unnecessarily behind
# Person 1.
# ============================================================

executor = (
    ThreadPoolExecutor(
        max_workers=2
    )
)


# ============================================================
# DASHBOARD
# ============================================================

dashboard_server = (
    DashboardServer()
)


dashboard_server.start()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:


        # ====================================================
        # READ FRAME
        # ====================================================

        ret, frame = (
            cap.read()
        )


        if not ret:

            print(
                "WARNING: "
                "Frame read failed."
            )

            continue


        frame_count += 1


        # ====================================================
        # CURRENT TRACKS
        # ====================================================

        current_tracks = set()


        # ====================================================
        # DETECTION CYCLE
        # ====================================================

        if (
            frame_count
            %
            (
                SKIP_FRAMES + 1
            )
            == 0
        ):


            # =================================================
            # YUNET
            # =================================================

            yunet.setInputSize(
                (
                    frame.shape[1],
                    frame.shape[0]
                )
            )


            _, yunet_faces = (
                yunet.detect(
                    frame
                )
            )


            valid_yunet_boxes = []


            if (
                yunet_faces
                is not None
            ):

                for face in (
                    yunet_faces
                ):

                    fx = int(
                        face[0]
                    )

                    fy = int(
                        face[1]
                    )

                    fw = int(
                        face[2]
                    )

                    fh = int(
                        face[3]
                    )


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
                        and
                        fy2 > fy1
                    ):

                        valid_yunet_boxes.append(
                            (
                                fx1,
                                fy1,
                                fx2,
                                fy2
                            )
                        )


            # =================================================
            # YOLO + BYTETRACK
            # =================================================

            results = yolo.track(

                frame,

                persist=True,

                tracker=(
                    "bytetrack.yaml"
                ),

                conf=(
                    YOLO_CONFIDENCE
                ),

                iou=0.50,

                imgsz=640,

                verbose=False
            )


            result = (
                results[0]
            )


            candidates = []


            # =================================================
            # YOLO BOXES
            # =================================================

            if (
                result.boxes
                is not None
                and
                result.boxes.id
                is not None
            ):

                boxes = (
                    result.boxes
                )


                for i in range(
                    len(boxes)
                ):


                    # =========================================
                    # TRACK ID
                    # =========================================

                    track_id = int(
                        boxes.id[i].item()
                    )


                    # =========================================
                    # CONFIDENCE
                    # =========================================

                    if (
                        boxes.conf
                        is not None
                    ):

                        confidence = float(
                            boxes.conf[
                                i
                            ].item()
                        )

                    else:

                        confidence = 0.0


                    if (
                        confidence
                        <
                        YOLO_CONFIDENCE
                    ):

                        continue


                    # =========================================
                    # BOUNDING BOX
                    # =========================================

                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ) = (
                        boxes.xyxy[
                            i
                        ]
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
                        or
                        y2 <= y1
                    ):

                        continue


                    width = (
                        x2 - x1
                    )

                    height = (
                        y2 - y1
                    )


                    # =========================================
                    # FACE SIZE FILTER
                    # =========================================

                    if (
                        width
                        <
                        MIN_FACE_WIDTH
                    ):

                        continue


                    if (
                        height
                        <
                        MIN_FACE_HEIGHT
                    ):

                        continue


                    yolo_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )


                    # =========================================
                    # YUNET VALIDATION
                    # =========================================

                    face_confirmed = (
                        False
                    )


                    for yunet_box in (
                        valid_yunet_boxes
                    ):


                        # -------------------------------------
                        # IOU MATCH
                        # -------------------------------------

                        overlap = (
                            calculate_iou(
                                yolo_box,
                                yunet_box
                            )
                        )


                        if (
                            overlap
                            >=
                            FACE_VALIDATION_IOU
                        ):

                            face_confirmed = (
                                True
                            )

                            break


                        # -------------------------------------
                        # CENTER MATCH
                        # -------------------------------------

                        cx = (
                            x1 + x2
                        ) / 2

                        cy = (
                            y1 + y2
                        ) / 2


                        (
                            ux1,
                            uy1,
                            ux2,
                            uy2
                        ) = yunet_box


                        padding_x = (
                            ux2 - ux1
                        ) * 0.20


                        padding_y = (
                            uy2 - uy1
                        ) * 0.20


                        if (
                            ux1
                            - padding_x
                            <= cx
                            <= ux2
                            + padding_x
                            and
                            uy1
                            - padding_y
                            <= cy
                            <= uy2
                            + padding_y
                        ):

                            face_confirmed = (
                                True
                            )

                            break


                    if not face_confirmed:

                        continue


                    # =========================================
                    # CANDIDATE
                    # =========================================

                    candidates.append(
                        {
                            "track_id":
                                track_id,

                            "bbox":
                                yolo_box,

                            "confidence":
                                confidence,

                            "known":
                                (
                                    track_id
                                    in
                                    visitor_manager
                                    .track_to_visitor
                                )
                        }
                    )


            # =================================================
            # SORT CANDIDATES
            # =================================================

            candidates.sort(

                key=lambda item: (

                    item["known"],

                    item["confidence"]

                ),

                reverse=True
            )


            # =================================================
            # REMOVE DUPLICATES
            # =================================================

            filtered_candidates = []


            for candidate in (
                candidates
            ):

                duplicate = False


                for kept in (
                    filtered_candidates
                ):

                    if boxes_are_duplicate(

                        candidate[
                            "bbox"
                        ],

                        kept[
                            "bbox"
                        ]

                    ):

                        duplicate = True

                        break


                if not duplicate:

                    filtered_candidates.append(
                        candidate
                    )


            # =================================================
            # PROCESS TRACKS
            # =================================================

            for candidate in (
                filtered_candidates
            ):

                track_id = (
                    candidate[
                        "track_id"
                    ]
                )


                (
                    x1,
                    y1,
                    x2,
                    y2
                ) = (
                    candidate[
                        "bbox"
                    ]
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


                last_boxes[
                    track_id
                ] = candidate[
                    "bbox"
                ]


                # =============================================
                # SAVE FACE CROP
                # =============================================

                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if (
                    face_crop.size
                    > 0
                ):

                    last_face_crops[
                        track_id
                    ] = (
                        face_crop.copy()
                    )


                # =============================================
                # RECOGNITION STATE
                # =============================================

                already_identified = (
                    track_id
                    in
                    visitor_manager
                    .track_to_visitor
                )


                recognition_running = (
                    track_id
                    in
                    recognition_jobs
                )


                recognition_failed = (
                    track_id
                    in
                    failed_tracks
                )


                # =============================================
                # START RECOGNITION
                # =============================================

                if (

                    not already_identified

                    and
                    not recognition_running

                    and
                    not recognition_failed

                    and

                    track_seen_count[
                        track_id
                    ]
                    >=
                    STABLE_TRACK_COUNT

                    and

                    recognition_attempts[
                        track_id
                    ]
                    <
                    MAX_RECOGNITION_ATTEMPTS

                ):


                    recognition_frame = (
                        frame.copy()
                    )


                    recognition_jobs[
                        track_id
                    ] = (

                        executor.submit(

                            recognizer.get_embedding,

                            recognition_frame,

                            candidate[
                                "bbox"
                            ]

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


            # =================================================
            # CHECK RECOGNITION JOBS
            # =================================================

            for (
                track_id,
                future
            ) in list(
                recognition_jobs.items()
            ):


                if not future.done():

                    continue


                del recognition_jobs[
                    track_id
                ]


                # =============================================
                # EXPIRED TRACK
                # =============================================

                if (
                    track_id
                    in
                    expired_tracks
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


                # =============================================
                # GET EMBEDDING
                # =============================================

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


                # =============================================
                # FAILED EMBEDDING
                # =============================================

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
                        >=
                        MAX_RECOGNITION_ATTEMPTS

                    ):

                        failed_tracks.add(
                            track_id
                        )


                        last_boxes.pop(
                            track_id,
                            None
                        )


                    continue


                # =============================================
                # EMBEDDING SUCCESS
                # =============================================

                log_event(

                    f"EMBEDDING_GENERATED | "
                    f"track_id={track_id}"

                )


                # =============================================
                # IDENTIFY VISITOR
                # =============================================

                (
                    visitor_id,
                    is_new
                ) = (
                    visitor_manager.identify(

                        track_id,

                        embedding

                    )
                )


                # =============================================
                # NO IDENTIFICATION YET
                #
                # This is normal for a brand-new face because
                # VisitorManager waits for consistent
                # observations before registering it.
                # =============================================

                if (
                    visitor_id
                    is None
                ):

                    continue


                # =============================================
                # UNIQUE VISITOR
                #
                # A returning person does NOT increase this.
                # =============================================

                unique_visitors.add(
                    visitor_id
                )


                # =============================================
                # ENTRY
                #
                # IMPORTANT:
                #
                # We do NOT check is_new here.
                #
                # Therefore:
                #
                # NEW PERSON
                #     -> ENTRY
                #
                # KNOWN PERSON RETURNING
                #     -> ENTRY
                #
                # SAME TRACK
                #     -> no duplicate ENTRY
                # =============================================

                if (
                    track_id
                    not in
                    entry_logged_tracks
                ):


                    image_path = None


                    if SAVE_IMAGES:

                        face_crop = (
                            last_face_crops.get(
                                track_id
                            )
                        )


                        if (
                            face_crop
                            is not None
                        ):

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


                    entry_logged_tracks.add(
                        track_id
                    )


                # =============================================
                # NEW VISITOR LOG
                # =============================================

                if is_new:

                    log_event(

                        f"VISITOR_REGISTERED | "
                        f"visitor_id={visitor_id} | "
                        f"track_id={track_id}"

                    )


                # =============================================
                # EXISTING VISITOR LOG
                # =============================================

                else:

                    log_event(

                        f"VISITOR_RECOGNIZED | "
                        f"visitor_id={visitor_id} | "
                        f"track_id={track_id}"

                    )


            # =================================================
            # HANDLE MISSING TRACKS
            # =================================================

            for old_track in list(
                active_tracks
            ):


                if (
                    old_track
                    in
                    current_tracks
                ):

                    continue


                missing_frames[
                    old_track
                ] += 1


                # =============================================
                # WAIT BEFORE EXIT
                # =============================================

                if (
                    missing_frames[
                        old_track
                    ]
                    <
                    MAX_MISSING_FRAMES
                ):

                    continue


                # =============================================
                # REMOVE TRACK
                # =============================================

                visitor_id = (
                    visitor_manager.remove_track(
                        old_track
                    )
                )


                # =============================================
                # EXIT EVENT
                # =============================================

                if (
                    visitor_id
                    is not None
                ):


                    image_path = None


                    if SAVE_IMAGES:

                        face_crop = (
                            last_face_crops.get(
                                old_track
                            )
                        )


                        if (
                            face_crop
                            is not None
                        ):

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


                # =============================================
                # EXPIRE TRACK
                # =============================================

                expired_tracks.add(
                    old_track
                )


                # =============================================
                # VERY IMPORTANT
                #
                # Remove the ENTRY lock.
                #
                # When this person comes back with another
                # ByteTrack ID, a new ENTRY is allowed.
                # =============================================

                entry_logged_tracks.discard(
                    old_track
                )


                # =============================================
                # CLEAN TRACK STATE
                # =============================================

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


            # =================================================
            # KEEP TEMPORARILY MISSING TRACKS
            # =================================================

            active_tracks = (

                current_tracks

                |

                {
                    track_id

                    for track_id
                    in active_tracks

                    if (

                        track_id
                        not in
                        expired_tracks

                        and

                        missing_frames.get(
                            track_id,
                            0
                        )
                        <
                        MAX_MISSING_FRAMES

                    )
                }

            )


        # ====================================================
        # DRAW TRACKS
        # ====================================================

        for (
            track_id,
            bbox
        ) in list(
            last_boxes.items()
        ):


            # ================================================
            # FAILED
            # ================================================

            if (
                track_id
                in
                failed_tracks
            ):

                continue


            # ================================================
            # EXPIRED
            # ================================================

            if (
                track_id
                in
                expired_tracks
            ):

                continue


            # ================================================
            # STALE
            # ================================================

            if (
                missing_frames.get(
                    track_id,
                    0
                )
                >
                MAX_DISPLAY_MISSING_CYCLES
            ):

                continue


            # ================================================
            # IDENTIFIED
            # ================================================

            if (
                track_id
                in
                visitor_manager
                .track_to_visitor
            ):

                visitor_id = (
                    visitor_manager
                    .track_to_visitor[
                        track_id
                    ]
                )


                label = (
                    f"Person {visitor_id}"
                )


            # ================================================
            # RECOGNIZING
            # ================================================

            elif (
                track_id
                in
                recognition_jobs
            ):

                label = (
                    "Recognizing..."
                )


            # ================================================
            # NOTHING
            # ================================================

            else:

                continue


            x1, y1, x2, y2 = (
                bbox
            )


            # ================================================
            # BOX
            # ================================================

            cv2.rectangle(

                frame,

                (
                    x1,
                    y1
                ),

                (
                    x2,
                    y2
                ),

                (
                    0,
                    255,
                    0
                ),

                2

            )


            # ================================================
            # LABEL
            # ================================================

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

                (
                    0,
                    255,
                    0
                ),

                2

            )


        # ====================================================
        # UNIQUE VISITOR COUNT
        # ====================================================

        cv2.putText(

            frame,

            (
                "Unique Visitors: "
                f"{len(unique_visitors)}"
            ),

            (
                20,
                35
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (
                0,
                255,
                255
            ),

            2

        )


        # ====================================================
        # FRAME NUMBER
        # ====================================================

        cv2.putText(

            frame,

            (
                f"Frame: "
                f"{frame_count}"
            ),

            (
                20,
                65
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (
                255,
                255,
                255
            ),

            1

        )


        # ====================================================
        # SEND FRAME TO DASHBOARD
        # ====================================================

        dashboard_server.update_frame(
            frame
        )


        # ====================================================
        # OPENCV WINDOW
        # ====================================================

        if SHOW_WINDOW:

            cv2.imshow(

                "Intelligent Face Tracker",

                frame

            )


            if (
                cv2.waitKey(1)
                &
                0xFF
            ) == ord("q"):

                break


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()


    cv2.destroyAllWindows()


    executor.shutdown(

        wait=False,

        cancel_futures=True

    )


    dashboard_server.stop()


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