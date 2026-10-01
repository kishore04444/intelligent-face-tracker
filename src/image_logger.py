import os
import cv2
from datetime import datetime


def save_face_image(face_crop, event_type):

    today = datetime.now().strftime("%Y-%m-%d")

    event_type = event_type.upper()

    if event_type == "ENTRY":
        folder_name = "entries"
    elif event_type == "EXIT":
        folder_name = "exits"
    else:
        folder_name = event_type.lower()

    folder = os.path.join(
        "logs",
        folder_name,
        today
    )

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = f"{event_type.lower()}_{timestamp}.jpg"

    path = os.path.join(
        folder,
        filename
    )

    success = cv2.imwrite(
        path,
        face_crop
    )

    if not success:
        raise IOError(
            f"Failed to save image: {path}"
        )

    return path