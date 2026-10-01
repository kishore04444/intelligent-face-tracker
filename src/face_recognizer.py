import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    def get_embedding(self, frame, bbox):

        faces = self.app.get(frame)

        if not faces:
            return None

        x1, y1, x2, y2 = bbox

        best_face = None
        best_iou = 0.0

        for face in faces:

            fx1, fy1, fx2, fy2 = face.bbox.astype(int)

            # Calculate intersection
            ix1 = max(x1, fx1)
            iy1 = max(y1, fy1)
            ix2 = min(x2, fx2)
            iy2 = min(y2, fy2)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            # Calculate areas
            yolo_area = max(0, x2 - x1) * max(0, y2 - y1)
            face_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)

            union = yolo_area + face_area - intersection

            if union == 0:
                continue

            iou = intersection / union

            if iou > best_iou:
                best_iou = iou
                best_face = face

        if best_face is None or best_iou < 0.1:
            return None

        embedding = best_face.embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def compare(self, embedding1, embedding2):

        similarity = np.dot(
            embedding1,
            embedding2
        )

        return float(similarity)