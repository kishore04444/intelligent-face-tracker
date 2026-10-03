import numpy as np

from insightface.app import FaceAnalysis


class FaceRecognizer:

    def __init__(self):

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )


    def get_embedding(
        self,
        frame,
        bbox
    ):

        if frame is None:
            return None

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        face_width = x2 - x1
        face_height = y2 - y1

        if (
            face_width <= 0
            or face_height <= 0
        ):
            return None


        # =====================================================
        # ADD PADDING AROUND THE YOLO FACE
        # =====================================================

        padding_x = int(
            face_width * 0.25
        )

        padding_y = int(
            face_height * 0.25
        )

        x1 = max(
            0,
            x1 - padding_x
        )

        y1 = max(
            0,
            y1 - padding_y
        )

        x2 = min(
            width,
            x2 + padding_x
        )

        y2 = min(
            height,
            y2 + padding_y
        )


        if (
            x2 <= x1
            or y2 <= y1
        ):
            return None


        # =====================================================
        # CROP ONLY THE FACE AREA
        # =====================================================

        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        if face_crop.size == 0:
            return None


        # =====================================================
        # INSIGHTFACE ON SMALL FACE CROP
        #
        # Instead of running InsightFace on the complete
        # camera frame, only the detected face is processed.
        # This significantly reduces unnecessary work.
        # =====================================================

        faces = self.app.get(
            face_crop,
            max_num=1
        )


        if not faces:
            return None


        # =====================================================
        # SELECT LARGEST DETECTED FACE
        # =====================================================

        best_face = max(
            faces,
            key=lambda face: (
                max(
                    0,
                    face.bbox[2] - face.bbox[0]
                )
                *
                max(
                    0,
                    face.bbox[3] - face.bbox[1]
                )
            )
        )


        # =====================================================
        # GET ARCFace EMBEDDING
        # =====================================================

        embedding = np.asarray(
            best_face.embedding,
            dtype=np.float32
        )


        # =====================================================
        # NORMALIZE EMBEDDING
        # =====================================================

        norm = np.linalg.norm(
            embedding
        )


        if norm == 0:
            return None


        embedding = (
            embedding / norm
        )


        return embedding


    def compare(
        self,
        embedding1,
        embedding2
    ):

        if (
            embedding1 is None
            or embedding2 is None
        ):
            return -1.0


        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        )

        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        )


        norm1 = np.linalg.norm(
            embedding1
        )

        norm2 = np.linalg.norm(
            embedding2
        )


        if (
            norm1 == 0
            or norm2 == 0
        ):
            return -1.0


        embedding1 = (
            embedding1 / norm1
        )

        embedding2 = (
            embedding2 / norm2
        )


        return float(
            np.dot(
                embedding1,
                embedding2
            )
        )