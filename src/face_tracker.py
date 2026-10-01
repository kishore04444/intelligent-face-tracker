import math


class FaceTracker:
    def __init__(self, max_distance=100):
        self.next_id = 1
        self.tracked_faces = {}
        self.max_distance = max_distance

    def _center(self, face):
        x, y, w, h = face[:4]
        return (
            int(x + w / 2),
            int(y + h / 2)
        )

    def update(self, faces):
        current_centers = [
            self._center(face)
            for face in faces
        ]

        new_tracked_faces = {}
        used_old_ids = set()

        for center, face in zip(current_centers, faces):

            best_id = None
            best_distance = float("inf")

            for face_id, old_center in self.tracked_faces.items():

                if face_id in used_old_ids:
                    continue

                distance = math.dist(center, old_center)

                if distance < best_distance:
                    best_distance = distance
                    best_id = face_id

            if best_id is not None and best_distance <= self.max_distance:
                face_id = best_id
            else:
                face_id = self.next_id
                self.next_id += 1

            new_tracked_faces[face_id] = center
            used_old_ids.add(face_id)

            face_with_id = face.copy()
            face_with_id = list(face_with_id)
            face_with_id.append(face_id)

            new_tracked_faces[f"{face_id}_data"] = face_with_id

        self.tracked_faces = {
            face_id: center
            for face_id, center in new_tracked_faces.items()
            if isinstance(face_id, int)
        }

        tracked_faces = []

        for key, data in new_tracked_faces.items():
            if isinstance(key, str) and key.endswith("_data"):
                tracked_faces.append(data)

        return tracked_faces