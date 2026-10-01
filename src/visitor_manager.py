import numpy as np

from database import Database
from logger import log_event


class VisitorManager:

    def __init__(self, similarity_threshold=0.5):

        self.db = Database()

        # Similarity required to recognize an existing visitor
        self.similarity_threshold = similarity_threshold

        # Maps current ByteTrack track IDs to visitor IDs
        self.track_to_visitor = {}

        # Temporarily stores faces that are not yet confirmed
        self.pending_tracks = {}

        # Require 3 consistent observations before registering
        self.required_confirmation = 3

        # Stronger threshold for confirming a brand-new visitor
        self.pending_similarity_threshold = 0.72

        # Only add a new embedding template for strong matches
        self.template_add_threshold = 0.70

    def normalize(self, embedding):

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        return embedding / norm

    def find_matching_visitor(self, embedding):

        current_embedding = self.normalize(
            embedding
        )

        if current_embedding is None:
            return None, -1.0

        stored_embeddings = (
            self.db.get_all_embeddings()
        )

        best_visitor_id = None
        best_similarity = -1.0

        for visitor_id, embedding_blob in (
            stored_embeddings
        ):

            stored_embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float32
            )

            if stored_embedding.size == 0:
                continue

            stored_embedding = self.normalize(
                stored_embedding
            )

            if stored_embedding is None:
                continue

            similarity = float(
                np.dot(
                    current_embedding,
                    stored_embedding
                )
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_visitor_id = visitor_id

        if (
            best_visitor_id is not None
            and best_similarity >= self.similarity_threshold
        ):

            return (
                best_visitor_id,
                best_similarity
            )

        return (
            None,
            best_similarity
        )

    def check_pending_track(
        self,
        track_id,
        embedding
    ):

        current_embedding = self.normalize(
            embedding
        )

        if current_embedding is None:
            return False

        pending = self.pending_tracks.get(
            track_id
        )

        if pending is None:
            return False

        previous_embedding = (
            pending["embedding"]
        )

        similarity = float(
            np.dot(
                current_embedding,
                previous_embedding
            )
        )

        pending["observations"] += 1

        log_event(
            f"PENDING_FACE_CHECK | "
            f"track_id={track_id} | "
            f"similarity={similarity:.3f} | "
            f"observations="
            f"{pending['observations']}"
        )

        if (
            similarity
            >= self.pending_similarity_threshold
        ):

            pending["consistent"] += 1

        else:

            # The new observation is not sufficiently
            # similar, so restart the consistency count.
            pending["embedding"] = (
                current_embedding.copy()
            )

            pending["consistent"] = 1

        if (
            pending["consistent"]
            >= self.required_confirmation
        ):

            del self.pending_tracks[
                track_id
            ]

            return True

        return False

    def identify(
        self,
        track_id,
        embedding
    ):

        # --------------------------------------------------
        # Already recognized/tracked visitor
        # --------------------------------------------------

        if track_id in self.track_to_visitor:

            visitor_id = (
                self.track_to_visitor[
                    track_id
                ]
            )

            self.db.update_last_seen(
                visitor_id
            )

            log_event(
                f"TRACKING | "
                f"track_id={track_id} | "
                f"visitor_id={visitor_id}"
            )

            return (
                visitor_id,
                False
            )

        # --------------------------------------------------
        # Try matching against known visitors
        # --------------------------------------------------

        visitor_id, similarity = (
            self.find_matching_visitor(
                embedding
            )
        )

        if visitor_id is not None:

            self.track_to_visitor[
                track_id
            ] = visitor_id

            self.db.update_last_seen(
                visitor_id
            )

            log_event(
                f"FACE_RECOGNIZED | "
                f"track_id={track_id} | "
                f"visitor_id={visitor_id} | "
                f"similarity={similarity:.3f}"
            )

            # Only add a new template when recognition
            # confidence is strong.
            if (
                similarity
                >= self.template_add_threshold
            ):

                self.db.add_embedding(
                    visitor_id,
                    embedding
                )

                log_event(
                    f"EMBEDDING_TEMPLATE_ADDED | "
                    f"visitor_id={visitor_id} | "
                    f"similarity={similarity:.3f}"
                )

            else:

                log_event(
                    f"EMBEDDING_TEMPLATE_SKIPPED | "
                    f"visitor_id={visitor_id} | "
                    f"similarity={similarity:.3f}"
                )

            # This track is no longer a pending candidate.
            self.pending_tracks.pop(
                track_id,
                None
            )

            return (
                visitor_id,
                False
            )

        # --------------------------------------------------
        # Face was not matched to an existing visitor
        # --------------------------------------------------

        log_event(
            f"FACE_NOT_MATCHED | "
            f"track_id={track_id} | "
            f"best_visitor_id="
            f"{visitor_id} | "
            f"best_similarity="
            f"{similarity:.3f}"
        )

        # --------------------------------------------------
        # First observation of a possible new visitor
        # --------------------------------------------------

        if (
            track_id
            not in self.pending_tracks
        ):

            normalized = self.normalize(
                embedding
            )

            if normalized is None:

                return (
                    None,
                    False
                )

            self.pending_tracks[
                track_id
            ] = {

                "embedding":
                    normalized.copy(),

                "observations":
                    1,

                "consistent":
                    1
            }

            log_event(
                f"NEW_FACE_PENDING | "
                f"track_id={track_id}"
            )

            return (
                None,
                False
            )

        # --------------------------------------------------
        # Check whether pending face is consistent
        # --------------------------------------------------

        confirmed = (
            self.check_pending_track(
                track_id,
                embedding
            )
        )

        if not confirmed:

            return (
                None,
                False
            )

        # --------------------------------------------------
        # Confirmed as a genuinely new visitor
        # --------------------------------------------------

        visitor_id = self.db.add_visitor(
            embedding
        )

        self.track_to_visitor[
            track_id
        ] = visitor_id

        log_event(
            f"VISITOR_REGISTERED | "
            f"track_id={track_id} | "
            f"visitor_id={visitor_id}"
        )

        return (
            visitor_id,
            True
        )

    def remove_track(self, track_id):

        # Remove pending registration candidate
        self.pending_tracks.pop(
            track_id,
            None
        )

        # Remove active visitor mapping
        if track_id in self.track_to_visitor:

            visitor_id = (
                self.track_to_visitor.pop(
                    track_id
                )
            )

            log_event(
                f"TRACK_EXIT | "
                f"track_id={track_id} | "
                f"visitor_id={visitor_id}"
            )

            return visitor_id

        return None

    def close(self):

        self.db.close()