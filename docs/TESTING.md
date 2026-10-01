# Testing and Validation

## 1. Testing Overview

The Intelligent Face Tracker was tested using a live camera input to validate face detection, tracking, recognition, visitor registration, entry/exit handling, and unique visitor counting.

## 2. Test Case: New Visitor Registration

### Objective

Verify that a previously unknown face is registered as a new visitor.

### Result

The system initially marked the face as a pending face and performed multiple observations.

After consistent observations, the system registered the visitor.

Example log:

VISITOR_REGISTERED | track_id=1 | visitor_id=1

### Expected Behavior

- New visitor receives a unique visitor ID.
- Visitor record is stored in SQLite.
- Initial face embedding is stored.
- Entry event is generated.
- Face image is saved.

### Status

PASSED

---

## 3. Test Case: Recognition of Existing Visitor

### Objective

Verify that a previously registered visitor is recognized without creating a duplicate visitor.

### Result

The same visitor was recognized with the existing visitor ID.

Observed similarity values included:

- 0.710
- 0.609
- 0.730
- 0.786
- 0.795
- 0.849

The visitor continued to use:

visitor_id=1

### Expected Behavior

The system must reuse the existing visitor ID instead of registering another visitor.

### Status

PASSED

---

## 4. Test Case: Multiple Person Detection

### Objective

Verify that multiple faces can be detected and tracked simultaneously.

### Result

The YOLO + ByteTrack pipeline successfully detected and tracked multiple people.

Each tracked face received an independent track ID.

### Expected Behavior

Each visible person should maintain an independent tracking identity.

### Status

PASSED

---

## 5. Test Case: Visitor Exit

### Objective

Verify that a visitor is correctly identified as having exited when their tracking session ends.

### Result

The system generated:

TRACK_EXIT | track_id=1 | visitor_id=1

and:

EXIT | visitor_id=1 | track_id=1

An exit image was also saved.

### Expected Behavior

- Track association is removed.
- EXIT event is recorded.
- Timestamp is stored.
- Exit face image is saved.
- Existing visitor record remains in the database.

### Status

PASSED

---

## 6. Test Case: Visitor Re-Entry

### Objective

Verify that a previously registered visitor is recognized after leaving and returning.

### Result

The visitor returned multiple times using different tracking sessions.

The system reused the existing visitor ID instead of creating duplicate visitor records.

Observed track IDs included:

- track_id=1
- track_id=4
- track_id=5

All were associated with:

visitor_id=1

### Expected Behavior

A returning visitor must be recognized as the same unique visitor.

### Status

PASSED

---

## 7. Test Case: Embedding Generation

### Objective

Verify that InsightFace generates a valid face embedding.

### Result

InsightFace generated a 512-dimensional embedding.

Example:

Embedding shape: (512,)

The embedding is normalized before similarity comparison.

### Status

PASSED

---

## 8. Test Case: Persistent Embedding Storage

### Objective

Verify that visitor embeddings are stored in the database.

### Result

The SQLite database contains entries in the visitor_embeddings table.

Each stored embedding contains 512 float32 values.

Embedding size:

2048 bytes

### Status

PASSED

---

## 9. Test Case: Event Database Logging

### Objective

Verify that entry and exit events are persisted in SQLite.

### Result

The database contained:

- ENTRY events
- EXIT events

During validation, the database contained:

ENTRY: 1
EXIT: 5

The event records include visitor ID, event type, timestamp, and image path.

### Status

PASSED

---

## 10. Test Case: Application Log

### Objective

Verify that the mandatory events.log records the AI pipeline activities.

### Result

The log contained events including:

- RECOGNITION_STARTED
- EMBEDDING_GENERATED
- FACE_RECOGNIZED
- EMBEDDING_TEMPLATE_ADDED
- VISITOR_RECOGNIZED
- TRACK_EXIT
- EXIT

### Status

PASSED

---

## 11. Test Case: Face Image Logging

### Objective

Verify that event-related face images are stored locally.

### Result

Timestamped exit images were successfully generated under:

logs/exits/YYYY-MM-DD/

Entry images were also generated during new visitor registration tests.

### Status

PASSED

---

## 12. Test Case: Unique Visitor Counting

### Objective

Verify that repeated appearances of the same person do not create duplicate visitor identities.

### Result

The same visitor was recognized across multiple tracking sessions.

The existing visitor ID was reused.

No duplicate visitor was created during the re-entry tests.

### Status

PASSED

---

## 13. Overall Validation

The implemented system successfully demonstrated the core visitor tracking pipeline:

Face Detection
→ Multi-Object Tracking
→ Face Embedding
→ Face Recognition
→ New Visitor Registration
→ Entry Logging
→ Continuous Tracking
→ Exit Detection
→ Re-Entry Recognition
→ Unique Visitor Identification

The tested components operated together using the configured YOLO, ByteTrack, InsightFace, OpenCV, and SQLite pipeline.

## 14. Validation Status

Core pipeline: PASSED

Face detection: PASSED

Multi-person tracking: PASSED

Face embedding generation: PASSED

New visitor registration: PASSED

Existing visitor recognition: PASSED

Exit detection: PASSED

Re-entry recognition: PASSED

Database persistence: PASSED

Event logging: PASSED

Image logging: PASSED

Unique visitor handling: PASSED
