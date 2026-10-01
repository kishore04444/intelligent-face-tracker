# AI Planning Document

## 1. Project Title

Intelligent Face Tracker

## 2. Objective

Build an AI-driven real-time visitor tracking system that detects, tracks, recognizes, and uniquely counts visitors.

The system automatically registers new faces, recognizes previously registered visitors, tracks their movement, records entry and exit events, and maintains a persistent visitor database.

## 3. AI Pipeline

The system follows this pipeline:

Camera / Video / RTSP
        ↓
YOLO Face Detection
        ↓
ByteTrack Multi-Object Tracking
        ↓
InsightFace / ArcFace Face Embedding
        ↓
Face Matching
        ↓
Known Visitor OR New Visitor
        ↓
Entry / Tracking / Exit
        ↓
SQLite Database + Logs + Images
        ↓
Unique Visitor Count

## 4. Face Detection

YOLO is used to detect faces in each processed video frame.

For every detected face, the system obtains a bounding box and confidence score.

The detection stage provides the input required by the tracking and recognition stages.

## 5. Multi-Object Tracking

ByteTrack is used to maintain stable track IDs across video frames.

A track ID allows the system to associate the same detected person across multiple frames.

This prevents the same person from being counted repeatedly while they remain visible.

## 6. Face Recognition

InsightFace with ArcFace embeddings is used for face recognition.

A detected face is converted into a 512-dimensional face embedding.

The embedding is normalized before comparison.

The system compares the current embedding against embeddings stored in the visitor database using cosine similarity.

## 7. Known Visitor Recognition

If the similarity between the current face embedding and a stored visitor embedding is greater than or equal to the configured recognition threshold, the visitor is considered recognized.

The existing visitor ID is reused.

A new visitor is not created.

Additional high-confidence embeddings can be stored as templates for improving recognition across different appearances and frames.

## 8. New Visitor Registration

When a face does not match any existing visitor, the system does not immediately create a new visitor.

The face first enters a pending state.

Multiple observations of the same track are compared for consistency.

After the required confirmation observations are obtained, the face is automatically registered as a new visitor with a unique visitor ID.

## 9. Entry Detection

When a new visitor is successfully registered, an ENTRY event is created.

The system records:

- Visitor ID
- Timestamp
- Event type
- Face image path
- Track ID in the application log

The cropped face image is stored locally.

## 10. Tracking

While a visitor remains visible, the system maintains the association between:

Track ID → Visitor ID

Tracking events are written to the application log.

The visitor's last-seen timestamp is also updated in the database.

## 11. Exit Detection

When a tracked visitor is no longer detected for the configured tracking period, the track is considered exited.

An EXIT event is recorded.

A cropped face image is saved for the exit event.

The track-to-visitor association is removed so that a later appearance can be treated as a new tracking session.

## 12. Re-Entry Handling

When a previously registered visitor appears again, the new track is compared with the stored visitor embeddings.

If a match is found, the existing visitor ID is reused.

The system therefore does not create another visitor record for the same person.

This allows the database to maintain a unique visitor count across multiple visits.

## 13. Embedding Management

Each visitor can have multiple stored face embeddings.

Additional embeddings can be added when a recognition result has sufficiently high similarity.

This allows the recognition database to contain multiple representations of the same visitor.

The stored embeddings are persisted in SQLite.

## 14. Database Design

SQLite is used as the local persistent database.

### visitors

Stores the main visitor records.

Fields include:

- id
- first_seen
- last_seen
- embedding

### visitor_embeddings

Stores additional face embeddings associated with visitors.

Fields include:

- id
- visitor_id
- embedding
- created_at

### events

Stores visitor entry and exit events.

Fields include:

- id
- visitor_id
- event_type
- timestamp
- image_path

## 15. Event Logging

The system maintains a mandatory application log at:

logs/events.log

The log records important AI pipeline activities including:

- Face recognition attempts
- Embedding generation
- Face recognition
- New face detection
- Visitor registration
- Tracking
- Track exit
- Entry events
- Exit events
- Embedding template creation

This provides traceability for the complete visitor lifecycle.

## 16. Image Logging

Face images associated with events are stored locally using date-based folders.

The intended structure is:

logs/
├── entries/
│   └── YYYY-MM-DD/
└── exits/
    └── YYYY-MM-DD/

Each image filename contains the event type and timestamp.

## 17. Configuration

Runtime parameters are controlled through config.json.

Important configurable parameters include:

- Input source
- Detection confidence
- Detection frame skip
- Recognition similarity threshold
- Maximum missing tracking frames
- Display window
- Image saving

This allows the system to be adjusted without modifying the core application code.

## 18. Unique Visitor Counting

The unique visitor count is based on persistent visitor IDs stored in the database.

A visitor may generate multiple tracking sessions and multiple entry/exit events.

However, recognition of the same visitor reuses the existing visitor ID.

Therefore, repeated appearances do not create duplicate visitor records.

## 19. AI Decision Flow

For each tracked face:

1. Detect the face using YOLO.
2. Assign or maintain a ByteTrack track ID.
3. Generate a face embedding using InsightFace.
4. Compare the embedding against stored visitor embeddings.
5. If a match is found, recognize the existing visitor.
6. If no match is found, place the track into pending confirmation.
7. Confirm consistent observations.
8. Register a new visitor if confirmation succeeds.
9. Record entry and tracking information.
10. Detect track disappearance.
11. Record the exit event.
12. Allow the same visitor to be recognized again during a future appearance.

## 20. Current AI Components

| Component | Technology |
|---|---|
| Face Detection | YOLO |
| Multi-Object Tracking | ByteTrack |
| Face Recognition | InsightFace |
| Face Embedding | ArcFace |
| Similarity Matching | Cosine Similarity |
| Database | SQLite |
| Image Processing | OpenCV |
| Configuration | JSON |
| Logging | Python Logging |

## 21. Design Goals

The AI pipeline is designed around:

- Unique visitor identification
- Persistent visitor identities
- Reliable tracking
- Automatic registration
- Entry and exit traceability
- Persistent face embeddings
- Local image evidence
- Database-backed visitor records
- Configurable runtime parameters
- Modular components

## 22. Future Improvements

Potential future improvements include:

- RTSP camera deployment
- GPU acceleration
- Improved re-identification
- Web dashboard
- Remote database support
- Real-time analytics
- Multiple camera support
- Camera-to-camera visitor tracking
- Advanced monitoring and reporting

