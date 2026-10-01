# Intelligent Face Tracker

An AI-powered real-time visitor tracking and unique visitor counting system using face detection, multi-object tracking, face recognition, automatic visitor registration, and event logging.

## 1. Project Overview

The Intelligent Face Tracker is designed to detect, track, recognize, and count unique visitors from a video or camera stream.

The system identifies whether a detected face belongs to an already registered visitor. If the face is not recognized after multiple consistent observations, the system automatically registers it as a new visitor with a unique visitor ID.

The system also records visitor entry and exit events with timestamps and stores face images and event metadata for later verification.

## 2. Problem Statement

Traditional visitor counting systems may count the same person multiple times when the person appears in different frames or temporarily leaves and reappears.

This project addresses that problem by combining:

- AI-based face detection
- Multi-object tracking
- Face embeddings
- Similarity-based face recognition
- Automatic visitor registration
- Persistent visitor identities
- Entry and exit event logging
- SQLite database storage

The objective is to maintain a reliable unique visitor count instead of simply counting detected faces in individual frames.

## 3. System Architecture

    VIDEO / CAMERA / RTSP
             |
             v
      YOLO Face Detection
             |
             v
          ByteTrack
             |
             v
       Stable Track IDs
             |
             v
      InsightFace / ArcFace
             |
             v
        Face Embedding
             |
       +-----+-----+
       |           |
       v           v
   Known Face   New Face
       |           |
       v           v
  Recognize ID  Confirmation
                   |
                   v
            Auto Registration
                   |
          +--------+--------+
          |        |        |
          v        v        v
       ENTRY   TRACKING    EXIT
          |        |        |
          +--------+--------+
                   |
          +--------+--------+
          |        |        |
          v        v        v
       SQLite  events.log  Face Images
                   |
                   v
          UNIQUE VISITOR COUNT

## 4. Technology Stack

### Programming Language

- Python 3.10

### Computer Vision

- OpenCV
- YOLO
- ByteTrack

### Face Recognition

- InsightFace
- ArcFace face embeddings

### Database

- SQLite

### Configuration

- JSON

### Logging

- Python logging
- Local event log

## 5. Main Features

### Face Detection

YOLO is used to detect faces from the input video stream.

### Multi-Object Tracking

ByteTrack provides stable track IDs for detected faces across video frames.

### Face Recognition

InsightFace generates a 512-dimensional face embedding for detected faces.

The embedding is compared with stored visitor embeddings using cosine similarity.

### Automatic Visitor Registration

When a face does not match an existing visitor, the system does not immediately create a new visitor.

The face is first kept as a pending track and multiple consistent observations are required.

After successful confirmation, a new visitor ID is created automatically.

### Unique Visitor Counting

Each registered visitor receives a persistent visitor ID.

If the same person leaves and later appears again, their face embedding is compared with stored visitor embeddings and the existing visitor ID is reused.

Therefore, the same person is not counted as a new visitor.

### Entry and Exit Events

The system records:

- Visitor ID
- Track ID
- Event type
- Timestamp
- Face image path

Supported events include:

- ENTRY
- TRACKING
- EXIT

### Persistent Storage

Visitor information, embeddings, and event metadata are stored in SQLite.

### Event Logging

Important system operations are recorded in:

    logs/events.log

The log includes events such as:

- Recognition started
- Embedding generated
- Face recognized
- New face pending
- Visitor registered
- Embedding template added
- Tracking
- Track exit
- Visitor exit

## 6. Project Structure

    intelligent-face-tracker/
    |
    +-- database/
    |   +-- visitors.db
    |
    +-- docs/
    |
    +-- logs/
    |   +-- events.log
    |   +-- entries/
    |   +-- exits/
    |
    +-- models/
    |   +-- face_detection_yunet_2023mar.onnx
    |   +-- yolov8n-face.pt
    |
    +-- output/
    |
    +-- src/
    |   +-- main.py
    |   +-- face_detector.py
    |   +-- face_tracker.py
    |   +-- face_recognizer.py
    |   +-- visitor_manager.py
    |   +-- database.py
    |   +-- logger.py
    |   +-- image_logger.py
    |   +-- ...
    |
    +-- tests/
    |
    +-- videos/
    |
    +-- app.py
    +-- config.json
    +-- requirements.txt
    +-- README.md
    +-- .gitignore

## 7. Configuration

The system uses config.json for runtime configuration.

Example configuration:

    {
        "input": {
            "source": "0",
            "type": "camera"
        },
        "detection": {
            "confidence": 0.5,
            "skip_frames": 5
        },
        "recognition": {
            "similarity_threshold": 0.5
        },
        "tracking": {
            "max_missing_frames": 30
        },
        "output": {
            "show_window": true,
            "save_images": true
        }
    }

### Input Source

    "source": "0"

uses the default camera.

A video file or RTSP stream can also be supplied through the input source configuration.

### Detection

    "confidence": 0.5

controls the detection confidence threshold.

    "skip_frames": 5

controls the configured detection frame interval.

### Recognition

    "similarity_threshold": 0.5

controls the minimum similarity required to recognize a stored visitor.

### Tracking

    "max_missing_frames": 30

controls how long a missing track can remain active.

## 8. Installation

### Step 1: Clone the repository

    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd intelligent-face-tracker

### Step 2: Create a virtual environment

    python -m venv .venv

### Step 3: Activate the environment

Windows PowerShell:

    .venv\Scripts\Activate.ps1

### Step 4: Install dependencies

    pip install -r requirements.txt

## 9. Running the System

From the project root:

    python src\main.py

The configured input source is loaded from:

    config.json

The application opens the configured camera or video source and processes the detected faces.

## 10. Database

The system uses SQLite:

    database/visitors.db

### Visitors

The visitor database stores:

- Visitor ID
- First seen timestamp
- Last seen timestamp
- Face embedding

### Visitor Embeddings

Multiple embedding templates can be associated with a visitor to improve recognition across different observations.

### Events

Events store:

- Visitor ID
- Event type
- Timestamp
- Image path

## 11. Event Logging

Runtime events are written to:

    logs/events.log

Example:

    2026-10-01 20:48:44 | INFO | FACE_RECOGNIZED | track_id=1 | visitor_id=1 | similarity=0.849
    2026-10-01 20:48:44 | INFO | VISITOR_RECOGNIZED | visitor_id=1 | track_id=1
    2026-10-01 20:49:10 | INFO | TRACK_EXIT | track_id=1 | visitor_id=1
    2026-10-01 20:49:10 | INFO | EXIT | visitor_id=1 | track_id=1

## 12. Image Logging

Face images associated with events are stored locally.

Example structure:

    logs/
    +-- entries/
    |   +-- YYYY-MM-DD/
    |
    +-- exits/
        +-- YYYY-MM-DD/

Images are timestamped to make individual events traceable.

## 13. Recognition Flow

The recognition process follows these steps:

1. YOLO detects a face.
2. ByteTrack assigns a tracking ID.
3. InsightFace generates a face embedding.
4. The embedding is compared with stored visitor embeddings.
5. If a match is found, the existing visitor ID is reused.
6. If no match is found, the track enters a pending state.
7. Multiple consistent observations are required.
8. After confirmation, a new visitor is registered.
9. The visitor ID remains associated with the active track.
10. When the track exits, an EXIT event is recorded.

## 14. Unique Visitor Counting

The unique visitor count is based on persistent visitor IDs rather than the number of detected frames.

Example:

    Person A enters
        |
        v
    Visitor ID = 1
        |
        v
    Person A leaves
        |
        v
    EXIT recorded
        |
        v
    Person A enters again
        |
        v
    Face recognized as Visitor ID = 1
        |
        v
    Unique visitor count = 1

The same person therefore does not create a duplicate visitor record when returning.

## 15. Logging and Traceability

The system maintains traceability between:

    Track ID
        |
        v
    Visitor ID
        |
        v
    Database Record
        |
        v
    Event
        |
        v
    Timestamp
        |
        v
    Face Image

This allows recognition and entry/exit events to be inspected after processing.

## 16. Current Implementation

The implemented pipeline includes:

- YOLO face detection
- ByteTrack tracking
- InsightFace face embeddings
- Cosine similarity based recognition
- Pending-face confirmation
- Automatic visitor registration
- Persistent visitor embeddings
- SQLite storage
- Entry and exit event handling
- Timestamped face image storage
- Runtime event logging
- Unique visitor counting
- Configurable runtime parameters

## 17. Testing

The system has been tested with:

- Single-person detection
- Repeated recognition of the same person
- Multiple-person tracking
- Visitor exit detection
- Re-entry recognition
- Persistent visitor ID reuse
- Database event storage
- Embedding storage
- Entry/exit image logging

## 18. Future Improvements

Possible extensions include:

- RTSP camera deployment
- Web-based monitoring dashboard
- Real-time visitor statistics
- Multiple camera support
- GPU acceleration
- Advanced track management
- Centralized PostgreSQL deployment
- Role-based access control
- Remote event monitoring
- Deployment using Docker

## 19. Hackathon

This project is developed as part of the visitor tracking and unique visitor counting challenge.

This project is a part of a hackathon run by https://katomaran.com
