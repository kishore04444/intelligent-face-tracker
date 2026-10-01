# Database Evidence

## Database

The project uses SQLite for persistent visitor and event storage.

Database file used by the application:

database/visitors.db

The actual database file is excluded from version control because it may contain runtime visitor data.

## Tables

### 1. visitors

Stores the primary visitor identity.

| Field | Description |
|---|---|
| id | Unique visitor ID |
| first_seen | First recorded appearance |
| last_seen | Most recent appearance |
| embedding | Primary face embedding |

### 2. visitor_embeddings

Stores face embeddings associated with registered visitors.

| Field | Description |
|---|---|
| id | Embedding record ID |
| visitor_id | Associated visitor |
| embedding | 512-dimensional float32 embedding |
| created_at | Embedding creation timestamp |

### 3. events

Stores visitor entry and exit events.

| Field | Description |
|---|---|
| id | Event ID |
| visitor_id | Associated visitor |
| event_type | ENTRY or EXIT |
| timestamp | Event timestamp |
| image_path | Associated face image |

## Validation Evidence

During testing, the database contained:

- Unique visitor: visitor_id=1
- ENTRY events: 1
- EXIT events: 5
- Stored face embeddings for the registered visitor
- Embedding size: 512 float32 values
- Embedding storage size: 2048 bytes

## Re-Entry Validation

The same visitor was detected across multiple tracking sessions.

Observed track IDs included:

- track_id=1
- track_id=4
- track_id=5

These tracking sessions were associated with the same:

visitor_id=1

No duplicate visitor identity was created during the re-entry tests.

## Event Example

An exit event was successfully stored with an associated image path:

logs/exits/YYYY-MM-DD/exit_<timestamp>.jpg

## Persistence

Visitor identity and event information are persisted in SQLite, allowing the unique visitor count and event history to remain available across application runs.
