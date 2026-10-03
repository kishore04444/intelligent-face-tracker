# VisitorIQ Dashboard

Professional local dashboard for the Intelligent Face Tracker.

## Live data
The UI reads the Python runtime APIs directly:
- `/api/stats` — current-session unique visitors, occupancy, entries and exits
- `/api/events` — current-session entry/exit events
- `/api/visitors` — persistent visitor registry
- `/api/stream` — live annotated camera feed

## Design goals
- No hard-coded visitor metrics on the overview
- Current-session metrics separated from persistent visitor history
- Live camera feed from the Python tracker
- Derived operational metrics such as return visits, peak occupancy, average completed visit duration and exit rate
- Responsive dark operations-console style
