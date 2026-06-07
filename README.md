# 🕉️ Ujjain Mahakumbh 2028 — Transportation & Mobility Portal

A production-ready Flask web application for managing transportation and mobility
during the Ujjain Mahakumbh 2028 mega-event.

![Alt Text](static/Screenshot 2026-06-07 124928.jpeg)

---

## 📁 Project Structure

```
ujjain_mahakumbh/
├── app.py                  # Main Flask app with all routes & video streaming
├── database.db             # SQLite database (auto-created on first run)
├── requirements.txt
├── templates/
│   ├── base.html           # Shared layout with nav & flash messages
│   ├── landing.html        # Public homepage
│   ├── login.html          # Auth – Login
│   ├── register.html       # Auth – Register
│   ├── dashboard.html      # Command Center hub (login required)
│   ├── roads.html          # Live CCTV feeds – 4 road corridors
│   ├── parking.html        # Parking slot availability
│   ├── booking.html        # Multi-step bus seat booking form
│   └── route.html          # Leaflet.js GPS navigation to Ujjain
└── static/
    └── videos/             # Drop MP4 files here for live CCTV feeds
        ├── indore.mp4      # → streamed on "Indore Rd" feed
        ├── dewas.mp4       # → streamed on "Dewas Rd" feed
        ├── ramghat.mp4     # → streamed on "Ram Ghat Rd" feed
        ├── dutt.mp4        # → streamed on "Dutt Akhand Rd" feed
        ├── p_indore.mp4    # → streamed on "Parking – Indore Gate"
        ├── p_dewas.mp4     # → streamed on "Parking – Dewas Gate"
        ├── p_ramghat.mp4   # → streamed on "Parking – Ram Ghat"
        └── p_dutt.mp4      # → streamed on "Parking – Dutt Akhand"
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional: YOLOv8 for real AI vehicle/person counting
pip install ultralytics
```

### 2. Run the app

```bash
python app.py
```

Visit → **http://localhost:5000**

---

## 🎥 Adding Video Feeds

Place your MP4 files in `static/videos/` using the filenames above.

- **Without video files**: The app shows an animated demo feed with simulated counters.
- **With video files + YOLOv8**: Full AI-powered vehicle and person counting with
  bytetrack multi-object tracking and counting line crossing detection.

### YOLOv8 Integration

In `app.py`, the YOLO logic uses:
- `model.track(frame, persist=True, tracker="bytetrack.yaml")` for tracking
- Center-point based counting line crossing (set via `COUNTING_LINE_Y_RATIO = 0.5`)
- Class filtering: vehicles (car/motorbike/bus/truck) and persons counted separately
- Live overlay: counts, timestamps, and labeled counting line drawn on each frame

---

## 🗺️ Features

| Page | URL | Description |
|------|-----|-------------|
| Landing | `/` | Public homepage with Mahakumbh theme |
| Register | `/register` | Create account |
| Login | `/login` | Authenticate |
| Dashboard | `/dashboard` | Command Center hub |
| Roads | `/roads` | 4× MJPEG CCTV streams + AI counts |
| Parking | `/parking` | 4× parking feeds + slot availability |
| Booking | `/booking` | Multi-step bus seat reservation |
| Route | `/route` | Leaflet.js map with GPS routing |

---

## 🛡️ Security Notes

- Passwords are hashed with Werkzeug's PBKDF2-SHA256.
- All internal pages require an active session (`@login_required`).
- Change `SECRET_KEY` via environment variable before deploying:
  ```bash
  export SECRET_KEY="your-strong-random-key"
  ```

---

## 🎨 Design

- **Theme**: Deep saffron / maroon / gold — culturally authentic to Mahakumbh
- **Fonts**: Cinzel Decorative (headings) + Cormorant Garamond (body) + Inter (UI)
- **Framework**: Tailwind CSS CDN + vanilla JS
- **Map**: Leaflet.js with OpenStreetMap tiles (no API key needed)
