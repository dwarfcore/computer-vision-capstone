# Vision Monitor — Facial Recognition Monitoring System

A desktop application that monitors people in a camera's field of view,
identifies them using facial recognition, and maintains a searchable database
of persons and their appearances.

---

## Features

| Feature | Status |
|---|---|
| Live webcam feed with face detection overlay | ✅ |
| 128-d face embedding + vector similarity search | ✅ |
| SQLite (local) relational database | ✅ |
| ChromaDB (local) vector database | ✅ |
| Remote PostgreSQL + Qdrant support | ✅ (stub — enable in requirements.txt) |
| Cloud Supabase + Pinecone support | ✅ (stub — enable in requirements.txt) |
| GUI settings panel (switch DB mode at runtime) | ✅ |
| Add person by webcam capture or photo upload | ✅ |
| Appearance log with timestamps and confidence | ✅ |
| Face snapshots saved to disk | ✅ |
| Audible + visual alert for monitored persons | ✅ |
| Mark/unmark monitored flag per person | ✅ |
| Delete person (cascades appearances + embeddings) | ✅ |

---

## Project Structure

```
capstone-project/
├── src/
│   ├── main.py               Entry point — launches GUI
│   ├── config.py             App + DB config, load/save JSON
│   ├── db_sqlite.py          Relational DB: persons & appearances
│   ├── db_vector.py          Vector DB factory (Chroma / Qdrant / Pinecone)
│   ├── face_processing.py    Detection, encoding, matching, drawing
│   └── gui/
│       ├── app.py            Main window (camera feed + person list)
│       ├── settings_dialog.py Database location & app settings
│       └── person_dialog.py  Add / edit person with face capture
├── data/                     Created at runtime
│   ├── monitoring.db         SQLite database (local mode)
│   ├── vectors/              ChromaDB persistence (local mode)
│   └── snapshots/            Face crops saved per appearance
├── config.json               Saved settings (created after first save)
└── requirements.txt
```

---

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note on `face-recognition`**: Requires `dlib`, which requires cmake.
> - **Windows**: Install from pre-built wheels — search PyPI for `dlib` wheels
>   or use `conda install -c conda-forge dlib` then `pip install face-recognition`.
> - **macOS**: `brew install cmake` then `pip install face-recognition`.
> - **Linux**: `sudo apt install cmake libopenblas-dev` then `pip install face-recognition`.

### 2. Run

```bash
cd src
python main.py
```

---

## Database Modes

Configure via the ⚙ Settings button or by editing `config.json`.

### Local (default)
- **SQLite** at `data/monitoring.db`
- **ChromaDB** at `data/vectors/`
- No server required — works completely offline.

### Remote
- **PostgreSQL** (provide host, port, DB name, credentials)
- **Qdrant** (provide URL, e.g. `http://myserver:6333`)
- Enable in `requirements.txt`: uncomment `psycopg2-binary` and `qdrant-client`
- Run Qdrant: `docker run -p 6333:6333 qdrant/qdrant`

### Cloud
- **Supabase** (PostgreSQL-compatible, provide project URL + anon key)
- **Pinecone** (provide index host + API key)
- Enable in `requirements.txt`: uncomment `psycopg2-binary` and `pinecone-client`
- Create a Pinecone index with dimension=128 and cosine metric.

---

## How it Works

1. **Detection**: Each Nth frame (configurable) is downscaled and passed to
   `face_recognition.face_locations()` using the HOG model (CPU-friendly).
2. **Encoding**: Detected faces are passed to `face_recognition.face_encodings()`
   to produce a 128-dimensional float vector per face.
3. **Matching**: The embedding is queried against the vector database.
   If the nearest neighbour's distance is below the configured threshold,
   the face is identified as that person.
4. **Logging**: Each confirmed sighting logs a row in the `appearances` table
   with timestamp, confidence, and snapshot path.
5. **Alerting**: If the matched person has `is_monitored = 1`, the app rings
   a bell and logs a red alert entry.

---

## Tuning Tips

| Parameter | Default | Effect |
|---|---|---|
| `detection_confidence` | 0.6 | Lower = stricter matching (fewer false positives) |
| `frame_skip` | 5 | Higher = less CPU usage, lower detection frequency |
| `camera_index` | 0 | Change if using an external webcam |

For GPU-accelerated detection, change `model="hog"` to `model="cnn"` in
`face_processing.py::process_frame()`. Requires CUDA + dlib compiled with CUDA.

---

## Extending

- **Multi-camera**: Run multiple `_camera_loop` threads, each with a different
  `camera_index`. Add a camera selector to the GUI.
- **REST API**: Add a FastAPI server in `api/` that exposes `/persons`,
  `/appearances`, and a `/webhook` endpoint for monitored-person alerts.
- **Email/SMS alerts**: Hook into `_handle_detections` to send a notification
  via `smtplib` or Twilio when a monitored person is detected.
