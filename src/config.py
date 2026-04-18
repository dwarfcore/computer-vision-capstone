import os

DATA_FOLDER = "monitoring_data"
DB_PATH = os.path.join(DATA_FOLDER, "monitoring.db")
VECTOR_DB_PATH = os.path.join(DATA_FOLDER, "vector_db.npz")
FACE_FOLDER = os.path.join(DATA_FOLDER, "faces")
SETTINGS_PATH = os.path.join(DATA_FOLDER, "settings.json")

MATCH_THRESHOLD = 0.88
DEFAULT_DB_MODE = "local"