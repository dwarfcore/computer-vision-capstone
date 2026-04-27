"""
config.py - Configuration management for database location and app settings.
Supports local SQLite, remote PostgreSQL, and cloud (Supabase/PlanetScale) modes.
"""
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class DatabaseMode(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
    CLOUD = "cloud"


@dataclass
class DatabaseConfig:
    mode: DatabaseMode = DatabaseMode.LOCAL

    # Local SQLite
    local_sqlite_path: str = "data/monitoring.db"
    local_vector_path: str = "data/vectors"

    # Remote PostgreSQL
    remote_host: str = ""
    remote_port: int = 5432
    remote_db: str = "monitoring"
    remote_user: str = ""
    remote_password: str = ""
    remote_vector_url: str = ""  # e.g., pgvector connection string or Qdrant URL

    # Cloud (Supabase example)
    cloud_url: str = ""
    cloud_api_key: str = ""
    cloud_vector_url: str = ""  # e.g., Pinecone / Weaviate endpoint
    cloud_vector_api_key: str = ""


@dataclass
class AppConfig:
    db: DatabaseConfig = None
    detection_confidence: float = 0.6   # face_recognition distance threshold
    frame_skip: int = 5                 # process every Nth frame
    alert_on_monitored: bool = True     # trigger visual/audio alert
    save_snapshots: bool = True         # save JPG on first detection
    snapshot_dir: str = "data/snapshots"
    camera_index: int = 0

    def __post_init__(self):
        if self.db is None:
            self.db = DatabaseConfig()


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def load_config() -> AppConfig:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                raw = json.load(f)
            db_raw = raw.pop("db", {})
            db_mode = DatabaseMode(db_raw.pop("mode", "local"))
            db_cfg = DatabaseConfig(mode=db_mode, **{
                k: v for k, v in db_raw.items() if hasattr(DatabaseConfig, k)
            })
            cfg = AppConfig(db=db_cfg, **{
                k: v for k, v in raw.items() if hasattr(AppConfig, k)
            })
            return cfg
        except Exception as e:
            print(f"[Config] Failed to load config.json: {e}. Using defaults.")
    return AppConfig()


def save_config(cfg: AppConfig) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    data = asdict(cfg)
    data["db"]["mode"] = cfg.db.mode.value
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
