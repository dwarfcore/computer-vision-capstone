"""
db_sqlite.py - Relational database layer using SQLite (local) or PostgreSQL (remote/cloud).

Tables:
  persons      - id, name, notes, is_monitored, created_at, thumbnail_path
  appearances  - id, person_id, timestamp, snapshot_path, confidence
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from config import AppConfig, DatabaseMode


def _get_connection(cfg: AppConfig) -> sqlite3.Connection:
    """Return a SQLite connection. For remote/cloud PostgreSQL, swap this for psycopg2."""
    mode = cfg.db.mode
    if mode == DatabaseMode.LOCAL:
        db_path = cfg.db.local_sqlite_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path, check_same_thread=False)
    elif mode in (DatabaseMode.REMOTE, DatabaseMode.CLOUD):
        # --- PostgreSQL via psycopg2 ---
        # Uncomment and install psycopg2-binary for production remote/cloud use:
        #
        # import psycopg2
        # if mode == DatabaseMode.REMOTE:
        #     return psycopg2.connect(
        #         host=cfg.db.remote_host, port=cfg.db.remote_port,
        #         dbname=cfg.db.remote_db, user=cfg.db.remote_user,
        #         password=cfg.db.remote_password
        #     )
        # else:  # cloud (e.g. Supabase postgres endpoint)
        #     return psycopg2.connect(cfg.db.cloud_url)
        #
        # For now, fall back to local SQLite if remote not configured:
        print(f"[DB] Remote/cloud PostgreSQL not connected - falling back to local SQLite")
        db_path = cfg.db.local_sqlite_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path, check_same_thread=False)
    raise ValueError(f"Unknown DB mode: {mode}")


def init_db(cfg: AppConfig) -> sqlite3.Connection:
    conn = _get_connection(cfg)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL DEFAULT 'Unknown',
            notes           TEXT DEFAULT '',
            is_monitored    INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now')),
            thumbnail_path  TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS appearances (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id       INTEGER NOT NULL,
            timestamp       TEXT DEFAULT (datetime('now')),
            snapshot_path   TEXT DEFAULT '',
            confidence      REAL DEFAULT 0.0,
            FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_appearances_person ON appearances(person_id);
        CREATE INDEX IF NOT EXISTS idx_appearances_ts ON appearances(timestamp);
    """)
    conn.commit()
    return conn


# ---------- Person CRUD ----------

def add_person(conn, name: str = "Unknown", notes: str = "", thumbnail_path: str = "") -> int:
    c = conn.cursor()
    c.execute(
        "INSERT INTO persons (name, notes, thumbnail_path) VALUES (?, ?, ?)",
        (name, notes, thumbnail_path)
    )
    conn.commit()
    return c.lastrowid


def get_person(conn, person_id: int) -> Optional[Dict]:
    c = conn.cursor()
    c.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
    row = c.fetchone()
    return _row_to_person(row) if row else None


def get_all_persons(conn) -> List[Dict]:
    c = conn.cursor()
    c.execute("SELECT * FROM persons ORDER BY created_at DESC")
    return [_row_to_person(r) for r in c.fetchall()]


def update_person(conn, person_id: int, **fields) -> None:
    allowed = {"name", "notes", "is_monitored", "thumbnail_path"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    c = conn.cursor()
    c.execute(f"UPDATE persons SET {set_clause} WHERE id = ?", (*updates.values(), person_id))
    conn.commit()


def delete_person(conn, person_id: int) -> None:
    c = conn.cursor()
    c.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.commit()


def _row_to_person(row) -> Dict:
    keys = ["id", "name", "notes", "is_monitored", "created_at", "thumbnail_path"]
    return dict(zip(keys, row))


# ---------- Appearances ----------

def log_appearance(conn, person_id: int, snapshot_path: str = "", confidence: float = 0.0) -> int:
    c = conn.cursor()
    c.execute(
        "INSERT INTO appearances (person_id, snapshot_path, confidence) VALUES (?, ?, ?)",
        (person_id, snapshot_path, confidence)
    )
    conn.commit()
    return c.lastrowid


def get_appearances(conn, person_id: int) -> List[Dict]:
    c = conn.cursor()
    c.execute(
        "SELECT id, person_id, timestamp, snapshot_path, confidence "
        "FROM appearances WHERE person_id = ? ORDER BY timestamp DESC",
        (person_id,)
    )
    keys = ["id", "person_id", "timestamp", "snapshot_path", "confidence"]
    return [dict(zip(keys, r)) for r in c.fetchall()]


def get_recent_appearances(conn, limit: int = 50) -> List[Dict]:
    c = conn.cursor()
    c.execute("""
        SELECT a.id, a.person_id, p.name, a.timestamp, a.snapshot_path, a.confidence
        FROM appearances a
        JOIN persons p ON a.person_id = p.id
        ORDER BY a.timestamp DESC
        LIMIT ?
    """, (limit,))
    keys = ["id", "person_id", "name", "timestamp", "snapshot_path", "confidence"]
    return [dict(zip(keys, r)) for r in c.fetchall()]


def get_appearance_count(conn, person_id: int) -> int:
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM appearances WHERE person_id = ?", (person_id,))
    return c.fetchone()[0]
