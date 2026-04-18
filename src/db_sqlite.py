import os
import sqlite3
from datetime import datetime
from config import DB_PATH


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def setup_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        person_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        is_monitored INTEGER NOT NULL DEFAULT 0,
        notes TEXT DEFAULT ''
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS appearances (
        appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        event_type TEXT NOT NULL,
        FOREIGN KEY(person_id) REFERENCES persons(person_id)
    )
    """)

    conn.commit()
    conn.close()


def add_person(name, is_monitored=0, notes=""):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO persons (name, is_monitored, notes)
    VALUES (?, ?, ?)
    """, (name, int(is_monitored), notes))

    conn.commit()
    person_id = cur.lastrowid
    conn.close()
    return person_id


def get_person(person_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
    row = cur.fetchone()

    conn.close()
    return dict(row) if row else None


def rename_person(person_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    UPDATE persons
    SET name = ?
    WHERE person_id = ?
    """, (new_name, person_id))

    conn.commit()
    conn.close()


def set_monitored(person_id, monitored, notes=""):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    UPDATE persons
    SET is_monitored = ?, notes = ?
    WHERE person_id = ?
    """, (int(monitored), notes, person_id))

    conn.commit()
    conn.close()


def log_appearance(person_id, event_type):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO appearances (person_id, timestamp, event_type)
    VALUES (?, ?, ?)
    """, (person_id, get_timestamp(), event_type))

    conn.commit()
    conn.close()


def list_persons():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM persons ORDER BY person_id")
    rows = cur.fetchall()

    conn.close()
    return [dict(row) for row in rows]


def list_appearances():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
    SELECT a.appearance_id, a.person_id, p.name, a.timestamp, a.event_type
    FROM appearances a
    JOIN persons p ON a.person_id = p.person_id
    ORDER BY a.appearance_id DESC
    """)
    rows = cur.fetchall()

    conn.close()
    return [dict(row) for row in rows]