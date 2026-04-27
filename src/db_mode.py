import json
import os
from config import SETTINGS_PATH, DEFAULT_DB_MODE


def ensure_settings_folder():
    folder = os.path.dirname(SETTINGS_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)


def load_settings():
    ensure_settings_folder()

    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    settings = {"db_mode": DEFAULT_DB_MODE}
    save_settings(settings)
    return settings


def save_settings(settings):
    ensure_settings_folder()

    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)


def get_db_mode():
    settings = load_settings()
    return settings.get("db_mode", DEFAULT_DB_MODE)


def set_db_mode(mode):
    if mode not in ["local", "remote", "cloud"]:
        raise ValueError("Mode must be 'local', 'remote', or 'cloud'.")

    settings = load_settings()
    settings["db_mode"] = mode
    save_settings(settings)