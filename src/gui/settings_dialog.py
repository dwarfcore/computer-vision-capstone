"""
gui/settings_dialog.py - Settings window for database location and app preferences.

Lets the user switch between Local (SQLite + ChromaDB), Remote (PostgreSQL + Qdrant),
and Cloud (PostgreSQL/Supabase + Pinecone) with appropriate credential fields.
"""
import tkinter as tk
from tkinter import ttk, filedialog
import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import AppConfig, DatabaseConfig, DatabaseMode

BG     = "#0f1117"
PANEL  = "#1a1d27"
ACCENT = "#00e5a0"
DANGER = "#ff4757"
TEXT   = "#e8eaf0"
MUTED  = "#6b7280"
BORDER = "#2a2d3a"
CARD   = "#21253a"
ENTRY_BG = "#12151f"


class SettingsDialog:
    def __init__(self, parent, cfg: AppConfig):
        self.cfg = copy.deepcopy(cfg)
        self.saved = False

        self.window = tk.Toplevel(parent)
        self.window.title("Settings")
        self.window.configure(bg=BG)
        self.window.geometry("560x620")
        self.window.resizable(False, True)
        self.window.grab_set()

        self._build()

    def _build(self):
        tk.Label(self.window, text="SETTINGS", bg=BG, fg=ACCENT,
                 font=("Courier", 13, "bold")).pack(anchor="w", padx=20, pady=(16, 4))

        # ── Database Mode ───────────────────────────────────
        self._section("DATABASE LOCATION")
        mode_frame = tk.Frame(self.window, bg=PANEL)
        mode_frame.pack(fill=tk.X, padx=16, pady=(0, 4))

        self._mode_var = tk.StringVar(value=self.cfg.db.mode.value)
        for label, val in [("Local (SQLite + ChromaDB)", "local"),
                            ("Remote (PostgreSQL + Qdrant)", "remote"),
                            ("Cloud (Supabase + Pinecone)", "cloud")]:
            rb = tk.Radiobutton(mode_frame, text=label, variable=self._mode_var,
                                value=val, bg=PANEL, fg=TEXT, selectcolor=BG,
                                activebackground=PANEL, font=("Courier", 10),
                                command=self._on_mode_change)
            rb.pack(anchor="w", padx=12, pady=2)

        # ── Dynamic fields frame ────────────────────────────
        self._fields_frame = tk.Frame(self.window, bg=BG)
        self._fields_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        self._build_local_fields()
        self._build_remote_fields()
        self._build_cloud_fields()
        self._on_mode_change()

        # ── App Settings ────────────────────────────────────
        self._section("APP SETTINGS")
        app_frame = tk.Frame(self.window, bg=PANEL)
        app_frame.pack(fill=tk.X, padx=16, pady=(0, 8))

        self._conf_var  = tk.DoubleVar(value=self.cfg.detection_confidence)
        self._skip_var  = tk.IntVar(value=self.cfg.frame_skip)
        self._alert_var = tk.BooleanVar(value=self.cfg.alert_on_monitored)
        self._snap_var  = tk.BooleanVar(value=self.cfg.save_snapshots)
        self._cam_var   = tk.IntVar(value=self.cfg.camera_index)

        self._field_row(app_frame, "Detection threshold (0.1–1.0):",
                        tk.Entry(app_frame, textvariable=self._conf_var,
                                 bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                                 relief=tk.FLAT, font=("Courier", 10), width=8))
        self._field_row(app_frame, "Frame skip (process every Nth frame):",
                        tk.Entry(app_frame, textvariable=self._skip_var,
                                 bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                                 relief=tk.FLAT, font=("Courier", 10), width=8))
        self._field_row(app_frame, "Camera index:",
                        tk.Entry(app_frame, textvariable=self._cam_var,
                                 bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                                 relief=tk.FLAT, font=("Courier", 10), width=8))

        tk.Checkbutton(app_frame, text="Alert on monitored person", variable=self._alert_var,
                       bg=PANEL, fg=TEXT, selectcolor=BG, activebackground=PANEL,
                       font=("Courier", 10)).pack(anchor="w", padx=12, pady=2)
        tk.Checkbutton(app_frame, text="Save face snapshots", variable=self._snap_var,
                       bg=PANEL, fg=TEXT, selectcolor=BG, activebackground=PANEL,
                       font=("Courier", 10)).pack(anchor="w", padx=12, pady=2)

        # ── Save / Cancel ───────────────────────────────────
        btn_frame = tk.Frame(self.window, bg=BG)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=16, pady=12)
        tk.Button(btn_frame, text="Save", bg=ACCENT, fg=BG, relief=tk.FLAT,
                  font=("Courier", 11, "bold"), padx=20, cursor="hand2",
                  command=self._save).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Cancel", bg=BORDER, fg=TEXT, relief=tk.FLAT,
                  font=("Courier", 10), padx=16, cursor="hand2",
                  command=self.window.destroy).pack(side=tk.LEFT, padx=8)

    # ── Field builders ──────────────────────────────────────

    def _build_local_fields(self):
        self._local_frame = tk.Frame(self._fields_frame, bg=PANEL)
        db = self.cfg.db
        self._local_sqlite = self._labeled_entry(self._local_frame,
                                                   "SQLite file path:", db.local_sqlite_path)
        self._local_vec    = self._labeled_entry(self._local_frame,
                                                   "ChromaDB directory:", db.local_vector_path)

    def _build_remote_fields(self):
        self._remote_frame = tk.Frame(self._fields_frame, bg=PANEL)
        db = self.cfg.db
        self._r_host    = self._labeled_entry(self._remote_frame, "PostgreSQL host:", db.remote_host)
        self._r_port    = self._labeled_entry(self._remote_frame, "Port:", str(db.remote_port))
        self._r_db      = self._labeled_entry(self._remote_frame, "Database name:", db.remote_db)
        self._r_user    = self._labeled_entry(self._remote_frame, "User:", db.remote_user)
        self._r_pass    = self._labeled_entry(self._remote_frame, "Password:", db.remote_password, show="*")
        self._r_vec_url = self._labeled_entry(self._remote_frame, "Qdrant URL (e.g. http://host:6333):",
                                               db.remote_vector_url)

    def _build_cloud_fields(self):
        self._cloud_frame = tk.Frame(self._fields_frame, bg=PANEL)
        db = self.cfg.db
        self._c_url       = self._labeled_entry(self._cloud_frame, "Supabase Postgres URL:", db.cloud_url)
        self._c_api       = self._labeled_entry(self._cloud_frame, "Supabase API key:", db.cloud_api_key, show="*")
        self._c_vec_url   = self._labeled_entry(self._cloud_frame, "Pinecone index host:", db.cloud_vector_url)
        self._c_vec_key   = self._labeled_entry(self._cloud_frame, "Pinecone API key:", db.cloud_vector_api_key, show="*")

    def _on_mode_change(self):
        for f in (self._local_frame, self._remote_frame, self._cloud_frame):
            f.pack_forget()
        mode = self._mode_var.get()
        if mode == "local":
            self._local_frame.pack(fill=tk.BOTH, expand=True)
        elif mode == "remote":
            self._remote_frame.pack(fill=tk.BOTH, expand=True)
        elif mode == "cloud":
            self._cloud_frame.pack(fill=tk.BOTH, expand=True)

    # ── Helpers ─────────────────────────────────────────────

    def _section(self, title: str):
        f = tk.Frame(self.window, bg=BG)
        f.pack(fill=tk.X, padx=16, pady=(10, 2))
        tk.Label(f, text=title, bg=BG, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(side=tk.LEFT)
        tk.Frame(f, bg=BORDER, height=1).pack(side=tk.LEFT, fill=tk.X,
                                               expand=True, padx=(8, 0))

    def _labeled_entry(self, parent, label: str, value: str, show: str = "") -> tk.StringVar:
        var = tk.StringVar(value=value)
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill=tk.X, padx=12, pady=3)
        tk.Label(row, text=label, bg=PANEL, fg=MUTED,
                 font=("Courier", 9), width=36, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var, bg=ENTRY_BG, fg=TEXT,
                 insertbackground=TEXT, relief=tk.FLAT,
                 font=("Courier", 10), show=show).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return var

    def _field_row(self, parent, label: str, widget):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill=tk.X, padx=12, pady=3)
        tk.Label(row, text=label, bg=PANEL, fg=MUTED,
                 font=("Courier", 9), width=38, anchor="w").pack(side=tk.LEFT)
        widget.pack(side=tk.LEFT)

    # ── Save ─────────────────────────────────────────────────

    def _save(self):
        mode = DatabaseMode(self._mode_var.get())
        db = self.cfg.db
        db.mode = mode

        if mode == DatabaseMode.LOCAL:
            db.local_sqlite_path = self._local_sqlite.get()
            db.local_vector_path = self._local_vec.get()
        elif mode == DatabaseMode.REMOTE:
            db.remote_host = self._r_host.get()
            db.remote_port = int(self._r_port.get() or 5432)
            db.remote_db   = self._r_db.get()
            db.remote_user = self._r_user.get()
            db.remote_password = self._r_pass.get()
            db.remote_vector_url = self._r_vec_url.get()
        elif mode == DatabaseMode.CLOUD:
            db.cloud_url = self._c_url.get()
            db.cloud_api_key = self._c_api.get()
            db.cloud_vector_url = self._c_vec_url.get()
            db.cloud_vector_api_key = self._c_vec_key.get()

        try:
            self.cfg.detection_confidence = float(self._conf_var.get())
            self.cfg.frame_skip = int(self._skip_var.get())
            self.cfg.camera_index = int(self._cam_var.get())
        except ValueError:
            pass

        self.cfg.alert_on_monitored = self._alert_var.get()
        self.cfg.save_snapshots     = self._snap_var.get()

        self.saved = True
        self.window.destroy()
