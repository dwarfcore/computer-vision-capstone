"""
gui/app.py - Main application window.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  Header bar: title + DB mode badge + status              │
  ├────────────────────────────┬─────────────────────────────┤
  │  Camera feed (live)        │  Right panel:               │
  │  [annotated OpenCV→PIL]    │    Persons list             │
  │                            │    Appearances log          │
  ├────────────────────────────┴─────────────────────────────┤
  │  Action bar: Add Person | Settings | Toggle Recording    │
  └──────────────────────────────────────────────────────────┘
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg_module
import db_sqlite as sql
import db_vector as vec
from face_processing import process_frame, draw_detections, crop_face, DetectedFace
from gui.settings_dialog import SettingsDialog
from gui.person_dialog import PersonDialog


# ─── Palette ───────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT  = "#00e5a0"
DANGER  = "#ff4757"
TEXT    = "#e8eaf0"
MUTED   = "#6b7280"
BORDER  = "#2a2d3a"
CARD    = "#21253a"


class MonitoringApp:
    def __init__(self):
        self.cfg = cfg_module.load_config()
        self._init_db()

        self.root = tk.Tk()
        self.root.title("Vision Monitor")
        self.root.configure(bg=BG)
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._alert_queue: queue.Queue = queue.Queue()
        self._cap = None
        self._recording = False
        self._camera_thread = None
        self._stop_event = threading.Event()

        self._build_ui()
        self._refresh_persons()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(33, self._update_frame)   # ~30 fps UI update
        self.root.after(500, self._update_log)

    # ─── DB Init ────────────────────────────────────────────

    def _init_db(self):
        os.makedirs("data/snapshots", exist_ok=True)
        self.sql_conn = sql.init_db(self.cfg)
        self.vector_db = vec.create_vector_db(self.cfg)

    def _reinit_db(self):
        """Called after settings change."""
        try:
            self.sql_conn.close()
        except Exception:
            pass
        try:
            self.vector_db.close()
        except Exception:
            pass
        self._init_db()
        self._refresh_persons()
        self._refresh_log()

    # ─── UI Build ───────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        self._build_camera_panel(content)
        self._build_right_panel(content)
        self._build_action_bar()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=PANEL, height=54)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="◉  VISION MONITOR", bg=PANEL, fg=ACCENT,
                 font=("Courier", 15, "bold")).pack(side=tk.LEFT, padx=16)

        self._db_badge = tk.Label(hdr, text="● LOCAL", bg=PANEL, fg=ACCENT,
                                   font=("Courier", 10, "bold"))
        self._db_badge.pack(side=tk.LEFT, padx=8)

        self._status_label = tk.Label(hdr, text="Ready", bg=PANEL, fg=MUTED,
                                       font=("Courier", 10))
        self._status_label.pack(side=tk.LEFT, padx=16)

        self._rec_label = tk.Label(hdr, text="", bg=PANEL, fg=DANGER,
                                    font=("Courier", 10, "bold"))
        self._rec_label.pack(side=tk.RIGHT, padx=16)

        self._update_db_badge()

    def _build_camera_panel(self, parent):
        cam_frame = tk.Frame(parent, bg=PANEL, bd=0)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(cam_frame, text="LIVE FEED", bg=PANEL, fg=MUTED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 0))

        self._canvas = tk.Canvas(cam_frame, bg="#000000", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._canvas_image_id = None

        # Overlay text when no camera
        self._no_cam_text = self._canvas.create_text(
            320, 240, text="Camera off\nPress  ▶ Start Camera",
            fill=MUTED, font=("Courier", 13), justify=tk.CENTER
        )

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=1)

        # Persons section
        p_frame = tk.Frame(right, bg=PANEL)
        p_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        p_frame.columnconfigure(0, weight=1)
        p_frame.rowconfigure(1, weight=1)

        phdr = tk.Frame(p_frame, bg=PANEL)
        phdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        tk.Label(phdr, text="PERSONS", bg=PANEL, fg=MUTED,
                 font=("Courier", 9, "bold")).pack(side=tk.LEFT)
        tk.Button(phdr, text="+ Add", bg=ACCENT, fg=BG, relief=tk.FLAT,
                  font=("Courier", 9, "bold"), cursor="hand2",
                  command=self._open_add_person).pack(side=tk.RIGHT)

        cols = ("Name", "Seen", "Monitor")
        self._persons_tree = ttk.Treeview(p_frame, columns=cols, show="headings",
                                           selectmode="browse", height=10)
        for col in cols:
            self._persons_tree.heading(col, text=col)
        self._persons_tree.column("Name", width=120)
        self._persons_tree.column("Seen", width=50, anchor="center")
        self._persons_tree.column("Monitor", width=60, anchor="center")
        self._persons_tree.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._persons_tree.bind("<Double-Button-1>", self._on_person_double_click)

        _style_treeview()

        scrollbar = ttk.Scrollbar(p_frame, orient=tk.VERTICAL,
                                   command=self._persons_tree.yview)
        self._persons_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 8))

        # Appearances log
        log_frame = tk.Frame(right, bg=PANEL)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        tk.Label(log_frame, text="RECENT APPEARANCES", bg=PANEL, fg=MUTED,
                 font=("Courier", 9, "bold")).grid(row=0, column=0, sticky="w",
                                                    padx=10, pady=(8, 4))

        self._log_text = tk.Text(log_frame, bg=CARD, fg=TEXT, relief=tk.FLAT,
                                  font=("Courier", 9), state=tk.DISABLED,
                                  wrap=tk.WORD, height=8)
        self._log_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._log_text.tag_configure("alert", foreground=DANGER)
        self._log_text.tag_configure("normal", foreground=TEXT)
        self._log_text.tag_configure("muted", foreground=MUTED)

    def _build_action_bar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=52)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        self._cam_btn = tk.Button(
            bar, text="▶  Start Camera", bg=ACCENT, fg=BG,
            font=("Courier", 11, "bold"), relief=tk.FLAT, cursor="hand2",
            padx=14, command=self._toggle_camera
        )
        self._cam_btn.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(bar, text="⚙  Settings", bg=BORDER, fg=TEXT,
                  font=("Courier", 10), relief=tk.FLAT, cursor="hand2",
                  padx=12, command=self._open_settings).pack(side=tk.LEFT, padx=4, pady=10)

        tk.Button(bar, text="↻  Refresh", bg=BORDER, fg=TEXT,
                  font=("Courier", 10), relief=tk.FLAT, cursor="hand2",
                  padx=12, command=self._refresh_persons).pack(side=tk.LEFT, padx=4, pady=10)

        tk.Button(bar, text="🗑  Delete Person", bg=DANGER, fg=TEXT,
                  font=("Courier", 10), relief=tk.FLAT, cursor="hand2",
                  padx=12, command=self._delete_selected_person).pack(side=tk.RIGHT, padx=10, pady=10)

        tk.Button(bar, text="★  Toggle Monitor", bg="#2d3748", fg=TEXT,
                  font=("Courier", 10), relief=tk.FLAT, cursor="hand2",
                  padx=12, command=self._toggle_monitor_selected).pack(side=tk.RIGHT, padx=4, pady=10)

    # ─── Camera Thread ──────────────────────────────────────

    def _toggle_camera(self):
        if self._recording:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self._stop_event.clear()
        self._recording = True
        self._cam_btn.configure(text="■  Stop Camera", bg=DANGER)
        self._rec_label.configure(text="⏺ REC")
        self._set_status("Camera running…")

        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

    def _stop_camera(self):
        self._stop_event.set()
        self._recording = False
        self._cam_btn.configure(text="▶  Start Camera", bg=ACCENT)
        self._rec_label.configure(text="")
        self._set_status("Camera stopped")
        if self._cap:
            self._cap.release()
            self._cap = None
        self._canvas.delete("all")
        self._no_cam_text = self._canvas.create_text(
            self._canvas.winfo_width() // 2 or 320,
            self._canvas.winfo_height() // 2 or 240,
            text="Camera off\nPress  ▶ Start Camera",
            fill=MUTED, font=("Courier", 13), justify=tk.CENTER
        )

    def _camera_loop(self):
        self._cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self._cap.isOpened():
            self._alert_queue.put(("error", "Cannot open camera"))
            return

        frame_count = 0
        last_faces: list = []

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1
            annotated = frame.copy()

            if frame_count % self.cfg.frame_skip == 0:
                try:
                    last_faces = process_frame(
                        frame,
                        self.vector_db,
                        self.sql_conn,
                        threshold=self.cfg.detection_confidence
                    )
                    self._handle_detections(last_faces, frame)
                except Exception as e:
                    pass  # Don't crash loop on processing error

            annotated = draw_detections(annotated, last_faces)

            # Convert BGR → RGB → PIL → put in queue
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            if not self._frame_queue.full():
                self._frame_queue.put(img)

        if self._cap:
            self._cap.release()

    def _handle_detections(self, faces: list, frame: np.ndarray):
        """Log appearances and save snapshots for detected faces."""
        for face in faces:
            if face.person_id is None:
                continue
            # Save snapshot
            snap_path = ""
            if self.cfg.save_snapshots:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(
                    self.cfg.snapshot_dir, f"person_{face.person_id}_{ts}.jpg"
                )
                os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
                cropped = crop_face(frame, face.bbox)
                cv2.imwrite(snap_path, cropped)

            sql.log_appearance(self.sql_conn, face.person_id, snap_path, face.confidence)

            if face.is_monitored and self.cfg.alert_on_monitored:
                self._alert_queue.put(("monitored", face.name))

    # ─── UI Updates ─────────────────────────────────────────

    def _update_frame(self):
        """Pull latest frame from queue and display on canvas."""
        try:
            img = self._frame_queue.get_nowait()
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw > 1 and ch > 1:
                img = img.resize((cw, ch), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self._canvas._photo = photo  # prevent GC
        except queue.Empty:
            pass

        # Check alert queue
        try:
            kind, name = self._alert_queue.get_nowait()
            if kind == "monitored":
                self._log_append(f"⚠ ALERT: {name} detected!", tag="alert")
                self.root.bell()
            elif kind == "error":
                messagebox.showerror("Camera Error", name)
        except queue.Empty:
            pass

        self.root.after(33, self._update_frame)

    def _update_log(self):
        self._refresh_log()
        self.root.after(3000, self._update_log)

    def _refresh_persons(self):
        for row in self._persons_tree.get_children():
            self._persons_tree.delete(row)
        persons = sql.get_all_persons(self.sql_conn)
        for p in persons:
            count = sql.get_appearance_count(self.sql_conn, p["id"])
            mon = "⚑" if p["is_monitored"] else ""
            tag = "monitored" if p["is_monitored"] else ""
            self._persons_tree.insert("", tk.END,
                iid=str(p["id"]),
                values=(p["name"], count, mon),
                tags=(tag,)
            )
        self._persons_tree.tag_configure("monitored", foreground=DANGER)

    def _refresh_log(self):
        rows = sql.get_recent_appearances(self.sql_conn, limit=30)
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        for r in rows:
            ts = r["timestamp"][:16]
            tag = "alert" if r.get("is_monitored") else "normal"
            self._log_text.insert(tk.END, f"{ts}  ", "muted")
            self._log_text.insert(tk.END, f"{r['name']}", tag)
            conf = r.get("confidence", 0)
            self._log_text.insert(tk.END, f"  ({conf:.0%})\n", "muted")
        self._log_text.configure(state=tk.DISABLED)

    def _log_append(self, msg: str, tag: str = "normal"):
        self._log_text.configure(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.insert("1.0", f"{ts}  {msg}\n", tag)
        self._log_text.configure(state=tk.DISABLED)

    def _set_status(self, msg: str):
        self._status_label.configure(text=msg)

    def _update_db_badge(self):
        mode = self.cfg.db.mode.value.upper()
        color = {
            "LOCAL": ACCENT, "REMOTE": "#4fc3f7", "CLOUD": "#ce93d8"
        }.get(mode, MUTED)
        self._db_badge.configure(text=f"● {mode}", fg=color)

    # ─── Actions ────────────────────────────────────────────

    def _open_add_person(self):
        dlg = PersonDialog(self.root, self.sql_conn, self.vector_db,
                           self._cap, self.cfg)
        self.root.wait_window(dlg.window)
        self._refresh_persons()

    def _on_person_double_click(self, event):
        sel = self._persons_tree.selection()
        if not sel:
            return
        person_id = int(sel[0])
        person = sql.get_person(self.sql_conn, person_id)
        if person:
            dlg = PersonDialog(self.root, self.sql_conn, self.vector_db,
                               self._cap, self.cfg, person=person)
            self.root.wait_window(dlg.window)
            self._refresh_persons()

    def _delete_selected_person(self):
        sel = self._persons_tree.selection()
        if not sel:
            messagebox.showinfo("Select Person", "Select a person from the list first.")
            return
        person_id = int(sel[0])
        person = sql.get_person(self.sql_conn, person_id)
        if not person:
            return
        if messagebox.askyesno("Delete", f"Delete '{person['name']}' and all their appearances?"):
            sql.delete_person(self.sql_conn, person_id)
            self.vector_db.delete(person_id)
            self._refresh_persons()
            self._set_status(f"Deleted {person['name']}")

    def _toggle_monitor_selected(self):
        sel = self._persons_tree.selection()
        if not sel:
            return
        person_id = int(sel[0])
        person = sql.get_person(self.sql_conn, person_id)
        if not person:
            return
        new_val = 0 if person["is_monitored"] else 1
        sql.update_person(self.sql_conn, person_id, is_monitored=new_val)
        self._refresh_persons()

    def _open_settings(self):
        was_recording = self._recording
        if was_recording:
            self._stop_camera()
        dlg = SettingsDialog(self.root, self.cfg)
        self.root.wait_window(dlg.window)
        if dlg.saved:
            self.cfg = dlg.cfg
            cfg_module.save_config(self.cfg)
            self._reinit_db()
            self._update_db_badge()
            self._set_status("Settings saved — DB reloaded")

    # ─── Lifecycle ──────────────────────────────────────────

    def _on_close(self):
        self._stop_event.set()
        if self._cap:
            self._cap.release()
        try:
            self.sql_conn.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─── Treeview style ────────────────────────────────────────

def _style_treeview():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background=CARD, foreground=TEXT,
                    fieldbackground=CARD, rowheight=26,
                    font=("Courier", 10))
    style.configure("Treeview.Heading", background=PANEL, foreground=MUTED,
                    font=("Courier", 9, "bold"), relief="flat")
    style.map("Treeview", background=[("selected", "#2a4a6b")])
