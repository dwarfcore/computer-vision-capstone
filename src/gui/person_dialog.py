"""
gui/person_dialog.py - Add or edit a person in the database.

When adding a new person, the user can:
  1. Type a name and notes
  2. Capture face from live webcam feed (if running) OR
  3. Load a photo from disk

The dialog encodes the face image and stores:
  - Embedding in the vector DB
  - Metadata (name, notes, thumbnail) in SQLite
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import db_sqlite as sql
from face_processing import detect_and_encode, crop_face

BG     = "#0f1117"
PANEL  = "#1a1d27"
ACCENT = "#00e5a0"
DANGER = "#ff4757"
TEXT   = "#e8eaf0"
MUTED  = "#6b7280"
BORDER = "#2a2d3a"
CARD   = "#21253a"
ENTRY_BG = "#12151f"

THUMB_SIZE = (160, 160)


class PersonDialog:
    def __init__(self, parent, sql_conn, vector_db, cap, cfg, person: dict = None):
        self.sql_conn  = sql_conn
        self.vector_db = vector_db
        self.cap       = cap          # May be None if camera not started
        self.cfg       = cfg
        self.person    = person       # None = add new, dict = edit existing
        self._embedding = None
        self._thumb_path = person["thumbnail_path"] if person else ""

        self.window = tk.Toplevel(parent)
        self.window.title("Add Person" if person is None else "Edit Person")
        self.window.configure(bg=BG)
        self.window.geometry("480x560")
        self.window.resizable(False, False)
        self.window.grab_set()

        self._build()
        if person:
            self._load_existing()

    def _build(self):
        tk.Label(self.window,
                 text="ADD PERSON" if self.person is None else "EDIT PERSON",
                 bg=BG, fg=ACCENT, font=("Courier", 13, "bold")).pack(
                     anchor="w", padx=20, pady=(16, 12))

        # ── Thumbnail preview ────────────────────────────────
        self._thumb_canvas = tk.Canvas(self.window, width=160, height=160,
                                        bg=CARD, highlightthickness=1,
                                        highlightbackground=BORDER)
        self._thumb_canvas.pack(pady=(0, 8))
        self._thumb_canvas.create_text(80, 80, text="No face\ncaptured",
                                        fill=MUTED, font=("Courier", 10),
                                        justify=tk.CENTER, tags="placeholder")

        # ── Capture buttons ──────────────────────────────────
        btn_row = tk.Frame(self.window, bg=BG)
        btn_row.pack()
        tk.Button(btn_row, text="📷 Capture from Camera", bg=ACCENT, fg=BG,
                  font=("Courier", 10, "bold"), relief=tk.FLAT, cursor="hand2",
                  command=self._capture_from_camera).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="📁 Load from File", bg=BORDER, fg=TEXT,
                  font=("Courier", 10), relief=tk.FLAT, cursor="hand2",
                  command=self._load_from_file).pack(side=tk.LEFT, padx=4)

        # ── Form fields ──────────────────────────────────────
        form = tk.Frame(self.window, bg=BG)
        form.pack(fill=tk.X, padx=20, pady=12)

        tk.Label(form, text="Name:", bg=BG, fg=MUTED,
                 font=("Courier", 10)).pack(anchor="w")
        self._name_var = tk.StringVar()
        tk.Entry(form, textvariable=self._name_var, bg=ENTRY_BG, fg=TEXT,
                 insertbackground=TEXT, relief=tk.FLAT,
                 font=("Courier", 11), width=36).pack(fill=tk.X, pady=(2, 10))

        tk.Label(form, text="Notes:", bg=BG, fg=MUTED,
                 font=("Courier", 10)).pack(anchor="w")
        self._notes_text = tk.Text(form, bg=ENTRY_BG, fg=TEXT,
                                    insertbackground=TEXT, relief=tk.FLAT,
                                    font=("Courier", 10), height=4, width=36)
        self._notes_text.pack(fill=tk.X, pady=(2, 10))

        self._monitor_var = tk.BooleanVar(value=False)
        tk.Checkbutton(form, text="Flag as monitored", variable=self._monitor_var,
                       bg=BG, fg=DANGER, selectcolor=BG,
                       activebackground=BG, font=("Courier", 10),
                       activeforeground=DANGER).pack(anchor="w")

        # ── Status ──────────────────────────────────────────
        self._status_var = tk.StringVar(value="")
        tk.Label(self.window, textvariable=self._status_var, bg=BG, fg=MUTED,
                 font=("Courier", 9)).pack()

        # ── Action buttons ───────────────────────────────────
        btn2 = tk.Frame(self.window, bg=BG)
        btn2.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=12)
        tk.Button(btn2, text="Save", bg=ACCENT, fg=BG,
                  font=("Courier", 11, "bold"), relief=tk.FLAT, cursor="hand2",
                  padx=20, command=self._save).pack(side=tk.LEFT)
        tk.Button(btn2, text="Cancel", bg=BORDER, fg=TEXT,
                  font=("Courier", 10), relief=tk.FLAT, cursor="hand2",
                  padx=16, command=self.window.destroy).pack(side=tk.LEFT, padx=8)

    def _load_existing(self):
        p = self.person
        self._name_var.set(p["name"])
        self._notes_text.insert("1.0", p.get("notes", ""))
        self._monitor_var.set(bool(p["is_monitored"]))
        if p.get("thumbnail_path") and os.path.exists(p["thumbnail_path"]):
            self._display_thumbnail_from_path(p["thumbnail_path"])
            self._status_var.set("Existing face loaded (capture new to update)")

    # ── Capture ──────────────────────────────────────────────

    def _capture_from_camera(self):
        if self.cap is None or not self.cap.isOpened():
            messagebox.showinfo("Camera", "Start the camera from the main window first.")
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera", "Could not read frame.")
            return
        self._process_image_frame(frame)

    def _load_from_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Load Error", "Could not open image file.")
            return
        self._process_image_frame(frame)

    def _process_image_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations, encodings = detect_and_encode(rgb)
        if not encodings:
            messagebox.showwarning("No Face", "No face detected in image. Try again.")
            return
        if len(encodings) > 1:
            messagebox.showinfo("Multiple Faces", "Multiple faces found — using the first one.")

        self._embedding = encodings[0]
        loc = locations[0]
        cropped_bgr = crop_face(frame_bgr, loc)

        # Save thumbnail
        os.makedirs("data/snapshots", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        thumb_path = f"data/snapshots/thumb_{ts}.jpg"
        cv2.imwrite(thumb_path, cv2.resize(cropped_bgr, (160, 160)))
        self._thumb_path = thumb_path

        self._display_thumbnail_from_path(thumb_path)
        self._status_var.set(f"✓ Face captured ({len(encodings)} face(s) found)")

    def _display_thumbnail_from_path(self, path: str):
        try:
            img = Image.open(path).resize(THUMB_SIZE, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._thumb_canvas.delete("all")
            self._thumb_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self._thumb_canvas._photo = photo
        except Exception as e:
            self._status_var.set(f"Thumbnail error: {e}")

    # ── Save ─────────────────────────────────────────────────

    def _save(self):
        name = self._name_var.get().strip() or "Unknown"
        notes = self._notes_text.get("1.0", tk.END).strip()
        is_monitored = 1 if self._monitor_var.get() else 0

        if self.person is None:
            # New person
            if self._embedding is None:
                messagebox.showwarning("No Face", "Capture or load a face image first.")
                return
            person_id = sql.add_person(self.sql_conn, name=name, notes=notes,
                                        thumbnail_path=self._thumb_path)
            self.vector_db.add(person_id, self._embedding)
            if is_monitored:
                sql.update_person(self.sql_conn, person_id, is_monitored=1)
            self._status_var.set(f"✓ {name} added (ID {person_id})")
        else:
            person_id = self.person["id"]
            sql.update_person(self.sql_conn, person_id, name=name, notes=notes,
                               is_monitored=is_monitored,
                               thumbnail_path=self._thumb_path)
            if self._embedding is not None:
                self.vector_db.add(person_id, self._embedding)  # upsert
            self._status_var.set(f"✓ {name} updated")

        self.window.after(800, self.window.destroy)
