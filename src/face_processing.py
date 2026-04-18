import os
import cv2
import numpy as np
from datetime import datetime

from config import FACE_FOLDER, MATCH_THRESHOLD
from db_mode import get_db_mode
from db_sqlite import add_person, get_person, log_appearance
from db_vector import add_embedding, find_match


def ensure_folders():
    os.makedirs(FACE_FOLDER, exist_ok=True)


def save_face_image(face_img):
    filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(FACE_FOLDER, filename)
    cv2.imwrite(path, face_img)
    return path


def generate_demo_embedding(face_img):
    """
    Simple demo embedding for class use.
    This is NOT a real face recognition model.
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 16))
    vector = resized.astype(np.float32).flatten()

    norm = np.linalg.norm(vector) + 1e-10
    vector = vector / norm
    return vector


def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    return faces


def process_face(face_img):
    db_mode = get_db_mode()

    if db_mode == "remote":
        print("Remote database mode selected. Placeholder for future implementation.")
        return

    if db_mode == "cloud":
        print("Cloud database mode selected. Placeholder for future implementation.")
        return

    embedding = generate_demo_embedding(face_img)
    matched_person_id, score = find_match(embedding)

    if matched_person_id is not None and score >= MATCH_THRESHOLD:
        person = get_person(matched_person_id)

        if person is None:
            print("Match found in vector database but not in SQLite database.")
            return

        if person["is_monitored"] == 1:
            event_type = "recognized_monitored"
        else:
            event_type = "recognized"

        log_appearance(matched_person_id, event_type)
        add_embedding(matched_person_id, embedding)

        print("\nKnown person detected")
        print("ID:", matched_person_id)
        print("Name:", person["name"])
        print("Similarity:", round(score, 4))
        print("Database mode:", db_mode)

        if person["is_monitored"] == 1:
            print("ALERT: This monitored person was detected again.")

    else:
        new_name = "Unknown"
        person_id = add_person(new_name, 0, "Auto-added by system")
        add_embedding(person_id, embedding)
        log_appearance(person_id, "new_person")
        save_face_image(face_img)

        print("\nNew person added")
        print("ID:", person_id)
        print("Name:", new_name)
        print("Database mode:", db_mode)


def run_webcam():
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("\nWebcam controls:")
    print("s = process first detected face")
    print("q = quit webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame.")
            break

        faces = detect_faces(frame, detector)
        display = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            display,
            "Press s to process face, q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Video Monitoring System", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if len(faces) == 0:
                print("No face detected.")
                continue

            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            process_face(face_crop)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()