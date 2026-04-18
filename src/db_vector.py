import os
import numpy as np
from config import VECTOR_DB_PATH


def load_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        data = np.load(VECTOR_DB_PATH, allow_pickle=True)
        embeddings = data["embeddings"]
        person_ids = data["person_ids"].tolist()
    else:
        embeddings = np.empty((0, 256), dtype=np.float32)
        person_ids = []

    return embeddings, person_ids


def save_vector_db(embeddings, person_ids):
    np.savez(
        VECTOR_DB_PATH,
        embeddings=embeddings,
        person_ids=np.array(person_ids, dtype=object)
    )


def add_embedding(person_id, embedding):
    embeddings, person_ids = load_vector_db()

    embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    if embeddings.shape[0] == 0:
        embeddings = embedding
    else:
        embeddings = np.vstack([embeddings, embedding])

    person_ids.append(person_id)
    save_vector_db(embeddings, person_ids)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def find_match(embedding):
    embeddings, person_ids = load_vector_db()

    if len(person_ids) == 0:
        return None, None

    best_person_id = None
    best_score = -1.0

    for i in range(len(person_ids)):
        score = cosine_similarity(embedding, embeddings[i])
        if score > best_score:
            best_score = score
            best_person_id = person_ids[i]

    return best_person_id, best_score