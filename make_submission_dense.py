import os
import csv

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loading import load_corpus, load_queries
from src.dense_encoder import build_or_load_embeddings


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# fichier Kaggle
INPUT_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

# fichier crée
OUTPUT_FILE = os.path.join(DATA_DIR, "submission_dense.csv")

EMB_PATH = os.path.join(DATA_DIR, "corpus_embeddings_all_MiniLM_L6_v2.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_doc_index(doc_ids):
    return {doc_id: i for i, doc_id in enumerate(doc_ids)}


def main():
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    queries_path = os.path.join(DATA_DIR, "queries.jsonl")

    if not os.path.exists(corpus_path):
        print(f"Fichier manquant  {corpus_path}")
        return
    if not os.path.exists(queries_path):
        print(f"Fichier manquant  {queries_path}")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"Fichier d entrée manquant  {INPUT_FILE}")
        print("sample_submission.csv est bien dans le dossier data ?")
        return

    print("Chargement du corpus et des requêtes")
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    print(f"nb de documents  {len(corpus)}")
    print(f"nb de requêtes   {len(queries)}")

    print(f"\nLecture du fichier Kaggle  {INPUT_FILE}")
    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected_fields = ["RowId", "query-id", "corpus-id", "score"]
        for field in expected_fields:
            if field not in reader.fieldnames:
                raise ValueError(
                    f"Colonne manquante dans sample_submission.csv  {field}"
                )
        for r in reader:
            row_id = r["RowId"]
            qid = r["query-id"]
            cid = r["corpus-id"]
            rows.append((row_id, qid, cid))

    print(f"Nombre de lignes lues (sans l en tête)  {len(rows)}")

    print("\nChargement/construction des embeddings du corpus")
    doc_ids, embeddings = build_or_load_embeddings(
        corpus,
        file_path=EMB_PATH,
        model_name=MODEL_NAME,
        batch_size=64,
    )
    doc_index = build_doc_index(doc_ids)
    print(f"Forme des embeddings  {embeddings.shape}")

    print(f"\nChargement du modèle de phrases  {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    query_emb_cache = {}

    out_rows = []

    print("\nCalcul des scores pour toutes les paires")
    for row_id, qid, cid in rows:
        # embedding de la requête
        if qid not in query_emb_cache:
            q = queries.get(qid)
            if q is None:
                query_emb_cache[qid] = None
            else:
                text = q.get("text") or q.get("title") or ""
                if not text:
                    query_emb_cache[qid] = None
                else:
                    emb = model.encode(
                        text,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    query_emb_cache[qid] = emb

        q_emb = query_emb_cache[qid]

        # embedding du document
        idx = doc_index.get(cid)
        if q_emb is None or idx is None:
            score = 0.0
        else:
            d_emb = embeddings[idx]
            score = float(d_emb @ q_emb)

        out_rows.append(
            {
                "RowId": row_id,
                "query-id": qid,
                "corpus-id": cid,
                "score": f"{score:.6f}",
            }
        )

    print(f"\nÉcriture du fichier de soumission CSV  {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["RowId", "query-id", "corpus-id", "score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print("Fichier de soumission écrit.")


if __name__ == "__main__":
    main()