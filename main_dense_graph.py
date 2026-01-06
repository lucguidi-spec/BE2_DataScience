import os
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.data_loading import load_corpus, load_queries, load_qrels
from src.dense_encoder import build_or_load_embeddings
from src.citation_graph import build_citation_graph, build_graph_enhanced_embeddings


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMB_FILE = os.path.join(DATA_DIR, "corpus_embeddings_all_MiniLM_L6_v2.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def encode_query(text: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def rank_candidates_for_query_dense(
    query_id: str,
    qrels: Dict[str, Dict[str, int]],
    queries: Dict[str, Dict],
    corpus: Dict[str, Dict],
    doc_index: Dict[str, int],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int | None = None,
) -> List[Tuple[str, int, float, str]]:


    q = queries.get(query_id)
    if q is None:
        return []

    text = q.get("text") or q.get("title") or ""
    if not text:
        return []

    q_emb = encode_query(text, model)

    scored: List[Tuple[str, int, float, str]] = []
    for cid, label in qrels[query_id].items():
        idx = doc_index.get(cid)
        if idx is None:
            continue
        d_emb = embeddings[idx]
        score = float(d_emb @ q_emb)
        title = corpus.get(cid, {}).get("title") or ""
        scored.append((cid, label, score, title))

    scored.sort(key=lambda x: x[2], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def evaluate_dense_on_qrels(
    qrels: Dict[str, Dict[str, int]],
    queries: Dict[str, Dict],
    corpus: Dict[str, Dict],
    doc_index: Dict[str, int],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    example_query_id: str | None = None,
) -> None:


    if example_query_id is None:
        example_query_id = next(iter(qrels.keys()))

    print("\nTest du moteur dense sur une requête exemple")
    ranked_example = rank_candidates_for_query_dense(
        example_query_id,
        qrels,
        queries,
        corpus,
        doc_index,
        embeddings,
        model,
        top_k=5,
    )
    print("Top cinq candidats")
    for cid, label, score, title in ranked_example:
        print(f"- label {label}  score {score:.3f}  {cid}  {title[:120]}")

    # scoring globale
    all_true: List[int] = []
    all_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for qid, cand_dict in qrels.items():
        ranked = rank_candidates_for_query_dense(
            qid,
            qrels,
            queries,
            corpus,
            doc_index,
            embeddings,
            model,
            top_k=None,
        )
        if not ranked:
            continue

        y_true = [label for _, label, _, _ in ranked]
        y_scores = [score for _, _, score, _ in ranked]

        # On ignore les requêtes avec une seule classe (aucun positif ou aucun négatif)
        if len(set(y_true)) < 2:
            continue

        # Seuil pour dériver des prédictions binaires
        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        all_true.extend(y_true)
        all_scores.extend(y_scores)

    if not all_true:
        print("Aucune requête exploitable eval")
        return

    mean_prec = float(np.mean(precisions))
    mean_rec = float(np.mean(recalls))
    mean_f1 = float(np.mean(f1s))
    auc = roc_auc_score(all_true, all_scores)

    print("\nÉvaluation du moteur dense")
    print(f"Précision moyenne  {mean_prec:.3f}")
    print(f"Rappel moyen      {mean_rec:.3f}")
    print(f"F-mesure moyenne  {mean_f1:.3f}")
    print(f"AUC globale       {auc:.3f}")


def main():
    # chemins des fichiers
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    queries_path = os.path.join(DATA_DIR, "queries.jsonl")
    qrels_path = os.path.join(DATA_DIR, "valid.tsv")

    print("Chargement du corpus")
    corpus = load_corpus(corpus_path)
    print("Nb de documents dans le corpus ", len(corpus))

    print("Chargement des requêtes")
    queries = load_queries(queries_path)
    print("Nb de requêtes ", len(queries))

    print("Chargement des jugements de pertinence")
    qrels_valid = load_qrels(qrels_path)
    print("Nb de requêtes couvertes par valid.tsv ", len(qrels_valid))

    # Embeddings denses de base (même logique que main_dense.py)
    print("\nConstruction/chargement des embeddings denses du corpus")
    doc_ids, embeddings = build_or_load_embeddings(
        corpus,
        EMB_FILE,
        model_name=MODEL_NAME,
        batch_size=64,
    )
    print("Forme des embeddings de base ", embeddings.shape)

    
    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    print("Chargement du modèle de phrases ", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # Évaluation du moteur dense basique
    any_query_id = next(iter(qrels_valid.keys()))
    print("\n===== MOTEUR DENSE SANS STRUCTURE =====")
    evaluate_dense_on_qrels(
        qrels_valid,
        queries,
        corpus,
        doc_index,
        embeddings,
        model,
        example_query_id=any_query_id,
    )

    print("\nConstruction du graphe de citations (corpus + requêtes)")
    G = build_citation_graph(corpus, queries, include_query_edges=True)

    print("\nConstruction des embeddings enrichis par les voisins du graphe")
    enhanced_embeddings = build_graph_enhanced_embeddings(
        G,
        doc_ids,
        embeddings,
        alpha=0.8,      
        normalize=True, # on renormalise pour garder un cosinus cohérent
    )
    print("Forme des embeddings enrichis ", enhanced_embeddings.shape)

    print("\n===== MOTEUR DENSE AVEC STRUCTURE (VOISINS DU GRAPHE) =====")
    evaluate_dense_on_qrels(
        qrels_valid,
        queries,
        corpus,
        doc_index,
        enhanced_embeddings,
        model,
        example_query_id=any_query_id,
    )


if __name__ == "__main__":
    main()
