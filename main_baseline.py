import os
from statistics import mean

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.data_loading import load_corpus, load_queries, load_qrels
from src.vectorizer_baseline import (
    build_bow_matrix,
    build_tfidf_matrix,
    print_top_terms_in_corpus,
    print_vector_features,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_doc_index(doc_ids):
    
    return {doc_id: i for i, doc_id in enumerate(doc_ids)}


def rank_candidates_for_query_bow(query_text, candidate_ids, vectorizer, X, doc_index):
    
    if not candidate_ids:
        return []

    q_vec = vectorizer.transform([query_text])

    rows = []
    row_cids = []
    for cid in candidate_ids:
        idx = doc_index.get(cid)
        if idx is not None:
            rows.append(idx)
            row_cids.append(cid)

    if not rows:
        return []

    X_sub = X[rows]
    sims = cosine_similarity(q_vec, X_sub).ravel()

    ranked = sorted(zip(row_cids, sims), key=lambda x: x[1], reverse=True)
    return ranked


def evaluate_bow_on_qrels(queries, qrels, X, vectorizer, doc_index):
    
    per_query_precisions = []
    per_query_recalls = []
    per_query_f1 = []

    all_scores = []
    all_labels = []

    for qid, cand_dict in qrels.items():
        query = queries.get(qid)
        if query is None:
            continue

        query_text = query.get("text") or query.get("title") or ""
        candidate_ids = list(cand_dict.keys())

        ranked = rank_candidates_for_query_bow(
            query_text, candidate_ids, vectorizer, X, doc_index
        )
        if not ranked:
            continue

        scores = []
        labels = []
        for cid, score in ranked:
            labels.append(cand_dict[cid])
            scores.append(score)

        y_true = np.array(labels)
        y_scores = np.array(scores)

        y_pred = (y_scores >= 0.5).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        per_query_precisions.append(p)
        per_query_recalls.append(r)
        per_query_f1.append(f1)

        if len(np.unique(y_true)) > 1:
            all_labels.extend(y_true.tolist())
            all_scores.extend(y_scores.tolist())

    mean_precision = float(np.mean(per_query_precisions)) if per_query_precisions else 0.0
    mean_recall = float(np.mean(per_query_recalls)) if per_query_recalls else 0.0
    mean_f1 = float(np.mean(per_query_f1)) if per_query_f1 else 0.0

    if all_labels and len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    else:
        auc = None

    return mean_precision, mean_recall, mean_f1, auc


def search_free_text(
    query_text,
    vectorizer,
    X,
    doc_ids,
    corpus,
    top_k=10,
):
 
    if not query_text:
        return []

    q_vec = vectorizer.transform([query_text])

    sims = cosine_similarity(q_vec, X).ravel()

    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        doc_id = doc_ids[idx]
        doc = corpus.get(doc_id, {})
        text = doc.get("title") or doc.get("text") or ""
        results.append((doc_id, float(sims[idx]), text))

    return results


def main():
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    queries_path = os.path.join(DATA_DIR, "queries.jsonl")
    qrels_path = os.path.join(DATA_DIR, "valid.tsv")

    if not os.path.exists(corpus_path):
        print(f"Fichier manquant  {corpus_path}")
        return
    if not os.path.exists(queries_path):
        print(f"Fichier manquant  {queries_path}")
        return
    if not os.path.exists(qrels_path):
        print(f"Fichier manquant  {qrels_path}")
        return

    print("Chargement du corpus")
    corpus = load_corpus(corpus_path)
    print(f"Nb de documents dans le corpus  {len(corpus)}")

    print("Chargement des requêtes")
    queries = load_queries(queries_path)
    print(f"Nb de requêtes  {len(queries)}")

    print("Chargement des qrels")
    qrels = load_qrels(qrels_path)
    print(f"Nb de requêtes couvertes par valid.tsv  {len(qrels)}")

    nb_pairs = sum(len(cands) for cands in qrels.values())
    print(f"Total de paires requête document  {nb_pairs}")

    proportions = []
    for qid, cands in qrels.items():
        labels = list(cands.values())
        if labels:
            proportions.append(sum(labels) / len(labels))
    if proportions:
        print(f"Proportion moyenne de candidats pertinents par requête  {mean(proportions):.3f}")

    print("\nExemple de requête avec candidats")
    example_qid = next(iter(qrels))
    example_query = queries.get(example_qid)

    if example_query is not None:
        query_text = example_query.get("text", "")
        print(f"Identifiant de la requête  {example_qid}")
        print(f"Texte de la requête  {query_text}")

        candidates = qrels[example_qid]
        positive_ids = [cid for cid, rel in candidates.items() if rel == 1]
        negative_ids = [cid for cid, rel in candidates.items() if rel == 0]

        print("\nCandidats positifs")
        for cid in positive_ids[:5]:
            doc = corpus.get(cid, {})
            doc_text = doc.get("title") or doc.get("text") or ""
            print(f"- {cid}  {doc_text}")

        print("\nCandidats négatifs")
        for cid in negative_ids[:5]:
            doc = corpus.get(cid, {})
            doc_text = doc.get("title") or doc.get("text") or ""
            print(f"- {cid}  {doc_text}")
    else:
        print("Identifiant introuvable dans la requête dans queries", example_qid)

    # choix de la variante creuse
    # mets "bow" pour CountVectorizer, "tfidf" pour TF IDF
    variant = "tfidf"

    if variant == "bow":
        print("\nConstruction de la matrice creuse variante sac de mots")
        X, vectorizer, doc_ids = build_bow_matrix(corpus)
    else:
        print("\nConstruction de la matrice creuse variante TF IDF")
        X, vectorizer, doc_ids = build_tfidf_matrix(corpus)

    doc_index = build_doc_index(doc_ids)

    print_top_terms_in_corpus(X, vectorizer, top_n=20)

    if doc_ids:
        example_doc_idx = 0
        example_doc_id = doc_ids[example_doc_idx]
        example_doc = corpus[example_doc_id]
        print(f"\nExemple de document  {example_doc_id}")
        print("Texte utilisé pour l encodage")
        text = example_doc.get("title") or example_doc.get("text") or ""
        print(text)
        print_vector_features(X[example_doc_idx], vectorizer, top_n=15)

    if example_query is not None:
        print("\nTest du moteur creux sur requête exemple")
        candidates_for_example = list(qrels[example_qid].keys())
        query_text_example = example_query.get("text") or example_query.get("title") or ""
        ranked_example = rank_candidates_for_query_bow(
            query_text_example, candidates_for_example, vectorizer, X, doc_index
        )

        print("Top cinq candidats")
        for cid, score in ranked_example[:5]:
            label = qrels[example_qid][cid]
            doc = corpus.get(cid, {})
            text = doc.get("title") or doc.get("text") or ""
            print(f"- label {label}  score {score:.3f}  {text}")

    print("\nÉvaluation du moteur creux sur valid.tsv")
    mean_p, mean_r, mean_f1, auc = evaluate_bow_on_qrels(
        queries, qrels, X, vectorizer, doc_index
    )
    print(f"Précision moyenne  {mean_p:.3f}")
    print(f"Rappel moyen      {mean_r:.3f}")
    print(f"F mesure moyenne  {mean_f1:.3f}")
    if auc is not None:
        print(f"AUC globale       {auc:.3f}")
    else:
        print("AUC non calculable")

    print("\nRecherche sur tout le corpus")
    query_text_free = "sentiment analysis financial microblogs"
    print(f"Requête  {query_text_free}")

    results = search_free_text(
        query_text=query_text_free,
        vectorizer=vectorizer,
        X=X,
        doc_ids=doc_ids,
        corpus=corpus,
        top_k=10,
    )

    for rank, (doc_id, score, text) in enumerate(results, start=1):
        short_text = text if len(text) < 120 else text[:117] + "..."
        print(f"{rank:2d}. score {score:.3f}  {doc_id}  {short_text}")


if __name__ == "__main__":
    main()
