import os
from statistics import mean

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sentence_transformers import SentenceTransformer

from src.data_loading import load_corpus, load_queries, load_qrels
from src.dense_encoder import build_or_load_embeddings, get_document_text


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMB_PATH = os.path.join(DATA_DIR, "corpus_embeddings_all_MiniLM_L6_v2.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_doc_index(doc_ids):
    return {doc_id: i for i, doc_id in enumerate(doc_ids)}


def rank_candidates_for_query_dense(
    query_text,
    candidate_ids,
    model,
    embeddings,
    doc_index,
):

    if not candidate_ids:
        return []

    if not query_text:
        return []

    q_emb = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    rows = []
    row_cids = []
    for cid in candidate_ids:
        idx = doc_index.get(cid)
        if idx is not None:
            rows.append(idx)
            row_cids.append(cid)

    if not rows:
        return []

    emb_sub = embeddings[rows]       
    scores = emb_sub @ q_emb         # produit scalaire car tous les vecteurs sont normalisés

    ranked = sorted(zip(row_cids, scores), key=lambda x: x[1], reverse=True)
    return ranked


def evaluate_dense_on_qrels(
    queries,
    qrels,
    embeddings,
    doc_ids,
    model,
):

    doc_index = build_doc_index(doc_ids)

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

        ranked = rank_candidates_for_query_dense(
            query_text,
            candidate_ids,
            model,
            embeddings,
            doc_index,
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

        # pour avoir une prédiction binaire
        y_pred = (y_scores >= 0.5).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
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


def search_free_text_dense(
    query_text,
    model,
    embeddings,
    doc_ids,
    corpus,
    top_k=10,
):

    if not query_text:
        return []

    q_emb = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = embeddings @ q_emb      # produit scalaire car vecteurs normalisés

    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        doc_id = doc_ids[idx]
        doc = corpus.get(doc_id, {})
        text = get_document_text(doc)
        results.append((doc_id, float(scores[idx]), text))

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

    print("Chargement des jugements de pertinence")
    qrels = load_qrels(qrels_path)
    print(f"Nb de requêtes couvertes par valid.tsv  {len(qrels)}")

    nb_pairs = sum(len(cands) for cands in qrels.values())
    print(f"Nb total de paires requête document  {nb_pairs}")

    proportions = []
    for qid, cands in qrels.items():
        labels = list(cands.values())
        if labels:
            proportions.append(sum(labels) / len(labels))
    if proportions:
        print(f"Proportion moyenne de candidats pertinents par requête  {mean(proportions):.3f}")

    print("\nExemple avec candidats")
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
            doc_text = get_document_text(doc)
            print(f"- {cid}  {doc_text}")

        print("\nCandidats négatifs")
        for cid in negative_ids[:5]:
            doc = corpus.get(cid, {})
            doc_text = get_document_text(doc)
            print(f"- {cid}  {doc_text}")
    else:
        print("Impossible de trouver la requête pour l id", example_qid)

    # embeddings du corpus chargés ou construits
    print("\nConstruction/chargement des embeddings du corpus")
    doc_ids, embeddings = build_or_load_embeddings(
        corpus,
        file_path=EMB_PATH,
        model_name=MODEL_NAME,
        batch_size=64,
    )
    print(f"Forme des embeddings  {embeddings.shape}")

    # modèle dense pour encoder les requêtes
    print(f"Chargement du modèle de phrases  {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # test sur la requête exemple
    if example_query is not None:
        print("\nTest du moteur dense")
        candidates_for_example = list(qrels[example_qid].keys())
        query_text_example = example_query.get("text") or example_query.get("title") or ""
        ranked_example = rank_candidates_for_query_dense(
            query_text_example,
            candidates_for_example,
            model,
            embeddings,
            build_doc_index(doc_ids),
        )

        print("Top cinq candidats")
        for cid, score in ranked_example[:5]:
            label = qrels[example_qid][cid]
            doc = corpus.get(cid, {})
            text = get_document_text(doc)
            short_text = text if len(text) < 120 else text[:117] + "..."
            print(f"- label {label}  score {score:.3f}  {short_text}")

    # évaluation globale
    print("\nÉvaluation du moteur dense sur valid.tsv")
    mean_p, mean_r, mean_f1, auc = evaluate_dense_on_qrels(
        queries,
        qrels,
        embeddings,
        doc_ids,
        model,
    )
    print(f"Précision moyenne  {mean_p:.3f}")
    print(f"Rappel moyen      {mean_r:.3f}")
    print(f"F mesure moyenne  {mean_f1:.3f}")
    if auc is not None:
        print(f"AUC globale       {auc:.3f}")
    else:
        print("AUC non calculable")

    print("\nRecherche libre dense sur tout le corpus")
    query_text_free = "sentiment analysis financial microblogs"
    print(f"Requête  {query_text_free}")

    results = search_free_text_dense(
        query_text=query_text_free,
        model=model,
        embeddings=embeddings,
        doc_ids=doc_ids,
        corpus=corpus,
        top_k=10,
    )

    for rank, (doc_id, score, text) in enumerate(results, start=1):
        short_text = text if len(text) < 120 else text[:117] + "..."
        print(f"{rank:2d}. score {score:.3f}  {doc_id}  {short_text}")


if __name__ == "__main__":
    main()
