from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, find


def extract_documents(corpus: Dict[str, Dict]) -> Tuple[List[str], List[str]]:

    texts: List[str] = []
    ids: List[str] = []

    for doc_id, doc in corpus.items():
        text = doc.get("title") or doc.get("text") or ""
        texts.append(text)
        ids.append(doc_id)

    return texts, ids


def build_bow_matrix(
    corpus: Dict[str, Dict],
) -> Tuple[csr_matrix, CountVectorizer, List[str]]:

    texts, ids = extract_documents(corpus)
    vectorizer = CountVectorizer(
        stop_words="english",  
        max_df=0.9,            
        min_df=5,              
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer, ids


def build_tfidf_matrix(
    corpus: Dict[str, Dict],
) -> Tuple[csr_matrix, TfidfVectorizer, List[str]]:

    texts, ids = extract_documents(corpus)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=5,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer, ids


def print_top_terms_in_corpus(
    X: csr_matrix,
    vectorizer,
    top_n: int = 30,
) -> None:

    weights = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    sorted_idx = np.argsort(weights)[::-1][:top_n]
    print("Termes les plus importants dans le corpus")
    for idx in sorted_idx:
        print(f"{vocab[idx]}  {weights[idx]:.3f}")


def print_vector_features(
    v: csr_matrix,
    vectorizer,
    top_n: int = 20,
) -> None:

    vocab = vectorizer.get_feature_names_out()
    _, ids, values = find(v)
    feats = [(ids[i], values[i], vocab[ids[i]]) for i in range(len(ids))]
    top_feats = sorted(feats, key=lambda x: x[1], reverse=True)[:top_n]
    print("Termes principaux pour ce document")
    for idx, val, token in top_feats:
        print(f"{token}  {val:.3f}")
