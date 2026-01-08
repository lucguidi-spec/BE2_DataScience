import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Fonction pour extraire le texte complet d'un document
def get_document_text(doc: Dict) -> str:

    title = doc.get("title") or ""
    text = doc.get("text") or ""
    if title and text:
        return title + " " + text
    elif title:
        return title
    else:
        return text

# Fonction pour construire les embeddings du corpus
def build_corpus_embeddings(
    corpus: Dict[str, Dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:

    print(f"Chargement du modèle de phrases  {model_name}")
    model = SentenceTransformer(model_name)

    # Préparation des textes à encoder
    doc_ids = sorted(corpus.keys())
    texts = [get_document_text(corpus[doc_id]) for doc_id in doc_ids]

    print(f"Encodage de {len(doc_ids)} documents")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # on normalise pour que le produit scalaire corresponde au cosinus
    )

    return doc_ids, embeddings

# Fonctions pour sauvegarder et charger les embeddings ( moins lourd ensuite )
def save_embeddings(
    file_path: str,
    doc_ids: List[str],
    embeddings: np.ndarray,
) -> None:

    # Sauvegarde des embeddings dans un fichier pickle
    data = {
        "doc_ids": doc_ids,
        "embeddings": embeddings,
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Embeddings sauvegardés dans  {file_path}")

# Fonction pour charger les embeddings depuis un fichier
def load_embeddings(file_path: str) -> Tuple[List[str], np.ndarray]:

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    doc_ids = data["doc_ids"]
    embeddings = data["embeddings"]
    print(f"Embeddings chargés depuis  {file_path}")
    return doc_ids, embeddings

# Fonction principale pour construire ou charger les embeddings
def build_or_load_embeddings(
    corpus: Dict[str, Dict],
    file_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:

    # Vérification de l'existence du fichier d'embeddings
    if os.path.exists(file_path):
        print(f"Fichier d embeddings trouvé  {file_path}")
        return load_embeddings(file_path)

    # Si le fichier n'existe pas, construire les embeddings et les sauvegarder
    print(f"Aucun fichier d embeddings trouvé à  {file_path}")
    doc_ids, embeddings = build_corpus_embeddings(
        corpus,
        model_name=model_name,
        batch_size=batch_size,
    )
    save_embeddings(file_path, doc_ids, embeddings)
    return doc_ids, embeddings
