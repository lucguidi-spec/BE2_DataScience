import os
from src.data_loading import load_corpus
from src.dense_encoder import build_or_load_embeddings


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Chemin pour sauvegarder/charger les embeddings
EMB_PATH = os.path.join(DATA_DIR, "corpus_embeddings_all_MiniLM_L6_v2.pkl")


def main():
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")

    # Vérification de l'existence du fichier corpus
    if not os.path.exists(corpus_path):
        print(f"Fichier manquant  {corpus_path}")
        return

    print("Chargement du corpus")
    corpus = load_corpus(corpus_path)
    print(f"Nombre de documents dans le corpus  {len(corpus)}")

    # construction ou chargement des embeddings
    doc_ids, embeddings = build_or_load_embeddings(
        corpus,
        file_path=EMB_PATH,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,  
    )

    print("Forme des embeddings")
    print(embeddings.shape)
    print(f"Nombre d identifiants  {len(doc_ids)}")


if __name__ == "__main__":
    main()
