import os

from src.data_loading import load_corpus, load_queries
from src.citation_graph import (
    build_citation_graph,
    basic_graph_stats,
    compute_centrality_measures,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    queries_path = os.path.join(DATA_DIR, "queries.jsonl")

    # Vérifier l'existence des fichiers de données
    if not os.path.exists(corpus_path):
        print(f"Fichier corpus manquant  {corpus_path}")
        return
    if not os.path.exists(queries_path):
        print(f"Fichier queries manquant  {queries_path}")
        return

    print("Chargement du corpus")
    corpus = load_corpus(corpus_path)
    print(f"Nombre de documents dans le corpus  {len(corpus)}")

    print("Chargement des requêtes")
    queries = load_queries(queries_path)
    print(f"Nombre de requêtes  {len(queries)}")

    print("\nConstruction du graphe de citations (corpus + requêtes)")
    G = build_citation_graph(corpus, queries, include_query_edges=True)

    # Afficher les statistiques du graphe
    print("\nStatistiques du graphe")
    stats = basic_graph_stats(G)
    print(f"Nombre de nœuds      {int(stats['num_nodes'])}")
    print(f"Nombre d'arcs        {int(stats['num_edges'])}")
    print(f"Densité              {stats['density']:.6f}")
    print(f"Degré entrant moyen  {stats['in_degree_mean']:.3f}")
    print(f"Degré entrant var    {stats['in_degree_var']:.3f}")
    print(f"Degré sortant moyen  {stats['out_degree_mean']:.3f}")
    print(f"Degré sortant var    {stats['out_degree_var']:.3f}")

    print("\nCalcul des indicateurs de centralité")
    # Calculer et afficher les mesures de centralité
    compute_centrality_measures(
        G,
        corpus=corpus,
        queries=queries,
        top_k=10,
    )


if __name__ == "__main__":
    main()
