import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional


def build_citation_graph(
    corpus: Dict[str, Dict],
    queries: Optional[Dict[str, Dict]] = None,
    include_query_edges: bool = True,
) -> nx.DiGraph:

    G = nx.DiGraph()

    # ajouter les nœuds du corpus
    for doc_id, doc in corpus.items():
        meta = doc.get("metadata", {})
        year = meta.get("year")
        G.add_node(
            doc_id,
            kind="corpus",
            year=year,
        )

    # ajouter les arcs de citation entre documents du corpus
    for doc_id, doc in corpus.items():
        meta = doc.get("metadata", {})
        refs = meta.get("references", []) or []
        for ref_id in refs:
            if ref_id not in G:
                G.add_node(ref_id, kind="unknown", year=None)
            G.add_edge(doc_id, ref_id, relation="ref")

    # ajouter les nœuds et arcs des requêtes (optionnel)
    if queries is not None and include_query_edges:
        for qid, qdoc in queries.items():
            qmeta = qdoc.get("metadata", {})
            year = qmeta.get("year")

            # si la requête est déjà présente en tant que nœud du corpus
            if qid in G:
                G.nodes[qid]["kind"] = "query"
                if year is not None:
                    G.nodes[qid]["year"] = year
            else:
                G.add_node(qid, kind="query", year=year)

            # arcs de la requête vers ses références
            refs = qmeta.get("references", []) or []
            for ref_id in refs:
                if ref_id not in G:
                    G.add_node(ref_id, kind="unknown", year=None)
                G.add_edge(qid, ref_id, relation="ref")

            # arcs depuis les documents qui la citent (cited_by -> requête)
            cited_by = qmeta.get("cited_by", []) or []
            for citing_id in cited_by:
                if citing_id not in G:
                    G.add_node(citing_id, kind="unknown", year=None)
                G.add_edge(citing_id, qid, relation="cited_by")

    return G


def basic_graph_stats(G: nx.DiGraph) -> Dict[str, float]:

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)

    in_degrees = np.array([deg for _, deg in G.in_degree()])
    out_degrees = np.array([deg for _, deg in G.out_degree()])

    stats = {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "density": float(density),
        "in_degree_mean": float(in_degrees.mean()),
        "in_degree_var": float(in_degrees.var()),
        "out_degree_mean": float(out_degrees.mean()),
        "out_degree_var": float(out_degrees.var()),
    }
    return stats


def _node_title(
    node_id: str,
    corpus: Dict[str, Dict],
    queries: Optional[Dict[str, Dict]] = None,
    max_len: int = 100,
) -> str:

    doc = None
    if node_id in corpus:
        doc = corpus[node_id]
        title = doc.get("title") or doc.get("text") or ""
    elif queries is not None and node_id in queries:
        doc = queries[node_id]
        title = doc.get("title") or doc.get("text") or ""
    else:
        title = ""

    if not title:
        return "(titre nc)"

    title = title.replace("\n", " ").strip()
    if len(title) > max_len:
        return title[: max_len - 3] + "..."
    return title


def compute_centrality_measures(
    G: nx.DiGraph,
    corpus: Dict[str, Dict],
    queries: Optional[Dict[str, Dict]] = None,
    top_k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:

    centralities = {}

    print("\nCalcul des degrés de centralités entrant")
    in_deg_cent = nx.in_degree_centrality(G)
    centralities["in_degree_centrality"] = sorted(
        in_deg_cent.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    print("Calcul du PageRank")
    pagerank = nx.pagerank(G, alpha=0.85)
    centralities["pagerank"] = sorted(
        pagerank.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    print("Calcul de la centralité de la betweenness")
    betweenness = nx.betweenness_centrality(G, k=200, normalized=True, seed=42)
    centralities["betweenness"] = sorted(
        betweenness.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    for measure_name, top_nodes in centralities.items():
        print(f"\nTop {top_k} nœuds selon  {measure_name}")
        for node_id, score in top_nodes:
            title = _node_title(node_id, corpus, queries)
            print(f"- {node_id}  score {score:.4f}  {title}")

    return centralities



def build_graph_enhanced_embeddings(
    G: nx.DiGraph,
    doc_ids: List[str],
    embeddings: np.ndarray,
    alpha: float = 0.8,
    normalize: bool = True,
) -> np.ndarray:
  
    num_docs, dim = embeddings.shape
    new_embeddings = np.zeros_like(embeddings, dtype=np.float32)

    # dictionnaire doc_id -> index dans la matrice
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    for i, doc_id in enumerate(doc_ids):
        # voisins = prédécesseurs U successeurs
        neighbors = set(G.predecessors(doc_id)) | set(G.successors(doc_id))

        neighbor_vecs = []
        for n in neighbors:
            j = id_to_idx.get(n)
            if j is not None:
                neighbor_vecs.append(embeddings[j])

        if neighbor_vecs:
            neigh_mean = np.mean(neighbor_vecs, axis=0)
            vec = alpha * embeddings[i] + (1.0 - alpha) * neigh_mean
        else:
            # pas de voisins dans le corpus -> on garde l'embedding original
            vec = embeddings[i]

        new_embeddings[i] = vec

    if normalize:
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        new_embeddings = new_embeddings / norms

    return new_embeddings
