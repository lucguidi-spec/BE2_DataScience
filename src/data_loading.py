import json
from typing import Dict, Any


def _extract_id(obj: Dict[str, Any]) -> str:

    for key in ["_id", "id", "paper_id", "doc_id", "document_id"]:
        if key in obj:
            return str(obj[key])

    raise KeyError(
        f" ID non trouvé dans JSON  "
        f"clés disponibles  {list(obj.keys())}"
    )


def load_corpus(file_path: str) -> Dict[str, Dict[str, Any]]:

    corpus: Dict[str, Dict[str, Any]] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            doc_id = _extract_id(obj)
            corpus[doc_id] = obj

    return corpus


def load_queries(file_path: str) -> Dict[str, Dict[str, Any]]:

    queries: Dict[str, Dict[str, Any]] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            q_id = _extract_id(obj)
            queries[q_id] = obj

    return queries


def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:

    qrels: Dict[str, Dict[str, int]] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        first_line = True
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            # on ignore la première ligne d en tête
            if first_line:
                first_line = False
                if parts == ["query-id", "corpus-id", "score"]:
                    continue

            if len(parts) != 3:
                raise ValueError(f"Ligne mal formée dans {file_path}  {line}")

            qid, cid, rel_str = parts

            try:
                rel = int(rel_str)
            except ValueError as exc:
                raise ValueError(
                    f"Score non entier dans {file_path}  {line}"
                ) from exc

            if qid not in qrels:
                qrels[qid] = {}

            qrels[qid][cid] = rel

    return qrels
