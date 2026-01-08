# Projet moteur de recherche scientifique

Ce projet met en place un moteur de recherche d’articles scientifiques.
Il compare trois familles d’approches :

– représentations creuses (sac-de-mots, TF-IDF)  
– représentations denses (Sentence-Transformers)  
– enrichissement par la structure du graphe de citations



## Structure du projet

L’arborescence du projet est la suivante :


projet_recherche_info/

├── requirements.txt
├── README.md

├── data/
│   ├── corpus.jsonl               # corpus complet 
│   ├── queries.jsonl              # requêtes
│   ├── valid.tsv                  # paires (requête, document) avec score 0/1
│   ├── corpus_embeddings_all_MiniLM_L6_v2.pkl  # embeddings denses du corpus gardé en cache
│   ├── sample_submission.csv      # données de soumission Kaggle
│   ├── submission_dense.csv       # soumission dense simple 
│   ├── submission_dense_graph.csv # soumission dense + graphe 

├── src/
│   ├── data_loading.py            # chargement corpus / queries / qrels
│   ├── vectorizer_baseline.py     # approches creuses (CountVectorizer / TF-IDF)
│   ├── dense_encoder.py           # construction / chargement des embeddings denses
│   ├── citation_graph.py          # graphe de citations et centralités

├── build_embeddings.py            # script  pour pré-calculer les embeddings
├── main_baseline.py               # exploration + moteur de recherche creux + évaluation
├── main_dense.py                  # moteur dense (sans structure) + évaluation sur valid.tsv
├── main_dense_graph.py            # moteur dense avec embeddings enrichis par le graphe
├── main_citation.py               # construction du graphe et analyse des centralités
├── make_submission_dense.py       # génération de submission_dense.csv (dense simple)
├── make_submission_dense_graph.py # génération de submission_dense_graph.csv (dense + graphe)



## Approche creuse: représentation sac-de-mots

Cette partie décrit tout ce qui a été fait avec des représentations creuses de type sac-de-mots.

### 1. Chargement des données et mise en forme

Les données utilisées viennent de trois fichiers.

`corpus.jsonl`  
– chaque ligne contient un article scientifique  
– identifiant unique : `_id`  
– champs principaux : `title`, `text`  
– `metadata` avec `authors`, `year`, `cited_by`, `references`

`queries.jsonl`  
– chaque ligne contient une requête  
– identifiant : `_id`  
– champ `text` qui joue le rôle de titre de la requête  
– `metadata` avec les mêmes informations que pour le corpus

`valid.tsv`  
– fichier tabulé avec trois colonnes : `query-id`, `corpus-id`, `score` (0 ou 1)

Le module `src/data_loading.py` fournit trois fonctions :

– `load_corpus` qui retourne un dictionnaire `corpus[doc_id]`  
– `load_queries` qui retourne un dictionnaire `queries[query_id]`  
– `load_qrels` qui retourne un dictionnaire imbriqué `qrels[query_id][doc_id] = score`

Quelques statistiques affichées dans `main_baseline.py` :

– environ 25 657 documents dans le corpus  
– 1 000 requêtes  
– 700 requêtes annotées dans `valid.tsv`  
– 20 950 paires requête–document  
– proportion moyenne de documents pertinents par requête : environ 0,165

### 2. Construction d’un premier espace creux

Le module `src/vectorizer_baseline.py` construit une matrice documents-termes à partir du texte des articles.

Pour chaque document :  
– on récupère `title` si disponible  
– sinon on utilise `text`

#### Variante sac-de-mots (CountVectorizer)

La fonction `build_bow_matrix` fait :

– suppression des mots-outils en anglais : `stop_words="english"`  
– suppression des mots trop fréquents : `max_df=0.9`  
– suppression des mots trop rares : `min_df=5`

C’est une représentation sac-de-mots avec des fréquences entières.

Dans une première version sans TF-IDF, l’évaluation globale donnait des valeurs typiques proches de :

– précision moyenne autour de 0,11  
– rappel moyen autour de 0,03  
– F-mesure moyenne autour de 0,045  
– AUC globale autour de 0,71

Ce modèle sert de baseline creuse minimale.

#### Variante TF-IDF (TfidfVectorizer)

Pour améliorer l’encodage, la fonction `build_tfidf_matrix` utilise :

– `TfidfVectorizer` plutôt que `CountVectorizer`  
– mêmes filtres de fréquence  
– `stop_words="english"`  
– `max_df=0.9`  
– `min_df=5`  
– prise en compte de n-grammes de taille un et deux : `ngram_range=(1, 2)`

Les poids des termes ne sont plus de simples comptes mais des scores TF-IDF.
Cette pondération met en avant les mots discriminants et atténue les termes trop fréquents.

Dans le code, le choix de la variante creuse se fait dans `main_baseline.py` via la variable :

– `variant = "tfidf"`  (ou `variant = "bow"`)

### 3. Moteur de recherche sur tout le corpus

Un petit moteur de recherche a été implémenté pour respecter la consigne suivante :

prendre un ensemble de mots-clefs, calculer un score pour chaque document de la base, et retourner les dix premiers résultats.

Fonction concernée dans `main_baseline.py` :

– `search_free_text(query_text, vectorizer, X, doc_ids, corpus, top_k=10)`

Étapes :

– la requête textuelle libre est vectorisée avec `vectorizer.transform`  
– on calcule la similarité cosinus entre la requête et toutes les lignes de la matrice `X`  
– on trie tous les documents par score décroissant  
– on retourne les dix meilleurs `doc_id`, leur score et un extrait du texte (titre ou contenu)

Exemple utilisé pour tester :

– requête : `sentiment analysis financial microblogs`

Les premiers résultats sont bien des articles de sentiment analysis (Twitter, marchés financiers, etc.).
Cela montre que le modèle capture la thématique générale.

### 4. Moteur de recherche sur les candidats annotés

Pour coller au cahier des charges du projet, un moteur classe uniquement les candidats fournis pour chaque requête dans `valid.tsv`.

Fonctions principales :

`rank_candidates_for_query_bow`  
– construit le vecteur de la requête  
– extrait les vecteurs des candidats dans `X`  
– calcule les similarités cosinus  
– renvoie une liste `(candidate_id, score)` triée par score décroissant

`evaluate_bow_on_qrels`  
– applique ce classement à toutes les requêtes de `valid.tsv`  
– calcule précision, rappel et F-mesure par requête puis fait la moyenne  
– calcule une AUC globale sur tous les scores et labels

Résultat observé pour la variante TF-IDF avec n-grammes :

– précision moyenne environ 0,144  
– rappel moyen environ 0,038  
– F-mesure moyenne environ 0,058  
– AUC globale environ 0,718

Comparaison qualitative avec la baseline sac-de-mots :

– le passage à TF-IDF améliore légèrement la précision et la F-mesure  
– l’AUC passe d’environ 0,709 à 0,718  
– les termes les plus importants dans le corpus deviennent plus informatifs  
– les n-grammes TF-IDF font ressortir des expressions plus riches  
  – ex. : `genetic algorithm`, `particle swarm`, `network design`

### 5. Résumé des variantes creuses testées

Trois niveaux ont été mis en place pour l’approche creuse :

– sac-de-mots simple sur tout le corpus  
  – `CountVectorizer`  
  – vecteurs creux  
  – cosinus comme mesure de similarité

– sac-de-mots amélioré avec prétraitements  
  – suppression des stop-words  
  – filtrage des termes trop rares et trop fréquents

– représentation TF-IDF avec n-grammes  
  – `TfidfVectorizer`  
  – stop-words, seuils de fréquence  
  – unigrams et bigrams  
  – meilleure mise en évidence des termes et expressions discriminantes  
  – légère amélioration des scores de classement et de classification globale

Ces variantes constituent la partie creuse du projet.
Elles servent de point de comparaison avec les représentations denses et les approches exploitant la structure du graphe de citations.

## Approche dense: représentations sémantiques

### 1. Principe général

L’approche dense remplace les vecteurs sac-de-mots par des embeddings de phrases issus d’un modèle préentraîné Sentence-Transformers.

Idée de base :

– chaque document du corpus est encodé en un vecteur dense de dimension 384  
– chaque requête est encodée avec le même modèle  
– tous les vecteurs sont normalisés  
– la similarité est calculée avec le produit scalaire  
– avec des vecteurs normalisés, ce produit correspond à une similarité cosinus

Cette représentation capte la sémantique globale des textes.
Elle dépasse nettement les cooccurrences de mots.

### 2. Construction et réutilisation des embeddings du corpus

Fichier : `src/dense_encoder.py`

Le module `dense_encoder.py` gère le pipeline des embeddings.

Préparation du texte à encoder : `get_document_text(doc)`  
– récupère `title` si présent  
– concatène éventuellement avec `text`  
– le titre suffit souvent  
– le texte long sert de complément si le titre est absent

Construction des embeddings : `build_corpus_embeddings(corpus, model_name, batch_size)`  
– charge `sentence-transformers/all-MiniLM-L6-v2`  
– trie les identifiants pour avoir un ordre stable  
– construit la liste des textes via `get_document_text`  
– applique `model.encode(..., normalize_embeddings=True)`

Résultat :  
– `doc_ids` : liste des identifiants dans l’ordre des lignes  
– `embeddings` : matrice NumPy de forme `(25657, 384)`  
– chaque ligne est un vecteur de norme 1

Sauvegarde et chargement :  
– `save_embeddings(file_path, doc_ids, embeddings)` écrit un fichier pickle  
– `load_embeddings(file_path)` relit ces informations

Utilitaire : `build_or_load_embeddings(corpus, file_path, model_name, batch_size)`  
– si le fichier existe, il est chargé  
– sinon, il est calculé puis sauvegardé

Fichier cache :  
– `data/corpus_embeddings_all_MiniLM_L6_v2.pkl`

### 3. Moteur de recherche dense et évaluation locale

Fichier : `main_dense.py`

Chargement :  
– lecture de `corpus.jsonl`, `queries.jsonl`, `valid.tsv` via `data_loading.py`  
– affichage de statistiques comparables à l’approche creuse  
– appel à `build_or_load_embeddings` pour récupérer `doc_ids` et `embeddings`  
– construction de `doc_index` : `doc_id` → indice de ligne dans la matrice dense  
– rechargement du même modèle Sentence-Transformers pour encoder les requêtes

Classement des candidats : `rank_candidates_for_query_dense(query_text, candidate_ids, model, embeddings, doc_index)`  
– encode le texte de la requête  
– récupère les vecteurs des candidats dans `embeddings`  
– calcule le produit scalaire `d_emb @ q_emb`  
– comme les vecteurs sont normalisés, c’est un cosinus  
– retourne `(candidate_id, score)` trié par score décroissant

Évaluation globale : `evaluate_dense_on_qrels(queries, qrels, embeddings, doc_ids, model)`  
Pour chaque requête :  
– encode la requête  
– classe les candidats issus de `valid.tsv`  
– récupère `y_true` (0/1) et `y_scores` (cosinus)

Métriques :  
– prédictions binaires par seuil simple à 0,5  
– précision, rappel, F-mesure par requête avec `precision_recall_fscore_support`  
– moyenne sur l’ensemble des requêtes  
– AUC globale avec `roc_auc_score`

Résultats observés avec le modèle dense :  
– précision moyenne environ 0,684  
– rappel moyen environ 0,302  
– F-mesure moyenne environ 0,394  
– AUC globale environ 0,955 sur `valid.tsv`

Comparaison avec l’approche creuse TF-IDF :  
– précision moyenne autour de 0,144  
– rappel autour de 0,038  
– F-mesure autour de 0,058  
– AUC autour de 0,718

L’amélioration est très significative.

Recherche libre sur tout le corpus : `search_free_text_dense(query_text, model, embeddings, doc_ids, corpus, top_k)`  
– encode la requête  
– calcule le produit scalaire avec chaque ligne de `embeddings`  
– trie et renvoie les `top_k` documents avec leurs scores

Sur `sentiment analysis financial microblogs`, les premiers résultats portent sur le sentiment des microblogs financiers.
Cela illustre la pertinence de l’approche dense.

### 4. Génération du fichier de soumission Kaggle

Fichier : `make_submission_dense.py`

Le fichier `sample_submission.csv` contient :  
– colonnes `RowId`, `query-id`, `corpus-id`, `score`  
– une ligne par paire à évaluer  
– score initial à zéro dans l’exemple

Objectif : produire `submission_dense.csv` avec les mêmes colonnes et un score cosinus dense.

Lecture du fichier Kaggle :  
– ouverture via `csv.DictReader`  
– vérification des colonnes attendues  
– stockage des lignes sous forme `(RowId, query-id, corpus-id)`

Calcul des scores :  
– chargement corpus et requêtes (`load_corpus`, `load_queries`)  
– chargement ou construction des embeddings (`build_or_load_embeddings`)  
– construction de `doc_index`  
– chargement du modèle Sentence-Transformers  
– cache `query_emb_cache` pour éviter de réencoder plusieurs fois la même requête

Pour chaque `(RowId, qid, cid)` :  
– si `qid` absent du cache, encode et stocke  
– récupère `d_emb = embeddings[doc_index[cid]]`  
– calcule `score = d_emb @ q_emb`  
– si requête ou document manquant, met un score nul par sécurité  
– formate `score` avec 6 décimales

Écriture :  
– écrit `submission_dense.csv` avec l’en-tête `RowId,query-id,corpus-id,score`  
– conserve l’ordre de `sample_submission.csv`

Le fichier est accepté par Kaggle.
Score observé : AUC ≈ 0,964.

### 5. Synthèse dense contre creux

– l’approche creuse TF-IDF sert de baseline simple, AUC ≈ 0,71  
– l’approche dense basée sur `all-MiniLM-L6-v2` apporte un gain majeur  
– meilleure séparation pertinents / non-pertinents  
– AUC locale ≈ 0,955 et AUC Kaggle ≈ 0,964

La chaîne dense se découpe en trois blocs :

– `dense_encoder.py` : construction et cache des représentations denses  
– `main_dense.py` : moteur dense et évaluation locale  
– `make_submission_dense.py` : génération de soumission Kaggle

## Graphe de citations et mesures de centralité

### Construction du graphe

À partir de `corpus.jsonl` et `queries.jsonl`, un graphe orienté de citations est construit avec NetworkX :

– chaque nœud représente un article (corpus ou requête)  
– un arc `u -> v` est créé lorsque l’article `u` cite l’article `v`  
  – présence de `v` dans `metadata.references` de `u`  
– pour les requêtes, on utilise à la fois :  
  – leurs références (`references`)  
  – les articles qui les citent (`cited_by`)

Le graphe obtenu est un `DiGraph` de grande taille :

– nombre de nœuds ≈ 234 537  
– nombre d’arcs ≈ 284 123  
– densité ≈ 5×10⁻⁶  
– degré entrant moyen ≈ 1,21 avec une variance élevée  
– degré sortant moyen ≈ 1,21

Ces statistiques décrivent un graphe bibliométrique : très peu dense et très hétérogène en degrés.

### Mesures de centralité utilisées

Plusieurs mesures identifient des articles influents selon des critères différents.

Centralité de degré entrant (`nx.in_degree_centrality`)  
– mesure la proportion de nœuds pointant vers un article  
– un score élevé correspond à un article très cité

PageRank (`nx.pagerank`, `alpha = 0.85`)  
– un article reçoit un score élevé s’il est cité par des articles importants  
– mesure d’influence structurelle dans le réseau  
– pondère le nombre de citations par l’importance des citants

Centralité d’intermédiarité (betweenness)  
(`nx.betweenness_centrality`, version exacte)  
– mesure la fraction de plus courts chemins passant par un nœud  
– met en avant les nœuds jouant un rôle de pont entre sous-graphes

Pour chaque mesure :  
– extraction des 10 articles les plus centraux  
– affichage de l’identifiant, du score et du titre

### Articles influents et comparaison qualitative des mesures

Observations typiques :

Degré entrant  
– met en avant des articles très cités  
– exemples :  
  – *A Fast Learning Algorithm for Deep Belief Nets*  
  – *Sequence to Sequence Learning with Neural Networks*  
  – *Provable Data Possession at Untrusted Stores*  
  – *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*  
– correspond à des contributions majeures

PageRank  
– classement proche du degré entrant  
– souligne certains classiques structurants  
– exemples :  
  – *Gradient-based Learning Applied to Document Recognition*  
  – *Wavelet-based Statistical Signal Processing Using Hidden Markov Models*  
– capture l’influence globale dans le réseau

Betweenness  
– fait ressortir des articles transverses ou méthodologiques  
– exemples :  
  – *Edinburgh's Phrase-based MT Systems for WMT-14*  
  – *Scalable Modified Kneser-Ney Language Model Estimation*  
  – *IRSTLM*  
  – *Fields of Experts: a Framework for Learning Image Priors*  
– valeurs numériquement faibles après arrondi  
– ordre relatif informatif

Résumé :  
– degré entrant : influence par volume de citations  
– PageRank : influence par citations “de qualité”  
– betweenness : influence comme pont entre communautés

## Représentations enrichies par la structure du graphe

### Motivation

Les embeddings denses (Sentence-Transformers) modélisent la similarité sémantique.
Ils ignorent la structure des citations.

Le graphe apporte une information complémentaire :  
– deux articles liés par citation appartiennent souvent à la même communauté  
– un article peut être entouré de voisins proches, même si son texte est ambigu

Objectif : améliorer les embeddings denses en les lissant avec ceux des voisins du graphe.

### Méthode: lissage par les voisins du graphe

On utilise :

– un graphe orienté `G` construit à partir de `references` et `cited_by`  
– une matrice d’embeddings du corpus `E` de taille `(N_docs, d)`  
  – `N_docs = 25 657`  
  – `d = 384`  
– une liste `doc_ids` alignée sur `E`

Pour un document `d` :

– `e_d` : embedding dense de base  
– `N(d)` : union des prédécesseurs et successeurs appartenant au corpus et disposant d’un embedding

On construit :

```text
si N(d) non vide :
e'_d = alpha * e_d + (1 - alpha) * mean_{v in N(d)}(e_v)
sinon :
e'_d = e_d
