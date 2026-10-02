# code-compass

*Poser des questions sur une grosse base de code et être orienté vers la bonne fonction, avec une mesure chiffrée de la qualité de la recherche.*

[English version](README.md)

## Pourquoi

Je voulais un assistant capable de chercher intelligemment dans une énorme base de code, parce que le plus difficile dans une première contribution open source, c'est de trouver *où* les choses se passent. `grep` marche quand on connaît déjà le nom de la fonction ; il n'aide pas pour « où sont tirés les échantillons bootstrap de chaque arbre ? ».

code-compass est un pipeline RAG (génération augmentée par la recherche) pensé pour le code source. Il découpe un dépôt en morceaux qui ont du sens, les indexe dans ChromaDB et dans un index BM25, retrouve les morceaux les plus pertinents pour une question, et peut demander à Claude d'y répondre en citant les fichiers et les lignes exacts.

Plutôt que de m'arrêter à une démo, j'ai construit un benchmark pour mesurer quels choix de conception aident vraiment.

## Ce que ça fait

```text
$ code-compass search "How is the initial set of cluster centers chosen so that the centers are spread out?" -k 3
 1. sklearn/cluster/_kmeans.py:181-260  _kmeans_plusplus  [function]  score=0.032
 2. sklearn/cluster/_kmeans.py:1506-1563  KMeans.fit  [method]  score=0.029
 3. sklearn/cluster/_kmeans.py:937-948  _BaseKMeans._validate_center_shape  [method]  score=0.029
```

`code-compass ask "..."` envoie les morceaux retrouvés à Claude sous forme de documents séparés avec les citations activées : chaque affirmation de la réponse renvoie à un passage cité d'un fichier et d'un symbole précis.

## Fonctionnement

1. **Découpage.** Deux stratégies sont comparées :
   - *Fenêtres de lignes* : 60 lignes avec 15 lignes de recouvrement, la référence habituelle.
   - *Morceaux AST* : le fichier est analysé avec [tree-sitter](https://tree-sitter.github.io/). Chaque fonction et méthode devient un morceau. Chaque classe a un morceau résumé (signature, docstring, attributs et liste de ses méthodes). Le code de premier niveau (imports, constantes) est regroupé en blocs. Les fonctions de plus de 80 lignes sont coupées en fenêtres qui répètent toutes la signature.
2. **En-tête de contexte (optionnel).** Avant l'indexation, chaque morceau peut être précédé d'une ligne comme `# sklearn/cluster/_kmeans.py :: KMeans.fit (method, lines 1506-1563)`. Le corps d'une méthode seul dit rarement à quelle classe ou quel fichier il appartient ; l'en-tête le rajoute.
3. **Recherche.**
   - *BM25* avec un tokeniseur adapté au code : `check_is_fitted` et `KMeansPlusPlus` sont indexés entiers *et* découpés en sous-mots, donc « is fitted » correspond quand même.
   - *Dense* : embeddings dans une collection ChromaDB persistante (distance cosinus). Le modèle par défaut est all-MiniLM-L6-v2 via ONNX Runtime, sans GPU ni PyTorch. N'importe quel modèle sentence-transformers peut être branché avec `--embedding`.
   - *Hybride* : fusion des deux listes par Reciprocal Rank Fusion (RRF). Elle n'utilise que les rangs, donc aucun score à normaliser.
4. **Réponse.** Claude reçoit les meilleurs morceaux comme documents citables, avec la consigne de le dire quand les extraits ne contiennent pas la réponse.

## Benchmark

**Corpus :** [scikit-learn 1.5.2](https://github.com/scikit-learn/scikit-learn/tree/1.5.2), package `sklearn/` sans les tests : 269 fichiers Python et environ 186 000 lignes, soit 6 408 morceaux AST ou 4 450 fenêtres de lignes.

**Questions :** [67 questions écrites à la main](eval/questions.yaml), chacune avec la ou les fonctions qui y répondent, vérifiées dans le code.
- 42 questions *en langage naturel* qui évitent les noms du code (« How is a seed or None turned into a random number generator instance? »). C'est ce que taperait un nouveau venu.
- 25 questions *par mot-clé* qui citent un identifiant (« How does StratifiedShuffleSplit generate its splits? »).

**La pertinence** est définie sur le code et non sur les identifiants de morceaux, pour juger toutes les stratégies de découpage de la même façon : un morceau retrouvé compte s'il vient d'un fichier attendu et recouvre les lignes du symbole attendu.

![Recall@5 par stratégie de découpage et mode de recherche](docs/recall_at_5.png)

| Index | Recherche | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Bon fichier dans le top 5 | Latence (médiane) |
|---|---|---|---|---|---|---|---|
| Fenêtres de 60 lignes | BM25 | 33 % | 49 % | 66 % | 0,41 | 93 % | 1 ms |
| Fenêtres de 60 lignes | Dense | 22 % | 57 % | 69 % | 0,37 | 93 % | 206 ms |
| Fenêtres de 60 lignes | Hybride | 28 % | 58 % | 72 % | 0,39 | 93 % | 201 ms |
| Fenêtres + en-tête chemin | BM25 | 33 % | 49 % | 69 % | 0,42 | 91 % | 2 ms |
| Fenêtres + en-tête chemin | Dense | **40 %** | 64 % | 78 % | **0,50** | 96 % | 197 ms |
| Fenêtres + en-tête chemin | Hybride | 30 % | 58 % | 81 % | 0,43 | 96 % | 201 ms |
| Morceaux AST | BM25 | 27 % | 49 % | 64 % | 0,36 | 87 % | 1 ms |
| Morceaux AST | Dense | 24 % | 48 % | 58 % | 0,34 | 88 % | 194 ms |
| Morceaux AST | Hybride | 27 % | 51 % | 69 % | 0,37 | 90 % | 204 ms |
| Morceaux AST + en-tête chemin/symbole | BM25 | 31 % | 66 % | 73 % | 0,43 | 91 % | 1 ms |
| Morceaux AST + en-tête chemin/symbole | Dense | 31 % | 64 % | 72 % | 0,45 | 96 % | 202 ms |
| **Morceaux AST + en-tête chemin/symbole** | **Hybride** | 34 % | **69 %** | **84 %** | **0,50** | **97 %** | 204 ms |

La latence est le temps de requête sur un CPU 4 cœurs ; la recherche dense est dominée par l'embedding de la question.

### Ce que j'en retiens

- **Le contexte compte plus que les frontières des morceaux.** Le découpage AST *seul* ne fait pas mieux que les fenêtres de lignes, et même un peu moins bien : une méthode extraite de sa classe perd le nom de la classe et du fichier. Ajouter une ligne d'en-tête avec le chemin et le symbole qualifié est l'amélioration la plus forte (Recall@10 de 69 % à 84 % en recherche hybride).
- **L'en-tête rend le découpage AST rentable.** Avec l'en-tête, les morceaux AST battent les fenêtres de lignes au Recall@5 (69 % contre 58 %) et sur les questions qui citent un identifiant (80 % contre 64 % de Recall@5), car le nom du symbole est désormais dans le texte indexé.
- **La recherche hybride est le choix le plus sûr.** Elle a le meilleur Recall@10 dans les quatre configurations d'index. BM25 est fort quand la question nomme le code ; les embeddings aident quand elle ne le fait pas.
- **Trouver le fichier est facile, trouver la fonction beaucoup moins.** Le bon fichier est dans le top 5 pour 97 % des questions, la fonction exacte seulement pour 69 %. Les ratés sont en général des voisins : le résumé de la classe `StratifiedKFold` au lieu de sa méthode `_make_test_folds`, ou `BaseShuffleSplit.split` au lieu de `train_test_split`.
- **Précaution :** avec 67 questions, une question vaut 1,5 point, donc des écarts de moins de 5 points environ ne sont pas significatifs.

## Utilisation

```bash
pip install -e .                     # ajouter ".[llm]" pour `ask`, ".[st]" pour d'autres modèles d'embedding
git clone --depth 1 --branch 1.5.2 https://github.com/scikit-learn/scikit-learn.git corpora/scikit-learn

code-compass index corpora/scikit-learn               # morceaux AST + en-tête (par défaut)
code-compass search "where is the confusion matrix computed?"
ANTHROPIC_API_KEY=... code-compass ask "How does early stopping work in HistGradientBoosting?"

code-compass bench corpora/scikit-learn               # refait le tableau ci-dessus (environ 10 min sur CPU)
python scripts/plot_results.py                        # redessine le graphique
```

Options : `--chunker ast|lines`, `--no-header`, `--mode bm25|dense|hybrid`, `--embedding <modèle sentence-transformers>`.

## Organisation du projet

```text
codecompass/
  chunking.py   fenêtres de lignes et morceaux AST tree-sitter
  search.py     tokeniseur pour le code, BM25, recherche dense (ChromaDB) et hybride (RRF)
  index.py      construction / chargement d'un index (morceaux en JSONL + collection ChromaDB)
  evaluate.py   résolution des symboles, Recall@k, MRR, nDCG
  generate.py   réponses avec Claude et citations
  cli.py        index | search | ask | bench
eval/questions.yaml   le benchmark
results/bench.json    résultats bruts, question par question
```

## Limites et suites

- Un seul modèle d'embedding a été mesuré, un modèle de texte généraliste. Un modèle spécialisé code (par exemple `jinaai/jina-embeddings-v2-base-code`) peut être essayé avec `--embedding` et devrait aider sur les questions en langage naturel.
- Un reranker cross-encoder sur les 20 premiers résultats hybrides est la suite logique pour le Recall@1.
- La qualité des réponses n'est pas encore mesurée : la prochaine étape est de noter les réponses de Claude sur les mêmes questions (exactitude, et chaque affirmation appuyée par une citation).
- Python seulement pour l'instant ; avec tree-sitter, d'autres langages se résument à ajouter des grammaires.

## Licence

MIT
