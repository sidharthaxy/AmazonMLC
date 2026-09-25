# System Documentation

## Pipeline Overview
This is a high-performance Entity Resolution pipeline handling multiple noisy data sources across multiple countries. It uses an open-source LightGBM classifier as per the 8 Billion parameter constraint.

### Preprocessing
We perform text lowercasing, expand common corporate abbreviations (e.g. `pvt` -> `private`, `ltd` -> `limited`), standardize address components (`rd` -> `road`), and extract ZIP codes and numbers as independent matchable features.

### Blocking (Candidate Generation)
The strategy is partitioned by `country`. It implements:
1. **Lexical Blocking**: TF-IDF Vectorizer with character n-grams (3-4), followed by cosine similarity retrieval using `NearestNeighbors`.
2. **Rule-Based Blocking**: Fast lookup on matching country and first token of business name, as well as exact matching on postal codes.

### Feature Engineering
For every candidate pair (S1 -> S2/S3), the system calculates:
- Name similarities: Levenshtein ratio, Jaro-Winkler, Token Sort/Set Ratio.
- Address similarities: Jaccard similarity and character N-Gram Cosine Similarity.
- Exact match indicators for extracted numerical sequences.

### Model & Evaluation
A LightGBM Binary Classifier is trained to classify candidate pairs as a match or mismatch. Threshold optimization is performed using Grid Search to maximize the precision-heavy macro F_0.5 metric.

### Evaluation Metric
Macro-averaged F_0.5: `(1.25 * P * R) / (0.25 * P + R)`
Singletons correctly identified as empty receive 1.0, otherwise 0.0.
