# Amazon ML Challenge 2026: Business Entity Resolution
### Team `aads`

A scalable, memory-efficient Machine Learning solution for Business Entity Resolution across noisy, independent catalog sources (Source 1 reference against Source 2 and Source 3 target pools).

Designed to process $> 11$ million multi-country records within **32 GB RAM** on an AWS SageMaker `ml.m5.2xlarge` instance without triggering the Linux OOM killer.

---

## Key Highlights & Innovations

1. **Compact Inverted Indexing (`np.uint32`)**:
   Stores candidate index posting lists as 32-bit unsigned integer arrays, reducing memory by over $85\%$ compared to string objects or dense matrices.
2. **High-Frequency Key Pruning**:
   Keys indexing $> 10,000$ target records (generic business stop words and common tokens) are pruned to prevent $O(N \times M)$ quadratic candidate blowups.
3. **$S1$ Batch Chunking**:
   Slices Source 1 into batches (e.g. 50,000 rows). Candidate generation, feature extraction, and model scoring execute per chunk followed by immediate `gc.collect()`.
4. **Candidate Capping**:
   Limits maximum candidates to 30 per $S1$ query record, prioritizing candidates by blocking key overlap counts.
5. **Vectorized Feature Extraction**:
   Uses direct integer array indexing with RapidFuzz for high-speed similarity calculation ($> 500\text{k}$ pairs/sec) without DataFrame merge overhead.
6. **XGBoost Classifier + Fast Heuristic Fallback**:
   Trained on ground truth positive and negative pairs to maximize macro-averaged $F_{0.5}$. Checkpoint is only 359 KB ($< 8\text{B}$ parameter constraint).
7. **Streaming TSV Output**:
   Flushes predictions incrementally to disk (`output/matching_results.tsv`), adhering strictly to the competition format.

---

## Directory Structure
```
AmazonMLC/
├── README.md                                          # Project root documentation
├── models/
│   └── entity_model.json                              # Trained XGBoost model checkpoint
├── aads_submission/
│   ├── Documentation_template.md                      # Official challenge documentation
│   └── business_entity_resolution/
│       └── code/
│           ├── README.md                              # Code guide
│           ├── requirements.txt                       # Python dependencies
│           └── src/
│               ├── candidate_generation.py            # Memory-compact inverted index & batching
│               ├── blocking.py                        # Backward-compatibility alias
│               ├── features.py                        # Vectorized pairwise similarity features
│               ├── model.py                           # XGBoost classification & scoring
│               ├── preprocessing.py                   # Regex text normalization & token extraction
│               ├── evaluate.py                        # Macro F_0.5 evaluation & threshold tuning
│               └── pipeline.py                        # Country-partitioned execution pipeline
└── student_resource/
    └── dataset/                                       # Competition datasets
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r aads_submission/business_entity_resolution/code/requirements.txt
```

### 2. Run Entity Resolution Pipeline
```bash
export PYTHONPATH=.
python3 aads_submission/business_entity_resolution/code/src/pipeline.py \
    --data_dir student_resource/dataset \
    --matching_out output/matching_results.tsv \
    --candidate_out output/candidate_pairs.tsv
```

### 3. Fast Verification Run (Subset)
```bash
export PYTHONPATH=.
python3 aads_submission/business_entity_resolution/code/src/pipeline.py \
    --data_dir student_resource/dataset \
    --subset 2000
```

---

## Output Format
Predictions are written to `output/matching_results.tsv`:
```tsv
source1_entity_id	matched_entity_ids
S1-156285671	S2-611995673,S2-522132855
S1-717749279	
S1-913506265	S3-867068018
```
Singletons are represented as empty strings after the tab.
