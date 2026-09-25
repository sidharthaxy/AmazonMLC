# Business Entity Resolution — Team `aads`
Amazon ML Challenge 2026

## Overview
This repository contains the high-performance, memory-optimized Business Entity Resolution pipeline developed by team `aads` for the Amazon ML Challenge 2026.

The pipeline is designed to resolve entities from a deduplicated reference source (Source 1) against large, noisy target pools (Source 2 and Source 3) totaling over 11 million records across dynamic open-set countries (US, India, France).

### 32 GB RAM Architecture Optimizations
Engineered specifically to run reliably on an AWS SageMaker `ml.m5.2xlarge` instance (32 GB RAM) without triggering the Linux OOM killer:
1. **$S1$ Batch Chunking**: Slices Source 1 into batches (default: `batch_size = 50,000`). Blocking, feature extraction, scoring, and output writing are executed per batch, followed by immediate `gc.collect()`.
2. **Compact Inverted Index (`np.uint32`)**: Candidate blocking maps for the target pool store 32-bit unsigned integer arrays (`np.uint32`), reducing index memory by $> 85\%$.
3. **High-Frequency Pruning**: Keys indexing $> 10,000$ target records are dropped to eliminate quadratic $O(N \times M)$ blowups.
4. **Candidate Capping**: Candidates are prioritized by blocking key overlap count and capped at 30 candidates per $S1$ query record.
5. **Streaming Output Writing**: Predictions are flushed incrementally to `output/matching_results.tsv` and `output/candidate_pairs.tsv` to avoid accumulating all mappings in RAM.

---

## Project Structure
```
code/
├── README.md                          # Documentation and execution instructions
├── requirements.txt                   # Python dependencies
└── src/
    ├── __init__.py
    ├── candidate_generation.py        # CompactInvertedIndex (np.uint32), pruning, candidate capping
    ├── blocking.py                    # Backward-compatibility alias for candidate_generation.py
    ├── features.py                    # RapidFuzz string metrics, token Jaccard, address n-grams
    ├── model.py                       # EntityMatchingModel (XGBoost classifier & heuristic fallback)
    ├── preprocessing.py               # Fast regex text cleaning and abbreviation normalization
    ├── evaluate.py                    # Macro F_0.5 metric evaluation & threshold tuning
    └── pipeline.py                    # Orchestrates country partitioning, batching & streaming
```

---

## Installation & Setup

1. Clone the repository and install required packages:
```bash
pip install -r aads_submission/business_entity_resolution/code/requirements.txt
```

---

## Running the Pipeline

### 1. Test Inference (Default Submission Run)
To run end-to-end entity resolution on test data and stream predictions to disk:
```bash
export PYTHONPATH=.
python3 aads_submission/business_entity_resolution/code/src/pipeline.py \
    --data_dir student_resource/dataset \
    --matching_out output/matching_results.tsv \
    --candidate_out output/candidate_pairs.tsv
```

### 2. Fast Verification Run (Subset)
To test the full pipeline end-to-end on a subset (e.g., 2,000 rows):
```bash
export PYTHONPATH=.
python3 aads_submission/business_entity_resolution/code/src/pipeline.py \
    --data_dir student_resource/dataset \
    --subset 2000 \
    --matching_out output/matching_results.tsv \
    --candidate_out output/candidate_pairs.tsv
```

### 3. Model Training
To train or retrain the XGBoost classifier on training ground truth:
```bash
export PYTHONPATH=.
python3 aads_submission/business_entity_resolution/code/src/pipeline.py \
    --data_dir student_resource/dataset \
    --is_train \
    --model_path models/entity_model.json
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Path to directory containing `train/` and `test/` datasets |
| `--is_train` | `False` | Run in model training mode with ground truth pairs |
| `--matching_out` | `output/matching_results.tsv` | Destination path for final prediction results |
| `--candidate_out` | `output/candidate_pairs.tsv` | Destination path for generated candidate pairs |
| `--model_path` | `models/entity_model.json` | Path to load/save the trained XGBoost model |
| `--batch_size` | `50000` | Number of $S1$ records processed per batch |
| `--top_k` | `30` | Maximum candidate matches retained per $S1$ record |
| `--threshold` | `0.55` | Decision probability threshold for match acceptance |
| `--subset` | `0` | Number of $S1$ records to process (`0` for all records) |

---

## Output Schema

The final `output/matching_results.tsv` strictly conforms to the competition format:
- Tab-separated values (`.tsv`)
- Header: `source1_entity_id\tmatched_entity_ids`
- One row for every entity in Source 1
- Matched IDs from Source 2 and Source 3 separated by commas without spaces
- Singletons (no matches found) represented by an empty string

Example:
```tsv
source1_entity_id	matched_entity_ids
S1-156285671	S2-611995673,S2-522132855
S1-717749279	
S1-913506265	S3-867068018
```
