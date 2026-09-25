# Business Entity Resolution for Amazon ML Challenge 2026

## Project Structure
- `src/preprocessing.py`: Text normalization and feature extraction.
- `src/blocking.py`: Lexical and rule-based candidate generation.
- `src/features.py`: Fuzzy string metrics and token overlaps.
- `src/evaluate.py`: Macro F0.5 metric calculation and threshold tuning.
- `src/model.py`: LightGBM binary classifier for semantic reranking.
- `src/pipeline.py`: Orchestrates end-to-end execution.

## Requirements
To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Pipeline
To execute the baseline pipeline on test data:
```bash
export PYTHONPATH=.
python3 src/pipeline.py --data_dir ../../student_resource/dataset --candidate_out ../output/candidate_pairs.tsv --matching_out ../output/matching_results.tsv
```

For a fast trial on a small subset:
```bash
python3 src/pipeline.py --data_dir ../../student_resource/dataset --subset 100 --candidate_out ../output/candidate_pairs.tsv --matching_out ../output/matching_results.tsv
```
