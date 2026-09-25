import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT_DIR)
for p in [CODE_DIR, os.path.join(os.getcwd(), 'aads_submission', 'business_entity_resolution', 'code')]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Backward compatibility bridge to candidate_generation.py
from src.candidate_generation import (
    CompactInvertedIndex,
    generate_candidates,
    get_first_token
)

__all__ = ['CompactInvertedIndex', 'generate_candidates', 'get_first_token']
