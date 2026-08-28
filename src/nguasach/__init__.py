"""Nguasach: cross-linguistic phonosemantics pipeline.

Confirmatory test  -- does phonetic form alone predict the correct translation
                      above a label-permutation null, across language pairs
                      (transvec/ridge alignment + top-k retrieval, k-fold CV).
Interpretability   -- which phonemes are over-represented in words whose meaning
                      sits near each semantic pole ("hexagram" association).

See docs in the approved plan and README.md for the stage DAG.
"""

__version__ = "0.1.0"

# The permutation null is a serial outer loop over many ~0.3 GFLOP ridge fits;
# multi-threaded BLAS *helps* there (10x). Only force single-threaded BLAS when
# running the null under process-pool parallelism (`--jobs > 1`), where 24
# threads x N workers would thrash -- see nulls.py.
