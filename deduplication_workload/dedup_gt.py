"""
dedup_gt.py — Exact Jaccard Ground Truth Oracle for Deduplication Workload

For each novel vector v_i, computes:
    gt_max_jaccard[i] = max_j( J(S_{v_i}, S_{b_j}) )

where S_x is the sign-binarized set of positive dimensions of x, and
J(A, B) = |A ∩ B| / |A ∪ B|  is the exact Jaccard similarity.

The returned float array is THRESHOLD-FREE. Apply the threshold in the caller:
    gt_reject = gt_max_jaccard >= jaccard_threshold

Design constraints:
  - Base vectors: chunked mmap reads — never loaded fully into RAM (billion-scale)
  - Query vectors: fully in RAM 
  - Core compute: vectorized matrix multiply (float32) — no Python loops over pairs
  - Memory: O(query_batch x base_chunk x 4 bytes) at peak
"""

import numpy as np
from typing import Optional


# =============================================================================
# Public API
# =============================================================================

def compute_gt(
    base_vectors,
    query_vectors: np.ndarray,
    chunk_size:        int = 100_000,
    query_batch_size:  int = 2_000,
    verbose:           bool = True,
) -> np.ndarray:
    """
    Compute gt_max_jaccard[i] = max Jaccard similarity of query_vectors[i]
    against any vector in base_vectors, using exact set-based Jaccard on
    sign-binarized representations.

    Parameters
    ----------
    base_vectors : array-like, shape (N_base, dim)
        Base index vectors. Can be a numpy mmap slice — reads are sequential
        per chunk so the OS read-ahead (prefetching) cache is used efficiently.
    query_vectors : np.ndarray, shape (N_query, dim)
        Query vectors, fully in RAM. 
    chunk_size : int
        Number of base vectors to process per chunk. Tune so that
        (query_batch_size x chunk_size x 4 bytes) fits comfortably in RAM.
        Default 100K: peak memory ~ 2K x 100K x 4B = 800 MB.
    query_batch_size : int
        Number of novel vectors to process per inner mini-batch.
        Controls the intersection matrix size.
    verbose : bool
        Print progress per base chunk.

    Returns
    -------
    gt_max_jaccard : np.ndarray, shape (N_query,), dtype=float32
        gt_max_jaccard[i] = maximum Jaccard similarity of query_vectors[i]
        against any base vector. Threshold-free — apply threshold in caller.
    """
    n_query = len(query_vectors)
    n_base  = len(base_vectors)

    if n_query == 0 or n_base == 0:
        return np.zeros(n_query, dtype=np.float32)

    # Sign binarize query vectors once -> keep in RAM
    # active_dim[i, d] = 1 if query_vectors[i, d] > 0 else 0
    query_bin    = (query_vectors >= np.median(query_vectors, axis=1, keepdims=True)).astype(np.float32)    # (N_query, dim)
    query_counts = query_bin.sum(axis=1)                      # (N_query,) -- |S_{q_i}|

    gt_max_jaccard = np.zeros(n_query, dtype=np.float32)

    n_base_chunks  = (n_base  + chunk_size       - 1) // chunk_size

    # Iterate over base chunks (mmap-safe increasing index reads)
    for c_idx, base_start in enumerate(range(0, n_base, chunk_size)):
        base_end   = min(base_start + chunk_size, n_base)
        base_chunk = base_vectors[base_start:base_end]        

        base_bin    = (base_chunk >= np.median(base_chunk, axis=1, keepdims=True)).astype(np.float32)        # (B, dim)
        base_counts = base_bin.sum(axis=1)                        # (B,) -- |S_{b_j}|

        if verbose:
            print(
                f"[GT] Base chunk {c_idx+1}/{n_base_chunks} "
                f"({base_start}-{base_end}, {base_end - base_start} vecs)"
            )

        # Iterate over query mini-batches to control peak memory
        for qb_start in range(0, n_query, query_batch_size):
            qb_end   = min(qb_start + query_batch_size, n_query)
            query_slice  = query_bin[qb_start:qb_end]                 # (qb, dim)
            query_cnt  = query_counts[qb_start:qb_end]              # (qb,)

            # intersection[i, j] = |S_{query_i} ∩ S_{base_j}|
            # = dot product of indicator vectors (since both are 0/1)
            intersection = query_slice @ base_bin.T                   # (qb, B)

            # union[i, j] = |S_{query_i}| + |S_{base_j}| - intersection[i, j]
            union = (
                query_cnt[:, None]          # (qb, 1) broadcast
                + base_counts[None, :]    # (1, B) broadcast
                - intersection            # (qb, B)
            )                                                      # (qb, B)

            # Jaccard[i, j] = intersection / union   (0 if both empty)
            safe_union  = np.where(union > 0, union, 1.0)
            jaccard     = intersection / safe_union                # (qb, B)
            jaccard     = np.where(union > 0, jaccard, 0.0)

            # Running maximum per novel vector across all base chunks
            chunk_max = jaccard.max(axis=1)                        # (nb,)
            gt_max_jaccard[qb_start:qb_end] = np.maximum(
                gt_max_jaccard[qb_start:qb_end], chunk_max
            )

    return gt_max_jaccard


# =============================================================================
# Threshold utilities
# =============================================================================

def apply_threshold(
    gt_max_jaccard: np.ndarray,
    jaccard_threshold: float,
) -> np.ndarray:
    """
    Convert the float gt_max_jaccard array to a boolean GT label array.

    gt_reject[i] = True  -> pipeline should have rejected novel_vectors[i]
    gt_reject[i] = False -> pipeline should have inserted novel_vectors[i]
    """
    return gt_max_jaccard >= jaccard_threshold


def precision_recall_at_threshold(
    gt_max_jaccard: np.ndarray,
    pipeline_rejected: np.ndarray,
    jaccard_threshold: float,
) -> dict:
    """
    Compute precision, recall, F1, and counts at a given threshold.

    Parameters
    ----------
    gt_max_jaccard    : float32 array, shape (N_novel,)
    pipeline_rejected : bool array, shape (N_novel,) -- True = pipeline said REJECT
    jaccard_threshold : float

    Returns
    -------
    dict with keys: recall, precision, f1, tp, fn, fp, tn, gt_positive_rate
    """
    gt_reject = apply_threshold(gt_max_jaccard, jaccard_threshold)

    tp = int(np.sum( gt_reject &  pipeline_rejected))
    fn = int(np.sum( gt_reject & ~pipeline_rejected))  # missed dups
    fp = int(np.sum(~gt_reject &  pipeline_rejected))  # wrongly rejected
    tn = int(np.sum(~gt_reject & ~pipeline_rejected))

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "recall":            recall,
        "precision":         precision,
        "f1":                f1,
        "tp":                tp,
        "fn":                fn,
        "fp":                fp,
        "tn":                tn,
        "gt_positive_rate":  float(gt_reject.mean()),
    }


def sweep_thresholds(
    gt_max_jaccard: np.ndarray,
    pipeline_rejected: np.ndarray,
    thresholds: Optional[list] = None,
) -> list:
    """
    Run precision_recall_at_threshold for a list of thresholds.

    Returns list of dicts, each augmented with {"threshold": t}.
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    results = []
    for t in thresholds:
        row = precision_recall_at_threshold(gt_max_jaccard, pipeline_rejected, t)
        row["threshold"] = t
        results.append(row)
    return results
