"""
Data Utilities - Vector processing functions.
"""

import numpy as np
from typing import Tuple


def normalize_vectors(
    vectors,                  
    batch_size: int = 1_000_000,
    inplace: bool = False,
) -> np.ndarray:
    """
    Normalize vectors to unit length (L2 norm = 1), processing in batches
    to avoid OOM on large datasets (100M+ vectors).

    Args:
        vectors:    (N, D) float32 array of vectors, or a LazyHDF5Array.
        batch_size: Number of rows to normalise per iteration.
                    Each batch uses ``batch_size × D × 4`` bytes of extra RAM.
        inplace:    If True **and** *vectors* is a plain ndarray, normalise
                    in-place and return the same object (saves one full copy).
                    Ignored for lazy/non-ndarray inputs.

    Returns:
        Normalised ``np.ndarray`` of shape (N, D), dtype float32.
        Always a plain ndarray even when *vectors* was a LazyHDF5Array.
    """
    n, d = vectors.shape[0], vectors.shape[1]

    # Decide output buffer
    if inplace and isinstance(vectors, np.ndarray):
        out = vectors
    else:
        out = np.empty((n, d), dtype=np.float32)

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        chunk = np.asarray(vectors[start:end], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        np.maximum(norms, 1e-10, out=norms)        # avoid zero-division, in-place
        np.divide(chunk, norms, out=out[start:end])
    return out


class MmapSubset:
    """A proxy wrapper that presents a subset of an array as if it were a contiguous array."""
    def __init__(self, parent, indices):
        self._parent = parent
        self._indices = indices
        self.shape = (len(indices), parent.shape[1])
        self.dtype = parent.dtype
        self.ndim = parent.ndim

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        actual_indices = self._indices[idx]
        return self._parent[actual_indices]


def split_vectors(
    vectors: np.ndarray,
    query_ratio: float = 0.01,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split vectors into base and query sets.
    
    Args:
        vectors: (N, D) array of vectors
        query_ratio: Fraction of vectors to use as queries
        seed: Random seed for reproducibility
        
    Returns:
        (base_vectors, query_vectors, base_indices, query_indices)
    """
    rng = np.random.default_rng(seed)
    n_total = len(vectors)
    n_queries = max(1, int(n_total * query_ratio))
    
    # Shuffle indices
    indices = rng.permutation(n_total)
    
    query_indices = indices[:n_queries]
    query_indices.sort()
    
    base_indices = indices[n_queries:]
    base_indices.sort()
    
    query_vectors = vectors[query_indices]
    base_vectors = MmapSubset(vectors, base_indices)
    
    return base_vectors, query_vectors, base_indices, query_indices


def compute_recall(
    results: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    Compute recall@k.
    
    Args:
        results: (Q, k) array of returned neighbor indices
        ground_truth: (Q, K) array of true neighbor indices (K >= k)
        k: Number of neighbors to consider
        
    Returns:
        Recall@k (0.0 to 1.0)
    """
    results_k = results[:, :k]
    gt_k = ground_truth[:, :k]

    matches = (results_k[:, :, None] == gt_k[:, None, :]) & (results_k[:, :, None] != -1)
    hits_per_query = matches.any(axis=-1).sum(axis=-1)

    truth_lengths = (gt_k != -1).sum(axis=-1)
    recalls = hits_per_query / np.maximum(truth_lengths, 1)
    
    return float(recalls.mean())
