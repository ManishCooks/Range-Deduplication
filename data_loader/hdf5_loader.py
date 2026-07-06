"""
HDF5 Dataset Loader - Load ANN benchmark datasets.

Standard ANN benchmark HDF5 format:
- train: base vectors to ingest (N x D)
- test: query vectors (Q x D)
- neighbors: ground truth k-NN indices (Q x K)
- distances: ground truth distances (Q x K)
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


# Dataset directory relative to project root
DATASETS_DIR = Path(__file__).parent.parent / "datasets"


# =============================================================================
# LazyHDF5Array — streaming view over an HDF5 dataset
# =============================================================================

class LazyHDF5Array:
    """
    A thin wrapper around an h5py Dataset that keeps the HDF5 file open and
    provides the same shape / dtype / slicing interface as a numpy array,
    but data is only read from disk when a slice is accessed.

    This lets workloads stream through 100M+ vector datasets without ever
    materialising the full array in RAM.  Peak extra memory per slice is
    exactly  ``batch_size × dim × 4``  bytes.

    Usage::

        lazy = LazyHDF5Array(path, key="train")
        dim   = lazy.shape[1]          # free — metadata only
        batch = lazy[0:500_000]        # reads 500k rows → np.float32 ndarray
        lazy.close()                   # optional — also closed by GC / __del__

    Compatibility notes
    -------------------
    * ``len(lazy)``  →  number of rows
    * ``lazy.shape`` →  (N, D) tuple
    * ``lazy[i:j]``  →  np.ndarray (float32)
    * ``lazy[np.array([…])]``  →  fancy-index read (h5py supports this natively)
    * Works transparently with :func:`ingest_vectors` (uses ``vectors[i:j]``)
      and :func:`normalize_vectors` (uses ``vectors[i:j]``).
    * Does *not* support in-place writes (``lazy[i] = x`` will raise).
    """

    def __init__(self, path: Path, key: str = "train", dtype=np.float32) -> None:
        self._file  = h5py.File(path, "r")
        self._ds    = self._file[key]
        self._dtype = dtype
        self.shape  = self._ds.shape        # (N, D)
        self.dtype  = np.dtype(dtype)
        self.ndim   = len(self.shape)

    # ------------------------------------------------------------------
    # Array-like interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        """Read a slice or fancy index from HDF5 and cast to float32."""
        return self._ds[idx].astype(self._dtype)

    # Needed by np.asarray(), np.array() — iterate rows as float32 ndarray.
    def __array__(self, dtype=None):
        arr = self._ds[:].astype(self._dtype)
        return arr if dtype is None else arr.astype(dtype)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HDF5 file handle (idempotent)."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None   

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers used by workloads
    # ------------------------------------------------------------------

    def astype(self, dtype):
        """Return self — dtype is always applied at slice-read time."""
        self._dtype = np.dtype(dtype).type
        self.dtype  = np.dtype(dtype)
        return self

    def materialize(self, batch_size: int = 500_000) -> np.ndarray:
        """
        Fully materialise into RAM in batches.  Use only when the caller
        genuinely needs a contiguous ndarray (e.g. permutation shuffles).

        Args:
            batch_size: Rows to copy per iteration (controls peak extra RAM).

        Returns:
            np.ndarray of shape (N, D), dtype float32.
        """
        n, d = self.shape
        out  = np.empty((n, d), dtype=self._dtype)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            out[start:end] = self._ds[start:end].astype(self._dtype)
        return out


# =============================================================================
# Path helpers
# =============================================================================

def get_dataset_path(name: str) -> Path:
    """Get full path to dataset file."""
    path = Path(name)
    if path.exists() and path.suffix == ".hdf5":
        return path

    dataset_path = DATASETS_DIR / f"{name}.hdf5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return dataset_path


def get_dataset_info(name: str) -> Dict[str, Any]:
    """
    Get dataset metadata without loading full data.

    Returns:
        Dict with keys, shapes, dtypes
    """
    path = get_dataset_path(name)
    info: Dict[str, Any] = {"path": str(path), "keys": {}}

    with h5py.File(path, "r") as f:
        for key in f.keys():
            info["keys"][key] = {
                "shape": f[key].shape,
                "dtype": str(f[key].dtype),
            }

    return info


# =============================================================================
# Main loader
# =============================================================================

def load_hdf5_dataset(
    name: str,
    load_train: bool = True,
    load_test: bool = True,
    load_neighbors: bool = False,
    load_distances: bool = False,
    max_vectors: Optional[int] = None,
    mmap: bool = False,
) -> Dict[str, Any]:
    """
    Load ANN benchmark HDF5 dataset.

    Args:
        name:            Dataset name (e.g. ``"glove1M"``) or full HDF5 path.
        load_train:      Load training/base vectors.
        load_test:       Load test/query vectors.
        load_neighbors:  Load ground truth neighbor indices.
        load_distances:  Load ground truth distances.
        max_vectors:     Limit number of train vectors (for testing).
        mmap:            **OOM-safe mode.**  When *True* the ``train`` split is
                         returned as a :class:`LazyHDF5Array` — the HDF5 file
                         stays open and rows are read from disk on demand.
                         Peak RAM for ``train`` is then bounded by the batch
                         size used by the caller (e.g. ``ingest_batch_size``),
                         not by ``N × D × 4`` bytes.

                         All other splits (``test``, ``neighbors``,
                         ``distances``) are still loaded eagerly because they
                         are small relative to ``train``.

                         Defaults to *False* so all existing call-sites are
                         unaffected.

    Returns:
        Dict with loaded arrays. ``train`` is a :class:`LazyHDF5Array` when
        *mmap=True*, otherwise a plain ``np.ndarray``.
    """
    path = get_dataset_path(name)
    data: Dict[str, Any] = {}

    if mmap:
        # ── Lazy path ───────────────────────────────────────────────────
        _owned_file: Optional[h5py.File] = None 

        if load_train:
            with h5py.File(path, "r") as _peek:
                has_train = "train" in _peek

            if has_train:
                lazy = LazyHDF5Array(path, "train", dtype=np.float32)

                if max_vectors and max_vectors < lazy.shape[0]:
                    class _Capped(LazyHDF5Array):
                        def __init__(self, parent: LazyHDF5Array, n: int) -> None:  # type: ignore[override]
                            self._file  = parent._file
                            self._ds    = parent._ds
                            self._dtype = parent._dtype
                            self.shape  = (n,) + parent.shape[1:]
                            self.dtype  = parent.dtype
                            self.ndim   = parent.ndim
                            parent._file = None

                        def __getitem__(self, idx):
                            if isinstance(idx, slice):
                                start, stop, step = idx.indices(self.shape[0])
                                idx = slice(start, stop, step)
                            return self._ds[idx].astype(self._dtype)

                    lazy = _Capped(lazy, max_vectors)

                data["train"] = lazy
                src = lazy._file    # reuse — do NOT open a second handle
            else:
                # "train" key absent: fall through with no src from lazy
                src = h5py.File(path, "r")
                _owned_file = src
        else:
            # Not loading train at all — open file solely for small splits
            src = h5py.File(path, "r")
            _owned_file = src

        # Load remaining splits using the already-open handle
        try:
            if load_test and "test" in src:
                data["test"] = src["test"][:].astype(np.float32)
            if load_neighbors and "neighbors" in src:
                data["neighbors"] = src["neighbors"][:].astype(np.int32)
            if load_distances and "distances" in src:
                data["distances"] = src["distances"][:].astype(np.float32)
        finally:
            # Only close the handle if we own it (i.e. lazy is not using it)
            if _owned_file is not None:
                _owned_file.close()

        return data

    # ── Eager path  ─────────────────────────────────
    with h5py.File(path, "r") as f:
        if load_train and "train" in f:
            train = f["train"][:]
            if max_vectors and max_vectors < len(train):
                train = train[:max_vectors]
            data["train"] = train.astype(np.float32)

        if load_test and "test" in f:
            data["test"] = f["test"][:].astype(np.float32)

        if load_neighbors and "neighbors" in f:
            data["neighbors"] = f["neighbors"][:].astype(np.int32)

        if load_distances and "distances" in f:
            data["distances"] = f["distances"][:].astype(np.float32)

    return data


# =============================================================================
# Convenience helpers
# =============================================================================

def yield_hdf5_batches(name: str, batch_size: int = 10_000):
    """Yield train batches one at a time — never materialises the full array."""
    path = get_dataset_path(name)
    with h5py.File(path, "r") as f:
        dataset = f["train"]
        n_total = len(dataset)
        for i in range(0, n_total, batch_size):
            yield dataset[i:i + batch_size].astype(np.float32)


def load_train_vectors(name: str, max_vectors: Optional[int] = None) -> np.ndarray:
    """Load only training/base vectors (eager)."""
    data = load_hdf5_dataset(name, load_train=True, load_test=False, max_vectors=max_vectors)
    return data["train"]


def load_test_vectors(name: str) -> np.ndarray:
    """Load only test/query vectors."""
    data = load_hdf5_dataset(name, load_train=False, load_test=True)
    return data["test"]


def load_ground_truth(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth neighbors and distances."""
    data = load_hdf5_dataset(
        name,
        load_train=False,
        load_test=False,
        load_neighbors=True,
        load_distances=True,
    )
    return data["neighbors"], data["distances"]


# =============================================================================
# Helper for external query vectors
# =============================================================================

def load_query_vectors_file(path_str: str) -> np.ndarray:
    """
    Load an external query vectors file. Supports .npy, .hdf5, .h5.
    For HDF5, it expects the keys to be named 'test' or 'train' or uses the 'first key'.
    """
    path = Path(path_str)
    if not path.exists():
        # Try to resolve relative to datasets dir
        path = DATASETS_DIR / path_str
        if not path.exists():
            raise FileNotFoundError(f"Query vectors file not found: {path_str}")

    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)
    elif ext in (".hdf5", ".h5"):
        with h5py.File(path, "r") as f:
            if "test" in f:
                return f["test"][:]
            elif "train" in f:
                return f["train"][:]
            else:
                keys = list(f.keys())
                if keys:
                    return f[keys[0]][:]
                raise KeyError(f"No suitable datasets found in {path}")
    else:
        raise ValueError(f"Unsupported query vectors file format: {ext}. Expected .npy, .hdf5, or .h5")