"""
Abstract Database Adapter - Base class for all adapters.
All the database adapters should satisfy these base class methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

from database_adapter.exceptions import ConnectionError


class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()


@dataclass
class QueryResult:
    """Query result."""
    data: List[Dict[str, Any]]
    total_count: int
    execution_time_ms: float


@dataclass
class InsertResult:
    """Insert result."""
    inserted_count: int
    failed_count: int
    execution_time_ms: float


@dataclass
class DeleteResult:
    """Delete result."""
    deleted_count: int
    execution_time_ms: float


@dataclass 
class HealthStatus:
    """Health check result."""
    healthy: bool
    latency_ms: float
    message: str = ""


class DatabaseAdapter(ABC):
    """
    Abstract base for database adapters.
    
    Implement: _do_connect, _do_disconnect, insert, query, health_check
    """
    
    def __init__(self, name: str = "adapter"):
        self._name = name
        self._state = ConnectionState.DISCONNECTED
    
    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED
    
    def connect(self, **kwargs) -> None:
        """Connect to database."""
        if self.is_connected:
            return
        try:
            self._do_connect(**kwargs)
            self._state = ConnectionState.CONNECTED
        except Exception as e:
            raise ConnectionError(f"Connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from database."""
        if self._state == ConnectionState.DISCONNECTED:
            return
        self._do_disconnect()
        self._state = ConnectionState.DISCONNECTED
    
    @contextmanager
    def connection(self, **kwargs):
        """Context manager for connection."""
        try:
            self.connect(**kwargs)
            yield self
        finally:
            self.disconnect()
    
    def _ensure_connected(self) -> None:
        if not self.is_connected:
            raise ConnectionError("Not connected")
    
    @abstractmethod
    def _do_connect(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _do_disconnect(self) -> None:
        pass

    @abstractmethod
    def insert(self, batch: List[Dict[str, Any]]) -> InsertResult:
        pass

    @abstractmethod
    def query(self, params: Dict[str, Any]) -> QueryResult:
        pass

    @abstractmethod
    def health_check(self) -> HealthStatus:
        pass

    @abstractmethod
    def delete(self, ids: List[int]) -> DeleteResult:
        pass

    def flush(self) -> None:
        """Flush buffered writes to persistent storage."""
        pass

    def load_collection(self) -> None:
        """Load a collection into memory for querying."""
        pass

    # =============================================================================
    # ADAPTER EXTENSION (Filtered ANN)
    # =============================================================================

    def create_filtered_collection(self, vector_dim: int, drop_existing: bool = True):
        """
        Create a collection schema that supports both vectors and scalar labels.
        Must be implemented by adapters supporting filtered search.
        """
        raise NotImplementedError("This adapter does not support filtered collections.")

    def insert_filtered(self, ids: List[int], vectors: List[List[float]], labels: List[int]):
        """
        Insert vectors with associated scalar labels.
        """
        raise NotImplementedError("This adapter does not support filtered insertion.")

    def search_filtered(self, query_vector: List[float], k: int, filter_expr: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a search with a scalar filter expression.
        """
        raise NotImplementedError("This adapter does not support filtered search.")

    # =============================================================================
    # ADAPTER EXTENSION (Multi-Modal)
    # =============================================================================

    def create_multi_modal_collection(
        self,
        vector_dim: int,
        mode: str = "unified",
        partitions: Optional[List[str]] = None,
        drop_existing: bool = True,
    ) -> None:
        """
        Create a collection schema for multi-modal workloads.

        Unified mode:     Single collection with [id, vector, modality_id, partition_tag].
        Partitioned mode: Collection with [id, vector] plus one native partition per name
                          in `partitions`, isolating each modality's ANN index.
        """
        raise NotImplementedError("This adapter does not support multi-modal collections or hasnt been implemented.")

    def insert_multi_modal(
        self,
        batch: List[Dict[str, Any]],
    ) -> None:
        """
        Insert a batch of multi-modal vectors in **unified** mode.

        Each item must contain:
            id            (int)  — vector ID
            vector        (list) — float embedding
            modality_id   (int)  — integer modality label
            partition_tag (str)  — modality name string

        All items are written to the global collection with modality_id and
        partition_tag as metadata fields for scalar pre-filter support.
        The adapter does no routing — caller passes a flat list.
        """
        raise NotImplementedError("This adapter does not support multi-modal insertion.")

    def insert_multi_modal_partition(
        self,
        partition_name: str,
        batch: List[Dict[str, Any]],
    ) -> None:
        """
        Insert a **pre-grouped** batch of vectors into a single named partition.

        Routing / grouping by partition_tag is the caller's responsibility
        (done in the ingest layer, not here).  The adapter just writes.

        Each item must contain:
            id     (int)  — vector ID
            vector (list) — float embedding
        """
        raise NotImplementedError("This adapter does not support partitioned insertion.")

    def search_multi_modal(
        self,
        query_vector: List[float],
        k: int,
        modality_filter: Optional[int] = None,
        partition_filter: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Single-vector ANN search with optional modality / partition scoping.

        modality_filter:  When set, apply a scalar pre-filter `modality_id == value`
                          (used for Unified+Restricted mode).
        partition_filter: When set, restrict search to the named physical partition
                          (used for Partitioned+Restricted mode).
        Both None:        Full global search (Unified+Enabled or unrestricted path).

        Returns list of dicts with at least {"id": int, "distance": float}.
        """
        raise NotImplementedError("This adapter does not support multi-modal search.")

    def batch_search_multi_modal(
        self,
        query_vectors: List[List[float]],
        k: int,
        modality_filters: Optional[List[Optional[int]]] = None,
        partition_filters: Optional[List[Optional[str]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch ANN search across all four execution matrix modes.

        One hit-list is returned per query vector; original order is preserved.

        Args:
            query_vectors:     B query embeddings.
            k:                 Top-k hits to retrieve **per query** (and per partition
                               for Partitioned+Enabled, giving an M×K candidate pool).
            modality_filters:  Length-B list of per-query modality_id filters.
                               None entry → no modality filter for that query.
                               Pass None for the whole arg to skip filtering.
                               Used by: Unified+Restricted.
            partition_filters: Length-B list of per-query partition names.
                               None entry → global search for that query.
                               A single repeated name → Partitioned+Restricted.
                               A list of ALL partition names per query →
                               Partitioned+Enabled (adapter fans out internally).
                               Pass None for the whole arg to skip partition routing.
            params:            Index search params (ef, nprobe, …).

        Returns:
            List[List[Dict]] of length B.
            Each inner list: up to k dicts {"id": int, "distance": float}.

        Efficiency contract for Milvus implementations:
            Queries sharing the same filter value are batched into a single
            MilvusClient.search call (one network round-trip per unique filter),
            then results are reassembled in original order.
        """
        raise NotImplementedError(
            "This adapter does not implement batch_search_multi_modal().\n"
            "All four execution matrix modes require this method."
        )

    def get_partition_stats(self, partition_name: str) -> Dict[str, Any]:
        """
        Return entity count and storage metadata for a named partition.
        Returns dict with at least {'entity_count': int}.
        """
        raise NotImplementedError("This adapter does not support partition stats.")

    # =============================================================================
    # ADAPTER EXTENSION (Dense Dedup / Range Search)
    # =============================================================================

    def create_dense_dedup_collection(
        self,
        vector_dim: int,
        index_type: str = "HNSW",
        metric: str = "cosine",
        index_params: Optional[Dict[str, Any]] = None,
        drop_existing: bool = True,
    ) -> None:
        """
        Create a plain float-vector collection for dense range-based deduplication.
        Schema: [id INT64 PK, vector FLOAT_VECTOR(vector_dim)]
        No binary signature field — range search is done directly on float vectors.
        Must be implemented by adapters supporting dense range search.
        """
        raise NotImplementedError("This adapter does not support dense dedup collections.")

    def insert_dense_dedup(
        self,
        ids: List[int],
        vectors: List[List[float]],
    ) -> float:
        """
        Insert raw float vectors for dense dedup (no signature field).
        Returns elapsed time in milliseconds.
        """
        raise NotImplementedError("This adapter does not support dense dedup insertion.")

    def search_range_batch(
        self,
        query_vectors: List[List[float]],
        radius: float,
        top_k: int = 10,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[List[Dict[str, Any]]], float]:
        """
        Batch range search on raw float vectors.

        For each query, return ALL stored vectors within `radius` distance
        (up to top_k results per query).

        Distance semantics depend on metric:
            L2 / Euclidean:   distance = squared Euclidean distance; lower = closer
            Cosine:           distance = 1 - cosine_similarity;      lower = closer
            Inner product:    distance = -inner_product;              lower = closer

        A vector is a duplicate if len(hits[i]) > 0.

        Returns:
            hits        : List[List[Dict[id, distance]]]  — length == len(query_vectors)
            latency_ms  : float                           — wall-clock time for the batch
        """
        raise NotImplementedError("This adapter does not support range search.")

    def search_dedup(
        self,
        query_signature: bytes,
        radius: float,
        top_k: int = 1,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Single signature search.
        Returns:
            hits        : List[Dict[id, distance]]
            latency_ms  : float
        """
        raise NotImplementedError("This adapter does not support single signature search.")

    def search_range(
        self,
        query_vector: List[float],
        radius: float,
        top_k: int = 1,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Single float vector range search.
        Returns:
            hits        : List[Dict[id, distance]]
            latency_ms  : float
        """
        raise NotImplementedError("This adapter does not support single range search.")
