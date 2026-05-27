"""
Milvus Vector DB Adapter - Connects to Milvus via pymilvus MilvusClient (PyMilvus 3.x).
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from database_adapter.base import DatabaseAdapter, QueryResult, InsertResult, HealthStatus
from database_adapter.exceptions import ConnectionError, InsertError, QueryError

try:
    from pymilvus import MilvusClient, DataType
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False


@dataclass
class MilvusConfig:
    """Milvus connection config."""
    host: str = "localhost"
    port: int = 19530
    collection: str = "default"
    user: str = ""
    password: str = ""
    timeout: float = 10.0


METRIC_MAP = {
    "cosine": "COSINE",
    "inner_product": "IP",
    "l2": "L2",
    "ip": "IP",
}

INDEX_TYPE_MAP = {
    "hnsw": "HNSW",
    "ivf_flat": "IVF_FLAT",
    "ivf_pq": "IVF_PQ",
    "ivf_sq8": "IVF_SQ8",
    "flat": "FLAT",
    "autoindex": "AUTOINDEX",
}


class MilvusAdapter(DatabaseAdapter):
    """
    Milvus adapter using MilvusClient (PyMilvus 3.x).

    Usage:
        config = MilvusConfig(host="localhost", port=19530, collection="my_collection")
        adapter = MilvusAdapter(config)
        with adapter.connection():
            adapter.create_collection(dim=128)
            adapter.create_index(index_type="HNSW", params={"M": 16}, metric="cosine")
            adapter.insert(([0, 1], [[0.1, ...], [0.2, ...]]))
            result = adapter.query({"vector": [0.1, ...], "k": 10})
    """

    def __init__(self, config: Optional[MilvusConfig] = None, name: str = "milvus"):
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Install it with: pip install pymilvus"
            )
        super().__init__(name)
        self._config = config or MilvusConfig()
        self._client: Optional[MilvusClient] = None
        self._connected: bool = False
        self._vector_dim: Optional[int] = None
        self._index_params: Optional[Dict[str, Any]] = None
        self._metric_type: str = "COSINE"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def collection_name(self) -> str:
        return self._config.collection

    def _get_num_entities(self) -> int:
        """Return live entity count for the active collection."""
        try:
            stats = self._client.get_collection_stats(self._config.collection)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def _build_search_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Construct the search_params dict from runtime params and stored index type."""
        sp: Dict[str, Any] = {"metric_type": self._metric_type, "params": {}}

        if self._index_params:
            idx_type = self._index_params.get("index_type", "")
            if idx_type == "HNSW":
                sp["params"]["ef"] = params.get("ef", params.get("ef_search", 100))
            elif idx_type in ("IVF_FLAT", "IVF_PQ", "IVF_SQ8"):
                sp["params"]["nprobe"] = params.get("nprobe", 10)

        if params.get("radius") is not None:
            sp["params"]["radius"] = params["radius"]
        if params.get("range_filter") is not None:
            sp["params"]["range_filter"] = params["range_filter"]

        return sp

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _do_connect(self, **kwargs) -> None:
        host = kwargs.get("host", self._config.host)
        port = kwargs.get("port", self._config.port)
        user = kwargs.get("user", self._config.user)
        password = kwargs.get("password", self._config.password)

        uri = f"http://{host}:{port}"
        try:
            connect_kwargs: Dict[str, Any] = {"uri": uri}
            if user:
                connect_kwargs["user"] = user
            if password:
                connect_kwargs["password"] = password

            self._client = MilvusClient(**connect_kwargs)
            self._connected = True
            print(f"[MilvusAdapter] Connected to {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus at {host}:{port}: {e}")

    def _do_disconnect(self) -> None:
        try:
            if self._client:
                self._client.close()
                self._client = None
            self._connected = False
            print("[MilvusAdapter] Disconnected")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Collection / index management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        vector_dim: int,
        description: str = "",
        drop_existing: bool = False,
    ) -> Dict[str, Any]:
        """Create a collection with id and vector fields."""
        self._ensure_connected()
        self._vector_dim = vector_dim
        collection_name = self._config.collection

        if self._client.has_collection(collection_name):
            if drop_existing:
                self._client.drop_collection(collection_name)
                print(f"[MilvusAdapter] Dropped existing collection: {collection_name}")
            else:
                print(f"[MilvusAdapter] Using existing collection: {collection_name}")
                return {"status": "exists", "collection": collection_name}

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=description or f"Collection for {collection_name}",
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)

        self._client.create_collection(collection_name, schema=schema)
        print(f"[MilvusAdapter] Created collection: {collection_name} (dim={vector_dim})")
        return {"status": "created", "collection": collection_name, "dim": vector_dim}

    def create_index(
        self,
        index_type: str = "HNSW",
        params: Optional[Dict[str, Any]] = None,
        quantization: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
    ) -> Dict[str, Any]:
        """Build a vector index and load the collection into memory."""
        self._ensure_connected()

        self._metric_type = METRIC_MAP.get(metric.lower(), "COSINE")
        milvus_index_type = INDEX_TYPE_MAP.get(index_type.lower(), index_type.upper())
        raw = params or {}

        if milvus_index_type == "HNSW":
            idx_params = {
                "M": raw.get("M", 16),
                "efConstruction": raw.get("ef_construction", 200),
            }
        elif milvus_index_type in ("IVF_FLAT", "IVF_PQ", "IVF_SQ8"):
            idx_params = {"nlist": raw.get("nlist", 1024)}
            if milvus_index_type == "IVF_PQ":
                idx_params["m"] = raw.get("m", 8)
                idx_params["nbits"] = raw.get("nbits", 8)
        elif milvus_index_type == "FLAT":
            idx_params = {}
        else:
            idx_params = raw

        # Store for search-time lookup
        self._index_params = {
            "index_type": milvus_index_type,
            "metric_type": self._metric_type,
            "params": idx_params,
        }

        index_params_obj = self._client.prepare_index_params()
        index_params_obj.add_index(
            field_name="vector",
            index_type=milvus_index_type,
            metric_type=self._metric_type,
            params=idx_params,
        )

        self._client.create_index(self._config.collection, index_params_obj)
        print(f"[MilvusAdapter] Created index: {milvus_index_type} with metric {self._metric_type}")

       
        return {
            "status": "created",
            "index_type": milvus_index_type,
            "metric": self._metric_type,
            "params": idx_params,
        }
    def load_collection(self):
        self._client.load_collection(self._config.collection)
        print("[MilvusAdapter] Collection loaded into memory")

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def insert(self, batch) -> InsertResult:
        """Insert vectors; batch is (ids_list, vectors_list)."""
        self._ensure_connected()
        start = time.perf_counter()
        try:
            ids, vectors = batch
            data = [{"id": id_, "vector": vec} for id_, vec in zip(ids, vectors)]
            self._client.insert(self._config.collection, data)
            execution_ms = (time.perf_counter() - start) * 1000
            return InsertResult(
                inserted_count=len(vectors),
                failed_count=0,
                execution_time_ms=execution_ms,
            )
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            n = len(batch[1]) if (isinstance(batch, (list, tuple)) and len(batch) > 1) else 0
            return InsertResult(inserted_count=0, failed_count=n, execution_time_ms=elapsed)

    def flush(self) -> None:
        """Flush the collection to persist buffered data."""
        self._ensure_connected()
        print("[MilvusAdapter] Flushing collection...")
        self._client.flush(self._config.collection)
        print("[MilvusAdapter] Flush complete")

    def query(self, params: Dict[str, Any]) -> QueryResult:
        """Single-vector ANN search."""
        self._ensure_connected()
        start = time.perf_counter()
        try:
            query_vector = params.get("vector", [])
            k = params.get("k", 10)
            search_params = self._build_search_params(params)

            results = self._client.search(
                collection_name=self._config.collection,
                data=[query_vector],
                anns_field="vector",
                search_params=search_params,
                limit=k,
                output_fields=["id"],
            )

            elapsed = (time.perf_counter() - start) * 1000
            data = []
            for hits in results:
                for hit in hits:
                    data.append({
                        "id": hit["id"],
                        "distance": hit["distance"],
                        "score": -hit["distance"] if self._metric_type == "L2" else hit["distance"],
                    })

            return QueryResult(
                data=data,
                total_count=self._get_num_entities(),
                execution_time_ms=elapsed,
            )
        except Exception as e:
            raise QueryError(f"Query failed: {e}")

    def query_batch(
        self, vectors, k: int, params: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Batch ANN search — one QueryResult per query vector."""
        self._ensure_connected()
        params = params or {}
        start = time.perf_counter()
        try:
            search_params = self._build_search_params(params)

            results = self._client.search(
                collection_name=self._config.collection,
                data=vectors,
                anns_field="vector",
                search_params=search_params,
                limit=k,
                output_fields=["id"],
            )

            elapsed = (time.perf_counter() - start) * 1000
            total = self._get_num_entities()
            batch_results = []
            for hits in results:
                hit_data = [
                    {
                        "id": hit["id"],
                        "distance": hit["distance"],
                        "score": -hit["distance"] if self._metric_type == "L2" else hit["distance"],
                    }
                    for hit in hits
                ]
                batch_results.append(
                    QueryResult(data=hit_data, total_count=total, execution_time_ms=elapsed)
                )
            return batch_results
        except Exception as e:
            raise QueryError(f"Batch query failed: {e}")
    
    def delete(self, ids: List[int]) -> int:
        """Delete vectors by IDs."""
        self._ensure_connected()
        try:
            self._client.delete(self._config.collection, ids=ids)
            return len(ids)
        except Exception:
            return 0


    def compact(self, timeout: float = 60.0) -> Dict[str, Any]:
        """
        Compact collection to physically remove tombstoned (deleted) vectors.
        Uses ORM Collection since MilvusClient does not expose compact().
        Blocks until compaction completes or timeout is reached.
        """
        self._ensure_connected()
        from pymilvus import Collection

        try:
            t0 = time.perf_counter()
            col = Collection(self._config.collection)
            col.compact()
            col.wait_for_compaction_completed(timeout=timeout)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            print(f"[MilvusAdapter] Compaction completed in {duration_ms:.1f}ms")
            return {"status": "completed", "duration_ms": duration_ms}

        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[MilvusAdapter] Compaction failed: {e}")
            return {"status": "failed", "duration_ms": duration_ms, "error": str(e)}

    def health_check(self) -> HealthStatus:
        """Check Milvus connection health."""
        start = time.perf_counter()
        try:
            if not self._connected or self._client is None:
                return HealthStatus(healthy=False, latency_ms=0, message="Not connected")

            collections = self._client.list_collections()
            elapsed = (time.perf_counter() - start) * 1000

            msg = f"Connected. Collections: {len(collections)}"
            cname = self._config.collection
            if cname in collections:
                msg += f", Active: {cname} ({self._get_num_entities()} entities)"

            return HealthStatus(healthy=True, latency_ms=elapsed, message=msg)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return HealthStatus(healthy=False, latency_ms=elapsed, message=str(e))

    def clear(self) -> None:
        """Drop the active collection."""
        self._ensure_connected()
        if self._client.has_collection(self._config.collection):
            self._client.drop_collection(self._config.collection)
            print(f"[MilvusAdapter] Collection {self._config.collection} dropped")

    def get_stats(self) -> Dict[str, Any]:
        """Return basic collection statistics."""
        self._ensure_connected()
        try:
            stats = self._client.get_collection_stats(self._config.collection)
            return {
                "collection": self._config.collection,
                "num_entities": int(stats.get("row_count", 0)),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Filtered workload
    # ------------------------------------------------------------------

    def create_filtered_collection(self, vector_dim: int, drop_existing: bool = True) -> None:
        """Schema: [id INT64 PK, vector FLOAT_VECTOR, label INT32]"""
        self._ensure_connected()
        collection_name = self._config.collection

        if self._client.has_collection(collection_name):
            if drop_existing:
                self._client.drop_collection(collection_name)
            else:
                return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("label", DataType.INT32)
        self._client.create_collection(collection_name, schema=schema)
        print(f"[Milvus] Created filtered collection: {collection_name}")

    def insert_filtered(
        self, ids: List[int], vectors: List[List[float]], labels: List[int]
    ) -> None:
        data = [
            {"id": id_, "vector": vec, "label": lbl}
            for id_, vec, lbl in zip(ids, vectors, labels)
        ]
        self._client.insert(self._config.collection, data)

    def search_filtered(
        self,
        query_vector: List[float],
        k: int,
        filter_expr: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        search_params = self._build_search_params(params)
        results = self._client.search(
            collection_name=self._config.collection,
            data=[query_vector],
            anns_field="vector",
            search_params=search_params,
            limit=k,
            filter=filter_expr,
            output_fields=["id"],
        )
        hits = []
        if results:
            for hit in results[0]:
                hits.append({"id": hit["id"], "distance": hit["distance"]})
        return hits

    # ------------------------------------------------------------------
    # Multi-modal workload
    # ------------------------------------------------------------------

    def create_multi_modal_collection(
        self,
        vector_dim: int,
        mode: str = "unified",
        partitions: Optional[List[str]] = None,
        drop_existing: bool = True,
    ) -> None:
        """Unified mode adds a 'modality' INT8 field; partitioned mode creates partitions."""
        self._ensure_connected()
        collection_name = self._config.collection

        if self._client.has_collection(collection_name):
            if drop_existing:
                self._client.drop_collection(collection_name)
                print(f"[Milvus] Dropped existing collection: {collection_name}")
            else:
                return

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=f"Multi-Modal ({mode})",
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        if mode == "unified":
            schema.add_field("modality", DataType.INT8)

        self._client.create_collection(collection_name, schema=schema)
        print(f"[Milvus] Created collection: {collection_name} (Mode: {mode})")

        if mode == "partitioned" and partitions:
            for part_name in partitions:
                self._client.create_partition(collection_name, part_name)
                print(f"[Milvus] Created partition: {part_name}")

    def insert_multi_modal(
        self, batch: List[Dict[str, Any]], mode: str = "unified"
    ) -> None:
        self._ensure_connected()
        if mode == "unified":
            data = [
                {"id": item["id"], "vector": item["vector"], "modality": item["modality_id"]}
                for item in batch
            ]
            self._client.insert(self._config.collection, data)
        elif mode == "partitioned":
            batches_by_part: Dict[str, List[Dict]] = defaultdict(list)
            for item in batch:
                tag = item.get("partition_tag", "_default")
                batches_by_part[tag].append({"id": item["id"], "vector": item["vector"]})
            for tag, data in batches_by_part.items():
                self._client.insert(self._config.collection, data, partition_name=tag)

    def search_multi_modal(
        self,
        query_vector: List[float],
        k: int,
        modality_filter: Optional[int] = None,
        partition_filter: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_connected()
        params = params or {}
        search_params = self._build_search_params(params)

        expr = f"modality == {modality_filter}" if modality_filter is not None else None
        partition_names = [partition_filter] if partition_filter is not None else None

        results = self._client.search(
            collection_name=self._config.collection,
            data=[query_vector],
            anns_field="vector",
            search_params=search_params,
            limit=k,
            filter=expr,
            partition_names=partition_names,
            output_fields=["id"],
        )
        hits = []
        if results:
            for hit in results[0]:
                hits.append({"id": hit["id"], "distance": hit["distance"]})
        return hits

    def get_partition_stats(self, partition_name: str) -> Dict[str, Any]:
        """Return entity count for a named partition via count(*) query."""
        self._ensure_connected()
        try:
            rows = self._client.query(
                collection_name=self._config.collection,
                filter="",
                output_fields=["count(*)"],
                partition_names=[partition_name],
            )
            entity_count = int(rows[0].get("count(*)", 0)) if rows else 0
            return {"partition": partition_name, "entity_count": entity_count}
        except Exception as e:
            return {"partition": partition_name, "entity_count": -1, "error": str(e)}

    # ------------------------------------------------------------------
    # Deduplication workload
    # ------------------------------------------------------------------

    def create_dedup_collection(
        self, vector_dim: int, signature_dim: int, drop_existing: bool = True
    ) -> None:
        """Schema: [id INT64 PK, vector FLOAT_VECTOR, signature BINARY_VECTOR]"""
        self._ensure_connected()
        collection_name = self._config.collection

        if self._client.has_collection(collection_name):
            if drop_existing:
                self._client.drop_collection(collection_name)
                print(f"[Milvus] Dropped existing collection: {collection_name}")
            else:
                return

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Deduplication Collection with Signatures",
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("signature", DataType.BINARY_VECTOR, dim=signature_dim)
        self._client.create_collection(collection_name, schema=schema)
        print(f"[Milvus] Created dedup collection: {collection_name}")

    def insert_dedup(
        self, ids: List[int], vectors: List[List[float]], signatures: List[bytes]
    ) -> None:
        data = [
            {"id": id_, "vector": vec, "signature": sig}
            for id_, vec, sig in zip(ids, vectors, signatures)
        ]
        self._client.insert(self._config.collection, data)

    def search_dedup_batch(
        self,
        query_signatures: List[bytes],
        radius: float,
        top_k: int = 1,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Batch range search on the 'signature' (BINARY_VECTOR) field using JACCARD."""
        self._ensure_connected()
        params = params or {}
        search_params: Dict[str, Any] = {
            "metric_type": "JACCARD",
            "params": {"radius": radius},
        }
        idx_type = (
            self._index_params.get("index_type", "BIN_FLAT")
            if self._index_params
            else "BIN_FLAT"
        )
        if idx_type == "BIN_IVF_FLAT":
            search_params["params"]["nprobe"] = params.get("nprobe", 10)

        results = self._client.search(
            collection_name=self._config.collection,
            data=query_signatures,
            anns_field="signature",
            search_params=search_params,
            limit=top_k,
            output_fields=["id"],
        )

        if not results:
            return [[] for _ in range(len(query_signatures))]

        return [
            [{"id": hit["id"], "distance": hit["distance"]} for hit in hits]
            for hits in results
        ]
