"""
Milvus Vector DB Adapter - Connects to Milvus via pymilvus.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from database_adapter.base import DatabaseAdapter, QueryResult, InsertResult, HealthStatus
from database_adapter.exceptions import ConnectionError, InsertError, QueryError

try:
    from pymilvus import (
        connections,
        Collection,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
    )
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


# Map metric names to Milvus metric types
METRIC_MAP = {
    "cosine": "COSINE",
    "inner_product": "IP", 
    "l2": "L2",
    "ip": "IP",
}

# Map index types to Milvus index types
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
    Milvus adapter - connects to Milvus standalone or cluster.
    
    Usage:
        config = MilvusConfig(host="localhost", port=19530, collection="my_collection")
        adapter = MilvusAdapter(config)
        with adapter.connection():
            adapter.create_index(index_type="HNSW", params={"M": 16}, metric="cosine")
            adapter.insert([{"id": 0, "vector": [0.1, 0.2, ...]}])
            result = adapter.query({"vector": [0.1, 0.2, ...], "k": 10})
    """
    
    def __init__(self, config: Optional[MilvusConfig] = None, name: str = "milvus"):
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Install it with: pip install pymilvus"
            )
        super().__init__(name)
        self._config = config or MilvusConfig()
        self._collection: Optional[Collection] = None
        self._connection_alias = f"milvus_{id(self)}"
        self._vector_dim: Optional[int] = None
        self._index_params: Optional[Dict[str, Any]] = None
        self._metric_type: str = "COSINE"
    
    @property
    def collection_name(self) -> str:
        return self._config.collection
    
    def _do_connect(self, **kwargs) -> None:
        """Connect to Milvus server."""
        host = kwargs.get("host", self._config.host)
        port = kwargs.get("port", self._config.port)
        user = kwargs.get("user", self._config.user)
        password = kwargs.get("password", self._config.password)
        timeout = kwargs.get("timeout", self._config.timeout)
        
        try:
            connections.connect(
                alias=self._connection_alias,
                host=host,
                port=str(port),
                user=user if user else None,
                password=password if password else None,
                timeout=timeout,
            )
            print(f"[MilvusAdapter] Connected to {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus at {host}:{port}: {e}")
    
    def _do_disconnect(self) -> None:
        """Disconnect from Milvus."""
        try:
            if self._collection:
                self._collection.release()
                self._collection = None
            connections.disconnect(self._connection_alias)
            print(f"[MilvusAdapter] Disconnected")
        except Exception:
            pass
    
    def create_collection(
        self,
        vector_dim: int,
        description: str = "",
        drop_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Create a collection with id and vector fields.
        
        Args:
            vector_dim: Dimension of embedding vectors
            description: Collection description
            drop_existing: Drop collection if it exists
        """
        self._ensure_connected()
        self._vector_dim = vector_dim
        
        collection_name = self._config.collection
        
        # Check if collection exists
        if utility.has_collection(collection_name, using=self._connection_alias):
            if drop_existing:
                utility.drop_collection(collection_name, using=self._connection_alias)
                print(f"[MilvusAdapter] Dropped existing collection: {collection_name}")
            else:
                self._collection = Collection(
                    name=collection_name,
                    using=self._connection_alias
                )
                print(f"[MilvusAdapter] Using existing collection: {collection_name}")
                return {"status": "exists", "collection": collection_name}
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]
        schema = CollectionSchema(
            fields=fields,
            description=description or f"Collection for {collection_name}"
        )
        
        # Create collection
        self._collection = Collection(
            name=collection_name,
            schema=schema,
            using=self._connection_alias
        )
        print(f"[MilvusAdapter] Created collection: {collection_name} (dim={vector_dim})")
        
        return {"status": "created", "collection": collection_name, "dim": vector_dim}
    
    def create_index(
        self,
        index_type: str = "HNSW",
        params: Optional[Dict[str, Any]] = None,
        quantization: Optional[Dict[str, Any]] = None,
        metric: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Create vector index on the collection.
        
        Args:
            index_type: HNSW, IVF_FLAT, IVF_PQ, FLAT, etc.
            params: Index-specific params (M, ef_construction for HNSW; nlist for IVF)
            quantization: Quantization config (not used directly, but passed through params)
            metric: cosine, l2, inner_product
        """
        self._ensure_connected()
        
        if not self._collection:
            raise ConnectionError("No collection loaded. Call create_collection first.")
        
        # Map metric type
        self._metric_type = METRIC_MAP.get(metric.lower(), "COSINE")
        
        # Map index type  
        milvus_index_type = INDEX_TYPE_MAP.get(index_type.lower(), index_type.upper())
        
        # Build index params based on type
        index_params = params or {}
        
        if milvus_index_type == "HNSW":
            index_config = {
                "index_type": "HNSW",
                "metric_type": self._metric_type,
                "params": {
                    "M": index_params.get("M", 16),
                    "efConstruction": index_params.get("ef_construction", 200),
                }
            }
        elif milvus_index_type in ["IVF_FLAT", "IVF_PQ", "IVF_SQ8"]:
            index_config = {
                "index_type": milvus_index_type,
                "metric_type": self._metric_type,
                "params": {
                    "nlist": index_params.get("nlist", 1024),
                }
            }
            if milvus_index_type == "IVF_PQ":
                index_config["params"]["m"] = index_params.get("m", 8)
                index_config["params"]["nbits"] = index_params.get("nbits", 8)
        elif milvus_index_type == "FLAT":
            index_config = {
                "index_type": "FLAT",
                "metric_type": self._metric_type,
                "params": {}
            }
        else:
            # Generic/AutoIndex
            index_config = {
                "index_type": milvus_index_type,
                "metric_type": self._metric_type,
                "params": index_params
            }
        
        self._index_params = index_config
        
        # Create the index
        self._collection.create_index(
            field_name="vector",
            index_params=index_config
        )
        print(f"[MilvusAdapter] Created index: {milvus_index_type} with metric {self._metric_type}")
        
        # Load collection into memory for searching
        self._collection.load()
        print(f"[MilvusAdapter] Collection loaded into memory")
        
        return {
            "status": "created",
            "index_type": milvus_index_type,
            "metric": self._metric_type,
            "params": index_config["params"]
        }
    
    def insert(self, batch: List[Dict[str, Any]]) -> InsertResult:
        """Insert vectors into the collection."""
        self._ensure_connected()
        
        if not self._collection:
            raise InsertError("No collection loaded. Call create_collection first.")
        
        start = time.perf_counter()
        
        try:
            ids = [record["id"] for record in batch]
            vectors = [record["vector"] for record in batch]
            
            self._collection.insert([ids, vectors])
            # Note: flush removed to avoid rate limiting and for speed - flush called at end of ingestion
            
            elapsed = (time.perf_counter() - start) * 1000
            
            return InsertResult(
                inserted_count=len(batch),
                failed_count=0,
                execution_time_ms=elapsed
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            raise InsertError(f"Insert failed: {e}")
    
    def flush(self) -> None:
        """Flush collection to persist data."""
        self._ensure_connected()
        if self._collection:
            print("[MilvusAdapter] Flushing collection...")
            self._collection.flush()
            print("[MilvusAdapter] Flush complete")
    
    def query(self, params: Dict[str, Any]) -> QueryResult:
        """
        Query vectors.
        
        Args:
            params: Dict with 'vector' (query vector), 'k' (top-k), optional 'ef' (search param)
        """
        self._ensure_connected()
        
        if not self._collection:
            raise QueryError("No collection loaded. Call create_collection first.")
        
        start = time.perf_counter()
        
        try:
            query_vector = params.get("vector", [])
            k = params.get("k", 10)
            
            # Build search params
            search_params = {"metric_type": self._metric_type}
            
            # Add index-specific search params
            if self._index_params:
                idx_type = self._index_params.get("index_type", "")
                if idx_type == "HNSW":
                    actual_ef = params.get("ef", params.get("ef_search", 100))
                    search_params["params"] = {"ef":actual_ef}
                elif idx_type in ["IVF_FLAT", "IVF_PQ", "IVF_SQ8"]:
                    search_params["params"] = {"nprobe": params.get("nprobe", 10)}
                else:
                    search_params["params"] = {}
            else:
                search_params["params"] = {}
                
            # Add range search parameters if provided (for outlier workload)
            if "radius" in params and params["radius"] is not None:
                search_params["params"]["radius"] = params["radius"]
            if "range_filter" in params and params["range_filter"] is not None:
                search_params["params"]["range_filter"] = params["range_filter"]
            
            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["id"]
            )
            
            elapsed = (time.perf_counter() - start) * 1000
            
            # Format results
            data = []
            for hits in results:
                for hit in hits:
                    data.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": -hit.distance if self._metric_type == "L2" else hit.distance
                    })
            
            return QueryResult(
                data=data,
                total_count=self._collection.num_entities,
                execution_time_ms=elapsed
            )
        except Exception as e:
            raise QueryError(f"Query failed: {e}")
    
    def health_check(self) -> HealthStatus:
        """Check Milvus connection health."""
        start = time.perf_counter()
        
        try:
            # Check if connected
            if not connections.has_connection(self._connection_alias):
                return HealthStatus(
                    healthy=False,
                    latency_ms=0,
                    message="Not connected"
                )
            
            # Try to list collections as health check
            collections = utility.list_collections(using=self._connection_alias)
            elapsed = (time.perf_counter() - start) * 1000
            
            msg = f"Connected. Collections: {len(collections)}"
            if self._collection:
                msg += f", Active: {self._config.collection} ({self._collection.num_entities} entities)"
            
            return HealthStatus(
                healthy=True,
                latency_ms=elapsed,
                message=msg
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return HealthStatus(
                healthy=False,
                latency_ms=elapsed,
                message=str(e)
            )
    
    def clear(self) -> None:
        """Drop and recreate the collection."""
        self._ensure_connected()
        
        if utility.has_collection(self._config.collection, using=self._connection_alias):
            utility.drop_collection(self._config.collection, using=self._connection_alias)
            print(f"[MilvusAdapter] Collection {self._config.collection} dropped")
        
        self._collection = None
    
    def delete(self, ids: List[int]) -> int:
        """Delete vectors by IDs."""
        self._ensure_connected()
        
        if not self._collection:
            return 0
        
        try:
            expr = f"id in {ids}"
            self._collection.delete(expr)
            return len(ids)
        except Exception:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        self._ensure_connected()
        
        if not self._collection:
            return {}
        
        return {
            "collection": self._config.collection,
            "num_entities": self._collection.num_entities,
            "schema": str(self._collection.schema),
        }
    
    # =========================================================
    # Filtered Workload Implementation
    # =========================================================

    def create_filtered_collection(self, vector_dim: int, drop_existing: bool = True):
        """
        Creates a collection with schema: [id: INT64, vector: FLOAT_VECTOR, label: INT32]
        """
        collection_name = getattr(self._config,"collection", "default")
        
        if utility.has_collection(collection_name, using=self._connection_alias):
            if drop_existing:
                utility.drop_collection(collection_name, using=self._connection_alias)
            else:
                self._collection = Collection(collection_name, using=self._connection_alias)
                return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="label", dtype=DataType.INT32)  # The filtering field
        ]
        
        schema = CollectionSchema(fields=fields, description="Benchmarking with Filters")
        self._collection = Collection(name=collection_name, schema=schema, using=self._connection_alias)
        print(f"[Milvus] Created filtered collection: {collection_name}")

    def insert_filtered(self, ids: List[int], vectors: List[List[float]], labels: List[int]):
        """Batch insert with labels."""
        if not self._collection:
            raise RuntimeError("Collection not initialized.")
        
        # Milvus expects columnar data: [ids_list, vectors_list, labels_list]
        self._collection.insert([ids, vectors, labels])

    def search_filtered(self, query_vector: List[float], k: int, filter_expr: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search with `expr`.
        Params:
            filter_expr: e.g., "label in [1, 2]"
        """
        if not self._collection:
            raise RuntimeError("Collection not initialized.")
        
        # Translate generic params to Milvus specific
        search_params = {"metric_type": self._metric_type}
        
        # Handle HNSW/IVF specifics
        idx_type = self._index_params.get("index_type", "HNSW")
        if idx_type == "HNSW":
            search_params["params"] = {"ef": params.get("ef_search", 100)}
        elif idx_type in ["IVF_FLAT", "IVF_PQ"]:
            search_params["params"] = {"nprobe": params.get("nprobe", 10)}
        
        results = self._collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=k,
            expr=filter_expr,  
            output_fields=["id"]
        )

        hits = []
        if results:
            for hit in results[0]:
                hits.append({"id": hit.id, "distance": hit.distance})
        return hits

    # =========================================================
    # Multi-Modal Implementation
    # =========================================================

    def create_multi_modal_collection(self, vector_dim: int, mode: str = "unified", partitions: List[str] = None, drop_existing: bool = True):
        """
        Create collection for multi-modal workload.
        Unified: Adds 'modality' (INT8) field.
        Partitioned: Creates standard collection + partitions.
        """
        self._ensure_connected()
        collection_name = self._config.collection
        
        # Dropping existing
        if utility.has_collection(collection_name, using=self._connection_alias):
            if drop_existing:
                utility.drop_collection(collection_name, using=self._connection_alias)
                print(f"[Milvus] Dropped existing collection: {collection_name}")
            else:
                self._collection = Collection(collection_name, using=self._connection_alias)
        
        if not self._collection:
            # Define Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            ]
            
            if mode == "unified":
                # Unified mode uses a scalar field for modality
                fields.append(FieldSchema(name="modality", dtype=DataType.INT8))
            
            schema = CollectionSchema(fields=fields, description=f"Multi-Modal ({mode})")
            self._collection = Collection(name=collection_name, schema=schema, using=self._connection_alias)
            print(f"[Milvus] Created collection: {collection_name} (Mode: {mode})")

        # Create Partitions if needed
        if mode == "partitioned" and partitions:
            for part_name in partitions:
                if not self._collection.has_partition(part_name):
                    self._collection.create_partition(part_name)
                    print(f"[Milvus] Created partition: {part_name}")

    def insert_multi_modal(self, batch: List[Dict[str, Any]], mode: str = "unified"):
        """
        Insert batch with modality info.
        Batch items: {'id': int, 'vector': list, 'modality_id': int, 'partition_tag': str}
        """
        self._ensure_connected()
        if not self._collection: raise InsertError("Collection not initialized")

        ids = [item['id'] for item in batch]
        vectors = [item['vector'] for item in batch]
        
        if mode == "unified":
            modalities = [item['modality_id'] for item in batch]
            self._collection.insert([ids, vectors, modalities])
        
        elif mode == "partitioned":
            # Group by partition tag
            from collections import defaultdict
            batches_by_part = defaultdict(lambda: {"ids": [], "vectors": []})
            
            for item in batch:
                tag = item.get('partition_tag', '_default')
                batches_by_part[tag]["ids"].append(item['id'])
                batches_by_part[tag]["vectors"].append(item['vector'])
            
            for tag, data in batches_by_part.items():
                self._collection.insert(
                    [data["ids"], data["vectors"]],
                    partition_name=tag
                )

    def search_multi_modal(self, query_vector: List[float], k: int, 
                          modality_filter: Optional[int] = None, 
                          partition_filter: Optional[str] = None,
                          params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search with optional modality/partition filtering.
        """
        self._ensure_connected()
        search_params = {"metric_type": self._metric_type}
        
        # Build index params
        idx_type = self._index_params.get("index_type", "HNSW") if self._index_params else "HNSW"
        if idx_type == "HNSW":
            search_params["params"] = {"ef": params.get("ef_search", 100)}
        elif idx_type in ["IVF_FLAT", "IVF_PQ"]:
             search_params["params"] = {"nprobe": params.get("nprobe", 10)}
        
        # Configure filters
        expr = None
        partition_names = None
        
        if modality_filter is not None:
             expr = f"modality == {modality_filter}"
            
        if partition_filter is not None:
            partition_names = [partition_filter]

        results = self._collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=k,
            expr=expr,
            partition_names=partition_names,
            output_fields=["id"]
        )

        hits = []
        if results:
            for hit in results[0]:
                hits.append({"id": hit.id, "distance": hit.distance})
        return hits

    def get_partition_stats(self, partition_name: str) -> Dict[str, Any]:
        """
        Return entity count for a named Milvus partition.

        Uses collection.num_entities filtered to partition.
        Falls back to a count query if partition_name is not '_default'.
        """
        self._ensure_connected()
        if not self._collection:
            return {"partition": partition_name, "entity_count": 0}

        try:
            # Milvus Partition.num_entities is the authoritative source
            if self._collection.has_partition(partition_name):
                part = self._collection.partition(partition_name)
                part.load()
                entity_count = part.num_entities
            else:
                entity_count = 0

            return {
                "partition":    partition_name,
                "entity_count": entity_count,
            }
        except Exception as e:
            return {"partition": partition_name, "entity_count": -1, "error": str(e)}

    # =========================================================
    # Deduplication Workload Implementation
    # =========================================================

    def create_dedup_collection(self, vector_dim: int, signature_dim: int, drop_existing: bool = True):
        """
        Create collection for deduplication workload with schema: [id, vector, signature].
        signature_dim is the number of bits for the BINARY_VECTOR.
        """
        self._ensure_connected()
        collection_name = self._config.collection
        
        if utility.has_collection(collection_name, using=self._connection_alias):
            if drop_existing:
                utility.drop_collection(collection_name, using=self._connection_alias)
                print(f"[Milvus] Dropped existing collection: {collection_name}")
            else:
                self._collection = Collection(collection_name, using=self._connection_alias)
                return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="signature", dtype=DataType.BINARY_VECTOR, dim=signature_dim)
        ]
        
        schema = CollectionSchema(fields=fields, description="Deduplication Collection with Signatures")
        self._collection = Collection(name=collection_name, schema=schema, using=self._connection_alias)
        print(f"[Milvus] Created dedup collection: {collection_name}")

    def insert_dedup(self, ids: List[int], vectors: List[List[float]], signatures: List[bytes]):
        """Batch insert into dedup collection."""
        if not self._collection:
            raise RuntimeError("Collection not initialized.")
        self._collection.insert([ids, vectors, signatures])

    def search_dedup_batch(self, query_signatures: List[bytes], radius: float, top_k: int = 1, params: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
        """
        Batch range search on the 'signature' field.
        Returns a list of hit lists, one per query signature.
        """
        if not self._collection:
            raise RuntimeError("Collection not initialized.")
            
        params = params or {}
        
        search_params = {
            "metric_type": "JACCARD",
            "params": {"radius": radius}
        }
        
        idx_type = self._index_params.get("index_type", "BIN_FLAT") if getattr(self, "_index_params", None) else "BIN_FLAT"
        if idx_type in ["BIN_IVF_FLAT"]:
            search_params["params"]["nprobe"] = params.get("nprobe", 10)
        
        results = self._collection.search(
            data=query_signatures,
            anns_field="signature",
            param=search_params,
            limit=top_k,
            output_fields=["id"]
        )

        batch_hits = []
        if results:
            for hits in results:
                local_hits = []
                for hit in hits:
                    local_hits.append({"id": hit.id, "distance": hit.distance})
                batch_hits.append(local_hits)
        else:
            # If no results returned at all (shouldn't happen with valid queries)
            batch_hits = [[] for _ in range(len(query_signatures))]
            
        return batch_hits


    
