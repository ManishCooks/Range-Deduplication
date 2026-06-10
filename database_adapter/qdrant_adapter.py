"""
Qdrant Vector DB Adapter - Connects to Qdrant via qdrant-client.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from database_adapter.base import DatabaseAdapter, QueryResult, InsertResult, DeleteResult, HealthStatus
from database_adapter.exceptions import ConnectionError, InsertError, QueryError

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.models import QueryRequest
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@dataclass
class QdrantConfig:
    """Qdrant connection config."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    collection: str = "default"
    api_key: Optional[str] = None
    timeout: float = 10.0
    prefer_grpc: bool = True


METRIC_MAP = {
    "cosine": "Cosine",
    "euclidean": "Euclid",
    "l2": "Euclid",
    "dotproduct": "Dot",
    "inner_product": "Dot",
    "ip": "Dot",
    "jaccard": "Jaccard",
}


class QdrantAdapter(DatabaseAdapter):
    """
    Qdrant adapter using qdrant-client.
    """

    def __init__(self, config: Optional[QdrantConfig] = None, name: str = "qdrant"):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Install it with: pip install qdrant-client"
            )
        super().__init__(name)
        self._config = config or QdrantConfig()
        self._client: Optional[QdrantClient] = None
        self._connected: bool = False
        self._vector_dim: Optional[int] = None
        self._metric: str = "Cosine"
        self._mm_mode: str = "unified"
        self._mm_partitions: List[str] = []

    @property
    def collection_name(self) -> str:
        return self._config.collection

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _do_connect(self, **kwargs) -> None:
        host = kwargs.get("host", self._config.host)
        port = kwargs.get("port", self._config.port)
        grpc_port = kwargs.get("grpc_port", self._config.grpc_port)
        api_key = kwargs.get("api_key", self._config.api_key)
        prefer_grpc = kwargs.get("prefer_grpc", self._config.prefer_grpc)

        try:
            self._client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                api_key=api_key,
                timeout=600.0,
            )
            import time
            for attempt in range(15):
                try:
                    self._client.get_collections()
                    break
                except Exception as e:
                    print(f"[QdrantAdapter] Waiting for server... ({e})")
                    time.sleep(2)
            else:
                self._client.get_collections()
            self._connected = True
            print(f"[QdrantAdapter] Connected to {host}:{port} (gRPC={prefer_grpc})")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant at {host}:{port}: {e}")

    def _do_disconnect(self) -> None:
        try:
            if self._client:
                self._client.close()
                self._client = None
            self._connected = False
            print("[QdrantAdapter] Disconnected")
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
        """Create a default unnamed vector collection in Qdrant."""
        self._ensure_connected()
        self._vector_dim = vector_dim
        collection_name = self._config.collection

        if self._client.collection_exists(collection_name):
            if drop_existing:
                self._client.delete_collection(collection_name)
                print(f"[QdrantAdapter] Dropped existing collection: {collection_name}")
            else:
                print(f"[QdrantAdapter] Using existing collection: {collection_name}")
                return {"status": "exists", "collection": collection_name}

        # create index parameters later when create_index is called, 
        # or just create collection with default HNSW.
        # To match Milvus pipeline, we defer full setup to create_index or do it here.
        # But Qdrant requires vector size at collection creation.
        
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_dim,
                distance=rest.Distance.COSINE
            )
        )
        print(f"[QdrantAdapter] Created collection: {collection_name} (dim={vector_dim})")
        return {"status": "created", "collection": collection_name, "dim": vector_dim}

    def create_index(
        self,
        index_type: str = "HNSW",
        params: Optional[Dict[str, Any]] = None,
        quantization: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Qdrant only uses HNSW (with optional quantization). 
        We use this to update collection parameters if needed.
        """
        self._ensure_connected()
        self._metric = METRIC_MAP.get(metric.lower(), "Cosine")
        
        # If we need to update the HNSW parameters, we use update_collection
        raw = params or {}
        hnsw_config = rest.HnswConfigDiff(
            m=raw.get("M"),
            ef_construct=raw.get("ef_construction")
        )
        
        self._client.update_collection(
            collection_name=self._config.collection,
            hnsw_config=hnsw_config
        )
        print(f"[QdrantAdapter] Updated HNSW index parameters")

        return {
            "status": "created",
            "index_type": "HNSW",
            "metric": self._metric,
            "params": raw,
        }
        
    def load_collection(self):
        # Qdrant loads collections automatically.
        print("[QdrantAdapter] Collection ready in memory")

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def insert(self, batch) -> InsertResult:
        """Insert vectors; batch is (ids_list, vectors_list)."""
        self._ensure_connected()
        try:
            ids, vectors = batch
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            else:
                vectors = [v.tolist() if hasattr(v, "tolist") else v for v in vectors]
                
            points = [
                rest.PointStruct(id=int(id_), vector=vec)
                for id_, vec in zip(ids, vectors)
            ]
            start = time.perf_counter()
            self._client.upsert(
                collection_name=self._config.collection,
                points=points
            )
            execution_ms = (time.perf_counter() - start) * 1000
            return InsertResult(
                inserted_count=len(vectors),
                failed_count=0,
                execution_time_ms=execution_ms,
            )
        except Exception as e:
            n = len(batch[1]) if (isinstance(batch, (list, tuple)) and len(batch) > 1) else 0
            print(f"[QdrantAdapter] Insert error: {e}")
            return InsertResult(inserted_count=0, failed_count=n, execution_time_ms=0.0)

    def flush(self) -> None:
        """Flush the collection to persist buffered data."""
        pass # Qdrant doesn't have an explicit flush method

    def query(self, params: Dict[str, Any]) -> QueryResult:
        self._ensure_connected()

        try:
            query_vector = params["vector"]
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
            k = params.get("k", 10)

            search_params = None
            if params.get("ef") is not None:
                search_params = rest.SearchParams(
                    hnsw_ef=params["ef"]
                )

            start = time.perf_counter()
            results = self._client.query_points(
                collection_name=self._config.collection,
                query=query_vector,
                limit=k,
                search_params=search_params,
                with_payload=False,
            )

            elapsed = (time.perf_counter() - start) * 1000

            data = [
                {
                    "id": hit.id,
                    "score": hit.score,
                }
                for hit in results.points
            ]

            return QueryResult(
                data=data,
                total_count=self.get_stats().get(
                    "num_entities",
                    0,
                ),
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
        try:
            search_params = None

            if params.get("ef") is not None:
                search_params = rest.SearchParams(
                    hnsw_ef=params["ef"]
                )

            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            else:
                vectors = [v.tolist() if hasattr(v, "tolist") else v for v in vectors]

            requests = [
                rest.QueryRequest(        
                    query=vec,       
                    limit=k,
                    params=search_params,
                    with_payload=False
                )
                for vec in vectors
            ]
        
            start = time.perf_counter()
            results = self._client.query_batch_points(
                collection_name=self._config.collection,
                requests=requests,
            )

            elapsed = (time.perf_counter() - start) * 1000
            total = self.get_stats().get("num_entities", 0)
            batch_results = [
                QueryResult(
                    data=[
                        {
                            "id": hit.id,
                            "distance": hit.score,  
                        }
                        for hit in response.points
                    ],
                    total_count=total,
                    execution_time_ms=elapsed
                )
                for response in results
            ]
            return batch_results

        except Exception as e:
            raise QueryError(f"Batch query failed: {e}")

    def delete(self, ids: List[int]) -> DeleteResult:
        """Delete vectors by IDs."""
        self._ensure_connected()
        try:
            start = time.perf_counter()
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            else:
                ids = [int(i) for i in ids]
                
            self._client.delete(
                collection_name=self._config.collection,
                points_selector=rest.PointIdsList(points=ids)
            )
            elapsed = (time.perf_counter() - start) * 1000
            return DeleteResult(deleted_count=len(ids), execution_time_ms=elapsed)
        except Exception:
            print(f"[QdrantAdapter] Delete failed")
            return DeleteResult(deleted_count=0, execution_time_ms=0.0)

    def health_check(self) -> HealthStatus:
        """Check Qdrant connection health."""
        start = time.perf_counter()
        try:
            if not self._connected or self._client is None:
                return HealthStatus(healthy=False, latency_ms=0, message="Not connected")

            cols = self._client.get_collections().collections
            elapsed = (time.perf_counter() - start) * 1000

            msg = f"Connected. Collections: {len(cols)}"
            cname = self._config.collection
            if any(c.name == cname for c in cols):
                msg += f", Active: {cname}"

            return HealthStatus(healthy=True, latency_ms=elapsed, message=msg)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return HealthStatus(healthy=False, latency_ms=elapsed, message=str(e))

    def clear(self) -> None:
        """Drop the active collection."""
        self._ensure_connected()
        if self._client.collection_exists(self._config.collection):
            self._client.delete_collection(self._config.collection)
            print(f"[QdrantAdapter] Collection {self._config.collection} dropped")

    def get_stats(self) -> Dict[str, Any]:
        """Return basic collection statistics."""
        self._ensure_connected()
        try:
            stats = self._client.get_collection(self._config.collection)
            return {
                "collection": self._config.collection,
                "num_entities": stats.points_count,
            }
        except Exception as e:
            print(f"[QdrantAdapter] Failed to get collection stats: {e}")
            return {}

    # ------------------------------------------------------------------
    # Filtered workload
    # ------------------------------------------------------------------

    def create_filtered_collection(self, vector_dim: int, drop_existing: bool = True) -> None:
        self._ensure_connected()
        collection_name = self._config.collection

        if self._client.collection_exists(collection_name):
            if drop_existing:
                self._client.delete_collection(collection_name)
            else:
                return

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_dim,
                distance=rest.Distance.COSINE
            )
        )
        self._client.create_payload_index(
            collection_name=collection_name,
            field_name="label",
            field_schema=rest.PayloadSchemaType.INTEGER,
        )
        print(f"[QdrantAdapter] Created filtered collection: {collection_name}")

    def insert_filtered(
        self, ids: List[int], vectors: List[List[float]], labels: List[int]
    ) -> None:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        else:
            vectors = [v.tolist() if hasattr(v, "tolist") else v for v in vectors]
            
        points = [
            rest.PointStruct(id=int(id_), vector=vec, payload={"label": lbl})
            for id_, vec, lbl in zip(ids, vectors, labels)
        ]
        self._client.upsert(self._config.collection, points=points)

    def search_filtered(
        self,
        query_vector: List[float],
        k: int,
        filter_expr: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Map simple equality filter `label == x` to Qdrant Filter
        # Milvus sends things like `label == 1`
        # We need to parse simple expressions for Qdrant
        import re
        match = re.match(r"label\s*==\s*(\d+)", filter_expr)
        qdrant_filter = None
        if match:
            val = int(match.group(1))
            qdrant_filter = rest.Filter(
                must=[rest.FieldCondition(key="label", match=rest.MatchValue(value=val))]
            )
            
        search_params = None
        if params.get("ef") is not None:
            search_params = rest.SearchParams(hnsw_ef=params["ef"])

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        results = self._client.search(
            collection_name=self._config.collection,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            search_params=search_params,
            with_payload=False,
        )
        return [{"id": hit.id, "distance": hit.score} for hit in results]

    # ------------------------------------------------------------------
    # Multi-modal workload 
    # ------------------------------------------------------------------
    
    def create_multi_modal_collection(
        self,
        vector_dim: Optional[int] = None,
        mode: str = "unified",
        partitions: Optional[List[str]] = None,
        drop_existing: bool = True,
    ) -> None:
        self._ensure_connected()
        collection_name = self._config.collection

        if self._client.collection_exists(collection_name):
            if drop_existing:
                self._client.delete_collection(collection_name)
                print(f"[QdrantAdapter] Dropped existing collection: {collection_name}")
            else:
                print(f"[QdrantAdapter] Using existing collection: {collection_name}")
                return

        self._mm_mode = mode
        self._vector_dim = vector_dim
        
        if mode == "partitioned" and partitions:
            self._mm_partitions = partitions
            vectors_config = {
                part_name: rest.VectorParams(size=vector_dim, distance=rest.Distance.COSINE)
                for part_name in partitions
            }
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config
            )
            print(f"[QdrantAdapter] Created collection with named vectors for partitions: {partitions}")
        else:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=vector_dim, distance=rest.Distance.COSINE)
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="modality_id",
                field_schema=rest.PayloadSchemaType.INTEGER,
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="partition_tag",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            print(f"[QdrantAdapter] Created unified collection with payload indices")

    def insert_multi_modal(
        self,
        batch: List[Dict[str, Any]],
        mode: Optional[str] = None, 
    ) -> None:
        self._ensure_connected()
        points = [
            rest.PointStruct(
                id=int(item["id"]),
                vector=item["vector"].tolist() if hasattr(item["vector"], "tolist") else item["vector"],
                payload={
                    "modality_id": int(item.get("modality_id", 0)),
                    "partition_tag": str(item.get("partition_tag", "")),
                }
            )
            for item in batch
        ]
        self._client.upsert(self._config.collection, points=points)

    def insert_multi_modal_partition(
        self,
        partition_name: str,
        batch: List[Dict[str, Any]],
    ) -> None:
        self._ensure_connected()
        points = [
            rest.PointStruct(
                id=int(item["id"]),
                vector={partition_name: item["vector"].tolist() if hasattr(item["vector"], "tolist") else item["vector"]},
            )
            for item in batch
        ]
        self._client.upsert(self._config.collection, points=points)

    def search_multi_modal(
        self,
        query_vector: List[float],
        k: int,
        modality_filter: Optional[int] = None,
        partition_filter: Optional[str] = None, 
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_connected()
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        params = params or {}
        search_params = None
        if params.get("ef") is not None:
            search_params = rest.SearchParams(hnsw_ef=params["ef"])

        query_filter = None
        vector_name = None
        
        if self._mm_mode == "partitioned" and partition_filter:
            vector_name = partition_filter
        else:
            must_conds = []
            if modality_filter is not None:
                must_conds.append(
                    rest.FieldCondition(key="modality_id", match=rest.MatchValue(value=modality_filter))
                )
            if partition_filter is not None:
                must_conds.append(
                    rest.FieldCondition(key="partition_tag", match=rest.MatchValue(value=partition_filter))
                )
            if must_conds:
                query_filter = rest.Filter(must=must_conds)

        results = self._client.search(
            collection_name=self._config.collection,
            query_vector=(vector_name, query_vector) if vector_name else query_vector,
            limit=k,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=False,
        )

        return [{"id": hit.id, "distance": hit.score} for hit in results]

    def batch_search_multi_modal(
        self,
        query_vectors: List[List[float]],
        k: int,
        modality_filters: Optional[List[Optional[int]]] = None,
        partition_filters: Optional[List[Optional[str]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        self._ensure_connected()
        params = params or {}
        search_params = None
        if params.get("ef") is not None:
            search_params = rest.SearchParams(hnsw_ef=params["ef"])

        size = len(query_vectors)
        requests = []

        for i in range(size):
            m_filt = modality_filters[i] if modality_filters else None
            p_filt = partition_filters[i] if partition_filters else None
            
            # Note: For Partitioned+Enabled mode, the orchestrator passes a LIST of all partition names per query.
            # We must map one query vector to multiple search requests in Qdrant (one per named vector).
            # To preserve original response structure, we will handle this explicitly.

            if isinstance(p_filt, list) and self._mm_mode == "partitioned":
                # For Partitioned+Enabled: one request per partition
                for part in p_filt:
                    req = rest.SearchRequest(
                        vector=rest.NamedVector(name=part, vector=query_vectors[i].tolist() if hasattr(query_vectors[i], "tolist") else query_vectors[i]),
                        limit=k,
                        params=search_params,
                        with_payload=False,
                    )
                    requests.append((i, req))
            else:
                vector_name = None
                query_filter = None
                
                if self._mm_mode == "partitioned" and isinstance(p_filt, str):
                    vector_name = p_filt
                else:
                    must_conds = []
                    if m_filt is not None:
                        must_conds.append(
                            rest.FieldCondition(key="modality_id", match=rest.MatchValue(value=m_filt))
                        )
                    if isinstance(p_filt, str):
                        must_conds.append(
                            rest.FieldCondition(key="partition_tag", match=rest.MatchValue(value=p_filt))
                        )
                    if must_conds:
                        query_filter = rest.Filter(must=must_conds)

                qv = query_vectors[i].tolist() if hasattr(query_vectors[i], "tolist") else query_vectors[i]
                req = rest.SearchRequest(
                    vector=rest.NamedVector(name=vector_name, vector=qv) if vector_name else qv,
                    limit=k,
                    filter=query_filter,
                    params=search_params,
                    with_payload=False,
                )
                requests.append((i, req))

        # Send batch request
        # Qdrant client.search_batch() expects List[SearchRequest]
        raw_requests = [r[1] for r in requests]
        
        batch_results = []
        # Chunk to avoid excessively large batch requests if size is huge (e.g. 1000+)
        CHUNK_SIZE = 1000
        for i in range(0, len(raw_requests), CHUNK_SIZE):
            chunk = raw_requests[i:i+CHUNK_SIZE]
            res = self._client.search_batch(
                collection_name=self._config.collection,
                requests=chunk
            )
            batch_results.extend(res)

        # Reassemble
        final_results = [[] for _ in range(size)]
        
        for (query_idx, _), hits in zip(requests, batch_results):
            mapped_hits = [{"id": hit.id, "distance": hit.score} for hit in hits]
            final_results[query_idx].extend(mapped_hits)

        # For Partitioned+Enabled, sort by distance since we merged hits from multiple partitions
        if self._mm_mode == "partitioned" and partition_filters and isinstance(partition_filters[0], list):
            for i in range(size):
                final_results[i].sort(key=lambda h: h["distance"])

        return final_results

    def get_partition_stats(self, partition_name: str) -> Dict[str, Any]:
        """Return entity count for a named partition."""
        self._ensure_connected()
        # Qdrant does not provide per-named-vector count trivially without a filter if they share the same collection.
        # But we can just return the total count since points might have multiple vectors.
        # If we use payloads for partition_tag, we can do a count request.
        try:
            if self._mm_mode == "unified":
                cnt = self._client.count(
                    collection_name=self._config.collection,
                    count_filter=rest.Filter(
                        must=[rest.FieldCondition(key="partition_tag", match=rest.MatchValue(value=partition_name))]
                    )
                )
                return {"partition": partition_name, "entity_count": cnt.count}
            else:
                stats = self._client.get_collection(self._config.collection)
                return {"partition": partition_name, "entity_count": stats.points_count}
        except Exception as e:
            return {"partition": partition_name, "entity_count": -1, "error": str(e)}

    # ------------------------------------------------------------------
    # Deduplication workload
    # ------------------------------------------------------------------

    def create_dedup_collection(
        self, vector_dim: int, signature_dim: int, drop_existing: bool = True
    ) -> None:
        self._ensure_connected()
        collection_name = self._config.collection

        if self._client.collection_exists(collection_name):
            if drop_existing:
                self._client.delete_collection(collection_name)
            else:
                return

        # Create collection with a named binary vector
        vectors_config = {
            "vector": rest.VectorParams(size=vector_dim, distance=rest.Distance.COSINE)
        }
        sparse_vectors_config = None # Not sparse
        
        # Qdrant v1.7.0+ supports binary vectors. We pass datatype=VectorParams.datatype=uint8
        # Wait, the python client model might differ slightly based on version.
        # In recent qdrant-client:
        # rest.VectorParams(size=signature_dim, distance=rest.Distance.JACCARD, datatype=rest.Datatype.UINT8)
        try:
            vectors_config["signature"] = rest.VectorParams(
                size=signature_dim * 8, # Binary vectors in Qdrant are dimension of bits, but passed as bytes? Wait.
                distance=rest.Distance.JACCARD,
                datatype=rest.Datatype.UINT8
            )
        except AttributeError:
            # Fallback if qdrant client is old and doesn't support Datatype
            print("[QdrantAdapter] Warning: Old qdrant client, binary vectors might not be fully supported")
            vectors_config["signature"] = rest.VectorParams(
                size=signature_dim,
                distance=rest.Distance.JACCARD
            )

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
        print(f"[QdrantAdapter] Created dedup collection: {collection_name}")

    def insert_dedup(
        self, ids: List[int], vectors: List[List[float]], signatures: List[bytes]
    ) -> None:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        else:
            vectors = [v.tolist() if hasattr(v, "tolist") else v for v in vectors]
        # Convert bytes to lists of ints for JSON serialization
        import struct
        points = []
        for id_, vec, sig in zip(ids, vectors, signatures):
            # Convert bytes to uint8 array
            sig_list = list(sig)
            points.append(
                rest.PointStruct(
                    id=id_,
                    vector={"vector": vec, "signature": sig_list}
                )
            )
        self._client.upsert(self._config.collection, points=points)

    def search_dedup_batch(
        self,
        query_signatures: List[bytes],
        radius: float,
        top_k: int = 1,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        self._ensure_connected()
        params = params or {}
        search_params = None
        if params.get("ef") is not None:
            search_params = rest.SearchParams(hnsw_ef=params["ef"])

        requests = []
        for sig in query_signatures:
            sig_list = list(sig)
            req = rest.SearchRequest(
                vector=rest.NamedVector(name="signature", vector=sig_list),
                limit=top_k,
                params=search_params,
                score_threshold=radius, # Qdrant supports radius via score_threshold
                with_payload=False,
            )
            requests.append(req)

        batch_results = []
        CHUNK_SIZE = 1000
        for i in range(0, len(requests), CHUNK_SIZE):
            chunk = requests[i:i+CHUNK_SIZE]
            res = self._client.search_batch(
                collection_name=self._config.collection,
                requests=chunk
            )
            batch_results.extend(res)

        return [
            [{"id": hit.id, "distance": hit.score} for hit in hits]
            for hits in batch_results
        ]
