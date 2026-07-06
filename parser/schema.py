"""
Config Schema - Data models for configuration.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


def normalize_to_underscore(name: str) -> str:
    """
    Normalize underscore-separated name: 'Complete_Ingestion' -> 'complete_ingestion'
    User MUST use underscores as separators. Each part is lowercased.
    """
    parts = name.split('_')
    normalized_parts = [part.lower().strip() for part in parts if part]
    return '_'.join(normalized_parts)


# =============================================================================
# ENUMS
# =============================================================================

class WorkloadType(str, Enum):
    """Supported workload types."""
    DEDUPLICATION_WORKLOAD = "deduplication_workload"
    
    @classmethod
    def _missing_(cls, value):
        """Normalize underscore-separated input before matching."""
        if '_' not in str(value):
            return None
        normalized = normalize_to_underscore(str(value))
        for member in cls:
            if member.value == normalized:
                return member
        return None


class MetricType(str, Enum):
    """Distance/similarity metrics."""
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    L2 = "l2"
    JACCARD_UPPER = "JACCARD"
    JACCARD_LOWER = "jaccard"


# =============================================================================
# CONFIG MODELS
# =============================================================================

class DatabaseConfig(BaseModel):
    """Database connection settings."""
    adapter: str = Field(default="mock")
    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    collection: str = Field(default="default")



class QuantizationConfig(BaseModel):
    """Quantization settings."""
    method: Optional[str] = Field(default=None)  # PQ, SQ, etc.
    nbits: int = Field(default=8)


class IndexConfig(BaseModel):
    """Index parameters."""
    type: str = Field(default="HNSW")  # HNSW, IVF_FLAT, IVF_PQ, FLAT, etc.
    params: Dict[str, Any] = Field(default_factory=dict)  # M, ef_construction, nlist, etc.
    quantization: Optional[QuantizationConfig] = Field(default=None)
    metric: MetricType = Field(default=MetricType.COSINE)


class GlobalConfig(BaseModel):
    """Global settings: database, dataset, concurrency, seed."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    dataset: str = Field(default="")
    seed: int = Field(default=42)
    use_mmap : bool = Field(default=False)
    read_concurrency: Optional[int] = Field(default=None, ge=1)
    ingest_concurrency: Optional[int] = Field(default=None, ge=1)
    
    ingest_batch_size: int = Field(default=1500, ge=1)
    query_batch_size: int = Field(default=500, ge=1)
    vector_dimension: int = Field(default=128, ge=1)
    drop_collection_first: bool = Field(default=True)

# Common knobs for all workloads
class CommonWorkloadKnobs(BaseModel):
    k: int = Field(default=10, ge=1)
    query_time_budget_s: Optional[float] = Field(default=None, gt=0, description="Optional time budget for query phase in seconds")
    query_ratio: float = Field(default=0.01, ge=0, le=1)
    query_vectors: Optional[str] = Field(default=None)
    metrics: List[str] = Field(default=["latency_p50", "latency_p95", "qps"])
    drop_collection_first: bool = Field(default=True)

class DeduplicationWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for the Deduplication Workload.
    Two-phase deduplication using Bloom Filter and MinHash LSH.
    """
    type: Literal["deduplication_workload"] = "deduplication_workload"
    
    top_k: int = Field(default=1, ge=1)
    num_perm: int = Field(default=128, ge=8)
    jaccard_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    bloom_capacity: int = Field(default=100000, ge=1)
    bloom_error_rate: float = Field(default=0.01, gt=0.0, lt=1.0)



# Union of all workload types
WorkloadConfig = Union[
    DeduplicationWorkloadKnobs,
]


