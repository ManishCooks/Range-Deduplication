"""
Config Schema - Data models for configuration.
"""

from enum import Enum
from typing import Any, Dict, List, Optional,Literal
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
    COMPLETE_INGESTION_WORKLOAD = "complete_ingestion_workload"
    CONCURRENT_INGESTION_WORKLOAD = "concurrent_ingestion_workload"
    
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
    dataset: str
    seed: int = Field(default=42)
    concurrency: int = Field(default=4, ge=1)
    batch_size: int = Field(default=1000, ge=1)
    vector_dimension: int = Field(default=128, ge=1)


from pydantic import BaseModel, Field, RootModel
from typing import Union

# Common knobs for all workloads
class CommonWorkloadKnobs(BaseModel):
    k: int = Field(default=10, ge=1)
    query_ratio: float = Field(default=0.01, ge=0, le=1)
    query_vectors: Optional[str] = Field(default=None)
    metrics: List[str] = Field(default=["latency_p50", "latency_p95", "qps"])
    drop_collection_first: bool = Field(default=True)

# Complete ingestion workload knobs
class CompleteIngestionWorkloadKnobs(CommonWorkloadKnobs):
    type: Literal["complete_ingestion_workload"] = "complete_ingestion_workload"
    # Add any specific knobs for complete ingestion here

# Concurrent ingestion workload knobs
class ConcurrentIngestionWorkloadKnobs(CommonWorkloadKnobs):
    type: Literal["concurrent_ingestion_workload"] = "concurrent_ingestion_workload"
    
    # Existing Knobs
    initial_ingest_ratio: float = Field(default=0.5, ge=0, le=1)
    frequency_seconds: float = Field(default=5.0, gt=0)
    query_batch_size: int = Field(default=100, ge=1)
    max_duration_seconds: float = Field(default=0.0, ge=0)
    
    # Drift Detection Knobs 
    drift_check_interval: float = Field(default=10.0, gt=0, description="Seconds between drift checks")
    drift_threshold: float = Field(default=0.1, gt=0, description="MMD/Distance threshold to trigger re-index")
    drift_metric_type: Literal["mmd", "centroid"] = Field(default="mmd", description="Algorithm for drift detection")
    mmd_kernel_bandwidth: float = Field(default=1.0, gt=0, description="Sigma for MMD RBF kernel")

# Union of all workload types
WorkloadConfig = Union[CompleteIngestionWorkloadKnobs, ConcurrentIngestionWorkloadKnobs]
