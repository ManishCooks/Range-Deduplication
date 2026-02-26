"""
Config Schema - Data models for configuration.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


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
    RWD_WORKLOAD = "rwd_workload"
    BURST_RWD_WORKLOAD = "burst_rwd_workload"
    FILTERED_ANN_WORKLOAD = "filtered_ann_workload"
    MULTI_MODAL_WORKLOAD = "multi_modal_workload"
    HOT_COLD_WORKLOAD = "hot_cold_workload"
    
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


class RwdWorkloadKnobs(CommonWorkloadKnobs):
    type: Literal["rwd_workload"] = "rwd_workload"

    initial_ingest_ratio: float = Field(default=0.5, ge=0, le=1)

    read_ratio: float = Field(default=0.7, ge=0)
    write_ratio: float = Field(default=0.2, ge=0)
    delete_ratio: float = Field(default=0.1, ge=0)

    frequency_seconds: float = Field(default=5.0, gt=0)
    query_batch_size: int = Field(default=100, ge=1)
    max_duration_seconds: float = Field(default=0.0, ge=0)

    maintenance_check_interval: float = Field(default=10.0, gt=0)
    drift_threshold: float = Field(default=0.1, ge=0)
    zombie_threshold: float = Field(default=0.15, ge=0, le=1)
    drift_metric_type: Literal["mmd", "centroid"] = Field(default="mmd")
    mmd_kernel_bandwidth: float = Field(default=1.0, gt=0)


class BurstRwdWorkloadKnobs(CommonWorkloadKnobs):
    """Knobs for Burst Read-Write-Delete Workload."""
    type: Literal["burst_rwd_workload"] = "burst_rwd_workload"

    initial_ingest_ratio: float = Field(default=0.5, ge=0, le=1)

    # Baseline RWD ratios
    read_ratio: float = Field(default=0.7, ge=0)
    write_ratio: float = Field(default=0.2, ge=0)
    delete_ratio: float = Field(default=0.1, ge=0)

    # Burst pattern configuration
    burst_pattern: Literal["periodic", "sinusoidal", "step_function", "random"] = Field(default="periodic")
    burst_amplitude: float = Field(default=5.0, gt=1.0)
    burst_read_amplifier: float = Field(default=2.0, ge=1.0)
    burst_duration: float = Field(default=10.0, gt=0)
    burst_interval: float = Field(default=30.0, gt=0)
    cooldown_interval: float = Field(default=15.0, ge=0)
    num_bursts: int = Field(default=3, ge=1)
    recovery_threshold: float = Field(default=1.2, gt=1.0)
    recovery_timeout: float = Field(default=30.0, gt=0)

    # Op ratios during burst (separate from baseline)
    burst_write_ratio: float = Field(default=0.8, ge=0)
    burst_delete_ratio: float = Field(default=0.2, ge=0)

    # Timing
    frequency_seconds: float = Field(default=2.0, gt=0)
    query_batch_size: int = Field(default=100, ge=1)
    max_duration_seconds: float = Field(default=0.0, ge=0)

    # Maintenance
    maintenance_check_interval: float = Field(default=5.0, gt=0)
    drift_threshold: float = Field(default=0.05, ge=0)
    zombie_threshold: float = Field(default=0.15, ge=0, le=1)
    drift_metric_type: Literal["mmd", "centroid"] = Field(default="mmd")
    mmd_kernel_bandwidth: float = Field(default=1.0, gt=0)


class FilteredAnnWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for Filtered ANN Workload.
    Supports selectivity, post-filter thresholds, and concurrency overrides.
    """
    type: Literal["filtered_ann_workload"] = "filtered_ann_workload"
    
    # Workload-Specific Knobs
    filter_selectivity: float = Field(default=0.1, ge=0.0, le=1.0, description="Target fraction of vectors satisfying the filter")
    post_filter_threshold: float = Field(default=0.0, ge=0.0, description="Tau (τ) threshold for post-filtering optimization")
    
    # Overrides/Extensions
    concurrency: int = Field(default=4, ge=1, description="Concurrency level for filtered queries")
    query_limit: int = Field(default=1000, ge=1, description="Limit query set size for expensive ground truth calculation")

class MultiModalWorkloadKnobs(CommonWorkloadKnobs):
    type: Literal["multi_modal_workload"] = "multi_modal_workload"
    
    # "unified" : One single collection/index that holds every vector regardless of modality.
    # "partitioned" : One collection but physically split into multiple partitions (or separate collections) — one partition per modality

    # 1. Embedding Space Mode
    embedding_mode: Literal["unified", "partitioned"] = Field(default="unified")
    
    # 2. Modality Config (The "Database-side" mix)
    # Defines the ratio of vectors for each modality in the dataset
    modality_mix: Dict[str, float] = Field(default={"text": 1.0})
    
    # 3. Cross-Modal Eligibility
    # "enabled": Queries retrieve best matches regardless of modality
    # "restricted": Queries only retrieve matches from their own modality
    cross_modal_mode: Literal["enabled", "restricted"] = Field(default="enabled")
    
    # 4. Query Mix
    # If not provided, assumes query distribution matches dataset distribution
    query_modality_mix: Optional[Dict[str, float]] = None

    # 5. Hybrid Scoring (vector + BM25 tag match)
    # Combined final_score = (1 - w)*vector_score + w*bm25_score
    hybrid_scoring: bool = Field(default=False)
    hybrid_bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # 6. Advanced Metrics
    # Options: "precision_at_k", "ndcg", "latency_per_modality", "size_per_partition"
    advanced_metrics: List[str] = Field(default=["precision_at_k", "ndcg"])

class HotColdWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for the Hot-Cold / Zipf / Gaussian workload.
    Simulates non-uniform (power-law) query traffic to stress the DB cache.
    """
    type: Literal["hot_cold_workload"] = "hot_cold_workload"

    # Distribution mode
    distribution: Literal["bernoulli", "zipfian", "gaussian"] = Field(
        default="bernoulli",
        description="Query sampling distribution"
    )

    # Hot-set definition
    hot_fraction: float = Field(default=0.1, gt=0, lt=1,
        description="Fraction of base vectors designated as hot")

    # Bernoulli knob
    hot_query_ratio: float = Field(default=0.8, ge=0, le=1,
        description="P(query -> hot set) for bernoulli mode")

    # Zipfian knob
    zipf_exponent: float = Field(default=1.2, gt=0,
        description="s in P(rank=i) proportional to i^(-s)")

    # Gaussian knobs
    gaussian_sigma: Optional[float] = Field(default=None, gt=0,
        description="Noise std-dev per dim; defaults to 1/dim if None")

    # Query volume
    n_queries: int = Field(default=10000, ge=1,
        description="Total number of queries to issue")
    query_batch_size: int = Field(default=500, ge=1,
        description="Queries issued per concurrent batch")

    # Output
    export_histogram: bool = Field(default=False,
        description="Include per-vector access counts in results")


# Union of all workload types
WorkloadConfig = Union[
    CompleteIngestionWorkloadKnobs,
    ConcurrentIngestionWorkloadKnobs,
    RwdWorkloadKnobs,
    BurstRwdWorkloadKnobs,
    FilteredAnnWorkloadKnobs,
    MultiModalWorkloadKnobs,
    HotColdWorkloadKnobs,
]