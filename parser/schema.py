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
    COMPLETE_INGESTION_WORKLOAD = "complete_ingestion_workload"
    CONCURRENT_INGESTION_WORKLOAD = "concurrent_ingestion_workload"
    RWD_WORKLOAD = "rwd_workload"
    BURST_RWD_WORKLOAD = "burst_rwd_workload"
    FILTERED_ANN_WORKLOAD = "filtered_ann_workload"
    MULTI_MODAL_WORKLOAD = "multi_modal_workload"
    HOT_COLD_WORKLOAD = "hot_cold_workload"
    OUTLIER_WORKLOAD = "outlier_workload"
    SPARSE_WORKLOAD = "sparse_workload"
    COLD_START_WORKLOAD = "cold_start_workload"
    LID_WORKLOAD = "lid_workload"
    DEDUPLICATION_WORKLOAD = "deduplication_workload"
    OOD_WORKLOAD = "ood_workload"
    TEMPORAL_FRESHNESS_WORKLOAD = "temporal_freshness_workload"
    
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

class PineconeConfig(BaseModel):
    """Pinecone Serverless connection settings."""
    index_name: str = Field(default="dynavec-index")
    dimension: int = Field(default=128, ge=1)
    metric: Literal["cosine", "euclidean", "dotproduct"] = Field(default="cosine")
    cloud: Literal["aws", "gcp", "azure"] = Field(default="aws")
    region: str = Field(default="us-east-1")
    namespace: str = Field(default="")


class DatabaseConfig(BaseModel):
    """Database connection settings."""
    adapter: str = Field(default="mock")
    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    collection: str = Field(default="default")
    pinecone_config: Optional[PineconeConfig] = Field(default=None)


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
    pipeann_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

# Common knobs for all workloads
class CommonWorkloadKnobs(BaseModel):
    k: int = Field(default=10, ge=1)
    query_time_budget_s: Optional[float] = Field(default=None, gt=0, description="Optional time budget for query phase in seconds")
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
    max_duration_seconds: float = Field(default=0.0, ge=0)
    
    # Drift Detection Knobs 
    drift_check_interval: float = Field(default=10.0, gt=0, description="Seconds between drift checks")
    drift_threshold: float = Field(default=0.1, gt=0, description="MMD/Distance threshold to trigger re-index")
    drift_metric_type: Literal["mmd", "centroid"] = Field(default="mmd", description="Algorithm for drift detection")
    mmd_kernel_bandwidth: float = Field(default=1.0, gt=0, description="Sigma for MMD RBF kernel")


class RwdWorkloadKnobs(CommonWorkloadKnobs):
    type: Literal["rwd_workload"] = "rwd_workload"

    initial_ingest_ratio: float = Field(default=0.5, ge=0, le=1)
    write_batch_size : int = Field(default=500,ge=0)
    query_vectors_path : str = Field(default="")
    query_ratio : float = Field(default=None,ge=0,le=1)

    delete_batch_size : int = Field(default=500,ge=0)
    write_concurrency: Optional[int] = Field(default=None, ge=1)
    delete_concurrency: Optional[int] = Field(default=None, ge=1)
    write_ratio: float = Field(default=0.2, ge=0)
    delete_ratio: float = Field(default=0.1, ge=0)
    
    frequency_seconds: float = Field(default=5.0, gt=0)
    max_duration_seconds: float = Field(default=0.0, ge=0)

    maintenance_check_interval: float = Field(default=10.0, gt=0)
    drift_threshold: float = Field(default=0.1, ge=0,le=1)
    zombie_threshold: float = Field(default=0.15, ge=0, le=1)
    drift_metric_type: Literal["mmd", "centroid"] = Field(default="mmd")
    mmd_kernel_bandwidth: Optional[float] = Field(default=None, gt=0)
    drift_buffer_size : int = Field(default=2000,ge=0)
    drift_min_buffer : int = Field(default=100,ge=0)
    frequency_seconds : float = Field(default=5.0,ge=0)
    max_duration_seconds : float = Field(default=180.0,ge=0.0)
    ttl_seconds :float = Field(default=60.0,ge=0.0)
    ttl_jitter  :float = Field(default=10.0,ge=0.0)


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


class OodIngestionWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for the OOD (Out-of-Distribution) Ingestion Read-Only Workload.
    Uses LID-verified synthesis to generate OOD queries that escape the
    data manifold, then mixes them with in-distribution queries.
    """
    type: Literal["ood_ingestion_workload"] = "ood_ingestion_workload"

    # OOD query mixing
    ood_fraction: float = Field(default=0.2, ge=0, le=1,
        description="Fraction of query set that is OOD")
    ood_mix_ratio: float = Field(default=0.3, ge=0, le=1,
        description="Fraction of each batch that is OOD (Amt)")
    querying_pattern: Literal["uniform", "burst"] = Field(default="uniform",
        description="How OOD queries are interleaved with ID queries")
    burst_size: int = Field(default=50, ge=1,
        description="OOD queries per burst block (burst mode only)")

    # Query volume
    n_queries: int = Field(default=5000, ge=1,
        description="Total number of queries to issue")
    query_batch_size: int = Field(default=200, ge=1,
        description="Queries per concurrent batch")

    # OOD Synthesis — Phase 1: Boundary identification
    ood_boundary_percentile: float = Field(default=95.0, ge=0, le=100,
        description="k-NN distance percentile cutoff for boundary set")
    ood_k: int = Field(default=20, ge=2,
        description="k for kNN in boundary detection and LID computation")

    # OOD Synthesis — Phase 2: Geometric push
    ood_gamma: float = Field(default=2.0, gt=0,
        description="Step-size for outward push along boundary trajectory")
    ood_sigma: Optional[float] = Field(default=None, gt=0,
        description="Noise scale for isotropic perturbation; None = auto")

    # OOD Synthesis — Phase 3 & 4: LID filter
    ood_lid_threshold: Optional[float] = Field(default=None, gt=0,
        description="LID acceptance threshold tau; None = auto")
    ood_max_retries: int = Field(default=10, ge=1,
        description="Max retries per seed before force-accept")

    # Output
    export_ood_freq: bool = Field(default=False,
        description="Include per-vector OOD hit counts in results")


class OutlierWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for the Outlier Read Workload.
    Combines k-NN with Range/Radius Search (epsilon).
    Outlier queries are synthesized via NPOS (Tao et al., ICLR 2023).
    All k-NN lookups during synthesis use a local FAISS ShadowIndex to
    avoid O(B x N) distance matrices.
    """
    type: Literal["outlier_workload"] = "outlier_workload"

    # Query search parameters
    epsilon: float = Field(
        default=0.5, ge=0.0,
        description="Distance/radius threshold for range search")
    range_filter: Optional[float] = Field(
        default=None,
        description="Optional lower bound for range search distance")

    # Query volume
    n_queries: int = Field(default=1000, ge=1,
        description="Total normal queries to issue")
    query_batch_size: int = Field(default=100, ge=1,
        description="Queries per concurrent batch")

    # Outlier budget
    outlier_ratio: float = Field(default=0.05, gt=0.0, le=1.0,
        description=(
            "Fraction of n_queries to synthesize as outlier probes. "
            "e.g. 0.05 -> 50 outliers for n_queries=1000"))

    # NPOS synthesis knobs (paper: Tao et al., ICLR 2023)
    k_boundary: int = Field(default=300, ge=1,
        description=(
            "k for d_k(z, Z) boundary scoring and best-of-p selection (Eq. 4). "
            "Paper default: 300-400"))
    boundary_ratio: float = Field(default=0.1, gt=0.0, le=1.0,
        description="Fraction of boundary_sample_size kept as boundary vectors")
    sigma: float = Field(default=0.316, gt=0.0,
        description=(
            "sigma for isotropic Gaussian v ~ N(h(x_i), sigma^2 I) (Eq. 5). "
            "Paper reports sigma^2; supply sigma here. Paper default: sigma^2=0.1 -> sigma~0.316"))
    p_candidates: int = Field(default=1000, ge=1,
        description=(
            "Candidates drawn per boundary sample before best-of-p selection. "
            "Paper default: 1000 (Table 12)"))
    boundary_sample_size: int = Field(default=5000, ge=1,
        description="Base vectors sampled for boundary scoring (Phase 1)")

class SparseWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for the Sparse Workload.
    Tests system behaviour under low utilization (continuous low-QPS).
    """
    type: Literal["sparse_workload"] = "sparse_workload"
    
    target_qps: float = Field(default=0.5, gt=0.0, description="Target average Queries Per Second")
    traffic_pattern: Literal["poisson", "fixed"] = Field(default="poisson", description="Inter-arrival time distribution")
    duration_seconds: Optional[float] = Field(default=None, description="Optional cap on how long the benchmark runs")
    n_queries: int = Field(default=1000, ge=1, description="Total number of queries to issue")
    query_batch_size: int = Field(default=1, ge=1, description="Typically 1 for sparse to see individual latencies")


class ColdStartWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for the Cold-Start Workload.
    Measures cold-start query latency by cycling between query bursts
    and forced DB restarts (Docker) or sleep periods (serverless).
    """
    type: Literal["cold_start_workload"] = "cold_start_workload"

    # Restart mode
    restart_mode: Literal["docker", "sleep"] = Field(
        default="sleep",
        description="'docker' restarts a container; 'sleep' simulates serverless idle"
    )
    docker_containers: Optional[List[str]] = Field(
        default=None,
        description="List of Docker container names/IDs to restart (e.g. etcd, minio, standalone)"
    )

    @model_validator(mode="before")
    @classmethod
    def _compat_docker_container(cls, values):
        """Accept legacy singular 'docker_container' and promote to list."""
        if isinstance(values, dict):
            singular = values.pop("docker_container", None)
            if singular and not values.get("docker_containers"):
                values["docker_containers"] = [singular]
        return values

    docker_restart_timeout_s: float = Field(
        default=120.0, gt=0,
        description="Seconds to wait for container health after docker restart"
    )
    sleep_duration_seconds: float = Field(
        default=300.0, gt=0,
        description="Sleep timer (seconds) for serverless cold-start simulation"
    )
    queries_per_cycle: int = Field(
        default=50, ge=1,
        description="Number of queries per active burst after each restart"
    )
    num_cycles: int = Field(
        default=5, ge=1,
        description="Total number of restart → query cycles"
    )
    n_queries: int = Field(
        default=10000, ge=1,
        description="Hard cap on total queries across all cycles"
    )
    warmup_track_n: int = Field(
        default=20, ge=1,
        description="Number of queries to track in warmup curve per cycle"
    )



class LidWorkloadKnobs(CommonWorkloadKnobs):
    """
    Knobs for the LID-Aware RWD Workload.
    Layers Local Intrinsic Dimensionality awareness onto the RWD pipeline:
    LID-sorted ingestion, KS-test drift detection, and LID-ordered reindex.
    """
    type: Literal["lid_workload"] = "lid_workload"

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
    drift_metric_type: Literal["mmd", "centroid", "lid_ks"] = Field(default="lid_ks")
    mmd_kernel_bandwidth: float = Field(default=1.0, gt=0)

    # LID-specific knobs
    insertion_order: Literal["desc", "asc", "random"] = Field(
        default="desc",
        description="Sort order for bulk ingestion by LID score")
    lid_k: int = Field(
        default=100, ge=2,
        description="k for Hill MLE LID estimation")
    lid_sample_size: int = Field(
        default=0, ge=0,
        description="Sample size for LID scoring during ingestion (0 = all)")
    lid_drift_window: int = Field(
        default=200, ge=10,
        description="Write LID scores to accumulate before KS-test")
    reindex_lid_ordered: bool = Field(
        default=True,
        description="Sort live vectors by LID descending during reindex")
    reindex_lid_k: int = Field(
        default=50, ge=2,
        description="k for LID recomputation during reindex")

class OodWorkloadKnobs(CommonWorkloadKnobs):
    type: Literal["ood_workload"] = "ood_workload"
    ingestion_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    id_query_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    total_queries: int = Field(default=10000, ge=1)
    ood_query_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

class TemporalFreshnessWorkloadKnobs(CommonWorkloadKnobs):
    type: Literal["temporal_freshness_workload"] = "temporal_freshness_workload"
    k_prime: int = Field(default=50, ge=1)
    freshness_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    decay_lambda: float = Field(default=1/86400, gt=0.0)
    similarity_threshold: float = Field(default=0.0)
    n_queries: int = Field(default=1000, ge=1)
    time_window_days: int = Field(default=365, ge=1)
    distribution_mode: Literal["uniform", "recent_heavy", "old_heavy"] = Field(default="uniform")
    beta_a: float = Field(default=2.0, gt=0.0)
    beta_b: float = Field(default=5.0, gt=0.0)
    search_params: Dict[str, Any] = Field(default_factory=dict)
    rerank_concurrency: int = Field(default=1, ge=1)
    time_source: Literal["synthetic", "metadata"] = Field(default="synthetic")
    time_column: str = Field(default="timestamp")
    total_queries: int = Field(default=10000, ge=1)

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
    CompleteIngestionWorkloadKnobs,
    ConcurrentIngestionWorkloadKnobs,
    RwdWorkloadKnobs,
    BurstRwdWorkloadKnobs,
    FilteredAnnWorkloadKnobs,
    MultiModalWorkloadKnobs,
    HotColdWorkloadKnobs,
    OutlierWorkloadKnobs,
    SparseWorkloadKnobs,
    ColdStartWorkloadKnobs,
    LidWorkloadKnobs,
    DeduplicationWorkloadKnobs,
    OodWorkloadKnobs,
    TemporalFreshnessWorkloadKnobs,
]


# =============================================================================
# OOD PREPROCESSOR CONFIG
# Read from the top-level "preprocessor" key in the JSON config.
# Not part of the workload union — used by ood_workload/pipeline.py.
# =============================================================================

class PoolConfig(BaseModel):
    """
    Config for one pool (ID or OOD).

    Two mutually exclusive modes, determined by the type of `classes`:

    **Mode 1 — class-index filter** (``classes`` is a list of ints):
        Load ``dataset`` and keep only samples whose class label is in the list.
        Example: ``{"dataset": "CIFAR10", "classes": [0, 1, 2]}``

    **Mode 2 — multi-dataset** (``classes`` is a list of strings):
        Each string is a dataset name.  All samples from every listed dataset
        are loaded in full (no class filtering).  ``dataset`` is ignored.
        Example: ``{"classes": ["CIFAR100", "MNIST"]}``
    """
    name: Optional[str] = Field(
        default=None,
        description="Primary dataset name ('CIFAR10', 'CIFAR100', 'ImageNet'")
    root: str = Field(
        default="./datasets/raw",
        description="Local root directory for torchvision to cache/load the dataset.")
    classes: Optional[List[Union[int, str]]] = Field(
        default=None,
        description=(
                "Mode 1: list of int class indices to keep from `dataset`. "
                "Mode 2: list of str dataset names to load entirely. "
            )
    )

    @model_validator(mode="after")
    def _validate_pool(self) -> "PoolConfig":
        classes = self.classes

        if classes is None or len(classes) == 0:
            raise ValueError(
                "PoolConfig: 'classes' must be provided and non-empty."
            )

        # Mode 1 — list[int]
        if isinstance(classes[0], int):
            if not self.name:
                raise ValueError(
                    "PoolConfig: 'name' is required when 'classes' is a list of ints."
                )

        # Mode 2 — list[str]
        # no constraint, dataset ignored

        return self


class PreprocessorModelConfig(BaseModel):
    backbone: str = Field(default="resnet50")
    pretrained: bool = Field(default=True)
    epochs: int = Field(default=20)
    lr: float = Field(default=0.001)
    batch_size: int = Field(default=256)
    device: str = Field(default="cpu")


class PreprocessorDatasetsConfig(BaseModel):
    id: PoolConfig = Field(description="In-distribution pool settings")
    ood: PoolConfig = Field(description="Out-of-distribution pool settings")


class OodPreprocessorConfig(BaseModel):
    """
    Top-level config for the OOD preprocessor utility (ood_workload Phase 1).
    Read from the 'preprocessor' key of the JSON config file.
    """
    model: PreprocessorModelConfig
    output_dir: str = Field(
        default="./preprocessed_vectors",
        description="Directory where the output HDF5 file is written")
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric stored as metadata: 'cosine', 'l2', 'inner_product'")
    datasets: PreprocessorDatasetsConfig
