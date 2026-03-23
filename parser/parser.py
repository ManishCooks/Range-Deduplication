"""
Config Parser - Load JSON and return dicts.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union


class ConfigError(Exception):
    """Config parsing error."""
    pass


def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load config from JSON file.
    Returns dict with 'global', 'index', 'workload' sections.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ConfigError(f"File not found: {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    data = json.loads(content)
    
    if "global" not in data:
        raise ConfigError("Missing 'global' section in config")
    
    return data


# =============================================================================
# GLOBAL CONFIG HELPERS
# =============================================================================

def get_global_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract global config section."""
    return config.get("global", {})


def get_db_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract database settings from global."""
    return get_global_config(config).get("database", {})


def get_dataset_name(config: Dict[str, Any]) -> str:
    """Get dataset name from global."""
    return get_global_config(config).get("dataset", "")


def get_seed(config: Dict[str, Any]) -> int:
    """Get random seed from global."""
    return get_global_config(config).get("seed", 42)


def get_concurrency(config: Dict[str, Any]) -> int:
    """Get concurrency from global."""
    return get_global_config(config).get("concurrency", 4)


def get_batch_size(config: Dict[str, Any]) -> int:
    """Get batch size from global."""
    return get_global_config(config).get("batch_size", 1000)


def get_vector_dimension(config: Dict[str, Any]) -> int:
    """Get vector dimension from global."""
    return get_global_config(config).get("vector_dimension", 128)


# =============================================================================
# INDEX CONFIG HELPERS
# =============================================================================

def get_index_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract index config section."""
    return config.get("index", {})


def get_index_type(config: Dict[str, Any]) -> str:
    """Get index type."""
    return get_index_config(config).get("type", "HNSW")


def get_index_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get index params (M, ef_construction, nlist, etc.)."""
    return get_index_config(config).get("params", {})


def get_quantization(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get quantization settings."""
    return get_index_config(config).get("quantization", {})


def get_metric(config: Dict[str, Any]) -> str:
    """Get distance metric."""
    return get_index_config(config).get("metric", "cosine")


# =============================================================================
# WORKLOAD CONFIG HELPERS
# =============================================================================

def get_workload_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract workload config section."""
    return config.get("workload", {})


def get_workload_type(config: Dict[str, Any]) -> str:
    """Get workload type."""
    return get_workload_config(config).get("type", "complete_ingestion_workload")


def get_k(config: Dict[str, Any]) -> int:
    """Get top-k for queries."""
    return get_workload_config(config).get("k", 10)


def get_query_ratio(config: Dict[str, Any]) -> float:
    """Get query ratio (fraction of dataset for queries)."""
    return get_workload_config(config).get("query_ratio", 0.01)


def get_query_vectors_path(config: Dict[str, Any]) -> str:
    """Get path to separate query vectors file."""
    return get_workload_config(config).get("query_vectors", None)


def get_metrics(config: Dict[str, Any]) -> list:
    """Get list of metrics to collect."""
    return get_workload_config(config).get("metrics", ["latency_p50", "latency_p95", "qps"])


# =============================================================================
# COMMON WORKLOAD HELPERS
# =============================================================================

def get_drop_collection_first(config: Dict[str, Any]) -> bool:
    """Get whether to drop collection before starting."""
    return get_workload_config(config).get("drop_collection_first", True)


# =============================================================================
# CONCURRENT INGESTION WORKLOAD HELPERS
# =============================================================================

def get_initial_ingest_ratio(config: Dict[str, Any]) -> float:
    """Get initial ingest ratio (fraction ingested before concurrent phase)."""
    return get_workload_config(config).get("initial_ingest_ratio", 0.5)


def get_frequency_seconds(config: Dict[str, Any]) -> float:
    """Get frequency interval for secondary operation in concurrent phase."""
    return get_workload_config(config).get("frequency_seconds", 5.0)


def get_query_batch_size(config: Dict[str, Any]) -> int:
    """Get number of queries per round in concurrent phase."""
    return get_workload_config(config).get("query_batch_size", 100)


def get_max_duration_seconds(config: Dict[str, Any]) -> float:
    """Get max duration for concurrent phase (0=unlimited)."""
    return get_workload_config(config).get("max_duration_seconds", 0.0)

def get_drift_check_interval(config: Dict[str, Any]) -> float:
    """Get interval for drift monitoring."""
    return get_workload_config(config).get("drift_check_interval", 10.0)

def get_drift_threshold(config: Dict[str, Any]) -> float:
    """Get threshold for triggering re-indexing."""
    return get_workload_config(config).get("drift_threshold", 0.1)

def get_drift_metric_type(config: Dict[str, Any]) -> str:
    """Get drift metric type (mmd or centroid)."""
    return get_workload_config(config).get("drift_metric_type", "mmd")

def get_mmd_kernel_bandwidth(config: Dict[str, Any]) -> float:
    """Get kernel bandwidth for MMD."""
    return get_workload_config(config).get("mmd_kernel_bandwidth", 1.0)


# =============================================================================
# RWD WORKLOAD HELPERS
# =============================================================================

def get_read_ratio(config: Dict[str, Any]) -> float:
    """Get read ratio for RWD workload."""
    return get_workload_config(config).get("read_ratio", 0.7)


def get_write_ratio(config: Dict[str, Any]) -> float:
    """Get write ratio for RWD workload."""
    return get_workload_config(config).get("write_ratio", 0.2)


def get_delete_ratio(config: Dict[str, Any]) -> float:
    """Get delete ratio for RWD workload."""
    return get_workload_config(config).get("delete_ratio", 0.1)


def get_zombie_threshold(config: Dict[str, Any]) -> float:
    """Get zombie threshold for re-indexing."""
    return get_workload_config(config).get("zombie_threshold", 0.15)


def get_maintenance_check_interval(config: Dict[str, Any]) -> float:
    """Get interval for maintenance (drift/zombie) checks."""
    return get_workload_config(config).get("maintenance_check_interval", 10.0)

# =============================================================================
# FILTERED ANN WORKLOAD HELPERS
# =============================================================================

def get_filter_selectivity(config: Dict[str, Any]) -> float:
    """Target fraction of vectors satisfying the filter."""
    return get_workload_config(config).get("filter_selectivity", 0.1)


def get_post_filter_threshold(config: Dict[str, Any]) -> float:
    """
    Tau threshold for post-filter optimization.
    Default mirrors workload code (tau) while staying close to schema intent.
    """
    return get_workload_config(config).get("post_filter_threshold", 0.05)


def get_query_limit(config: Dict[str, Any]) -> int:
    """Limit query set size for expensive filtered GT calculation."""
    return get_workload_config(config).get("query_limit", 1000)


# =============================================================================
# MULTI-MODAL WORKLOAD HELPERS
# =============================================================================

def get_embedding_mode(config: Dict[str, Any]) -> str:
    """Get embedding space mode (unified, partitioned)."""
    return get_workload_config(config).get("embedding_mode", "unified")


def get_modality_mix(config: Dict[str, Any]) -> Dict[str, float]:
    """Get dataset modality ratios."""
    return get_workload_config(config).get("modality_mix", {"text": 1.0})


def get_cross_modal_mode(config: Dict[str, Any]) -> str:
    """Get cross-modal eligibility (enabled, restricted)."""
    return get_workload_config(config).get("cross_modal_mode", "enabled")


def get_query_modality_mix(config: Dict[str, Any]) -> Any:
    """Get query modality ratios (optional)."""
    return get_workload_config(config).get("query_modality_mix", None)


def get_hybrid_scoring(config: Dict[str, Any]) -> bool:
    """Get whether hybrid scoring (vector + BM25) is enabled."""
    return bool(get_workload_config(config).get("hybrid_scoring", False))


def get_hybrid_bm25_weight(config: Dict[str, Any]) -> float:
    """Get the BM25 weight (w) in: final = (1-w)*vec + w*bm25."""
    return float(get_workload_config(config).get("hybrid_bm25_weight", 0.3))


def get_advanced_metrics(config: Dict[str, Any]) -> list:
    """Get list of advanced metrics to compute."""
    return get_workload_config(config).get("advanced_metrics", ["precision_at_k", "ndcg"])


# =============================================================================
# HOT-COLD WORKLOAD HELPERS
# =============================================================================

def get_distribution(config: Dict[str, Any]) -> str:
    """Get query sampling distribution (bernoulli, zipfian, gaussian)."""
    return get_workload_config(config).get("distribution", "bernoulli")


def get_hot_fraction(config: Dict[str, Any]) -> float:
    """Get fraction of base vectors designated as hot."""
    return get_workload_config(config).get("hot_fraction", 0.1)


def get_hot_query_ratio(config: Dict[str, Any]) -> float:
    """Get P(query -> hot set) for bernoulli distribution mode."""
    return get_workload_config(config).get("hot_query_ratio", 0.8)


def get_zipf_exponent(config: Dict[str, Any]) -> float:
    """Get Zipf exponent s in P(rank=i) ∝ i^(-s)."""
    return get_workload_config(config).get("zipf_exponent", 1.2)


def get_gaussian_sigma(config: Dict[str, Any]) -> float:
    """Get Gaussian noise std-dev per dim; None defaults to 1/dim."""
    return get_workload_config(config).get("gaussian_sigma", None)


def get_n_queries(config: Dict[str, Any]) -> int:
    """Get total number of queries to issue in hot-cold workload."""
    return get_workload_config(config).get("n_queries", 10000)


def get_export_histogram(config: Dict[str, Any]) -> bool:
    """Get whether to include per-vector access counts in results."""
    return bool(get_workload_config(config).get("export_histogram", False))


# =============================================================================
# BURST RWD WORKLOAD HELPERS
# =============================================================================

def get_burst_pattern(config: Dict[str, Any]) -> str:
    """Get burst pattern type (periodic, sinusoidal, step_function, random)."""
    return get_workload_config(config).get("burst_pattern", "periodic")


def get_burst_amplitude(config: Dict[str, Any]) -> float:
    """Get burst amplitude multiplier for mutation rate."""
    return float(get_workload_config(config).get("burst_amplitude", 5.0))


def get_burst_read_amplifier(config: Dict[str, Any]) -> float:
    """Get read concurrency multiplier during burst phase."""
    return float(get_workload_config(config).get("burst_read_amplifier", 2.0))


def get_burst_duration(config: Dict[str, Any]) -> float:
    """Get duration of each burst phase in seconds."""
    return float(get_workload_config(config).get("burst_duration", 10.0))


def get_burst_interval(config: Dict[str, Any]) -> float:
    """Get time between burst start events in seconds."""
    return float(get_workload_config(config).get("burst_interval", 30.0))


def get_cooldown_interval(config: Dict[str, Any]) -> float:
    """Get cooldown period (read-only, no mutations) after each burst in seconds."""
    return float(get_workload_config(config).get("cooldown_interval", 15.0))


def get_num_bursts(config: Dict[str, Any]) -> int:
    """Get number of burst cycles to execute."""
    return int(get_workload_config(config).get("num_bursts", 3))


def get_recovery_threshold(config: Dict[str, Any]) -> float:
    """Get latency multiplier threshold that signals recovery from burst."""
    return float(get_workload_config(config).get("recovery_threshold", 1.2))


def get_recovery_timeout(config: Dict[str, Any]) -> float:
    """Get max time to wait for recovery before moving on, in seconds."""
    return float(get_workload_config(config).get("recovery_timeout", 30.0))


def get_burst_write_ratio(config: Dict[str, Any]) -> float:
    """Get write fraction of mutations during burst phase."""
    return float(get_workload_config(config).get("burst_write_ratio", 0.8))


def get_burst_delete_ratio(config: Dict[str, Any]) -> float:
    """Get delete fraction of mutations during burst phase."""
    return float(get_workload_config(config).get("burst_delete_ratio", 0.2))


# =============================================================================
# OOD INGESTION WORKLOAD HELPERS
# =============================================================================

def get_ood_fraction(config: Dict[str, Any]) -> float:
    """Get fraction of query set that is OOD."""
    return float(get_workload_config(config).get("ood_fraction", 0.2))


def get_ood_mix_ratio(config: Dict[str, Any]) -> float:
    """Get fraction of each query batch that is OOD (Amt)."""
    return float(get_workload_config(config).get("ood_mix_ratio", 0.3))


def get_querying_pattern(config: Dict[str, Any]) -> str:
    """Get OOD query interleaving pattern (uniform, burst)."""
    return get_workload_config(config).get("querying_pattern", "uniform")


def get_ood_burst_size(config: Dict[str, Any]) -> int:
    """Get OOD queries per burst block."""
    return int(get_workload_config(config).get("burst_size", 50))


def get_ood_gamma(config: Dict[str, Any]) -> float:
    """Get step-size for outward push in OOD synthesis."""
    return float(get_workload_config(config).get("ood_gamma", 2.0))


def get_ood_sigma(config: Dict[str, Any]):
    """Get noise scale for OOD synthesis; None = auto."""
    val = get_workload_config(config).get("ood_sigma", None)
    return float(val) if val is not None else None


def get_ood_lid_threshold(config: Dict[str, Any]):
    """Get LID acceptance threshold tau; None = auto."""
    val = get_workload_config(config).get("ood_lid_threshold", None)
    return float(val) if val is not None else None


def get_ood_boundary_percentile(config: Dict[str, Any]) -> float:
    """Get k-NN distance percentile cutoff for boundary set."""
    return float(get_workload_config(config).get("ood_boundary_percentile", 95.0))


def get_ood_max_retries(config: Dict[str, Any]) -> int:
    """Get max retries per seed in OOD synthesis."""
    return int(get_workload_config(config).get("ood_max_retries", 10))


def get_ood_k(config: Dict[str, Any]) -> int:
    """Get k for kNN in OOD boundary detection and LID computation."""
    return int(get_workload_config(config).get("ood_k", 20))


def get_export_ood_freq(config: Dict[str, Any]) -> bool:
    """Get whether to include per-vector OOD hit counts in results."""
    return bool(get_workload_config(config).get("export_ood_freq", False))


# =============================================================================
# OUTLIER WORKLOAD HELPERS
# =============================================================================

def get_epsilon(config: Dict[str, Any]) -> float:
    """Get radius/distance threshold for range search."""
    return float(get_workload_config(config).get("epsilon", 0.5))


def get_range_filter(config: Dict[str, Any]):
    """Get optional lower bound for range search; None if not set."""
    val = get_workload_config(config).get("range_filter", None)
    return float(val) if val is not None else None


def get_outlier_ratio(config: Dict[str, Any]) -> float:
    """Get fraction of n_queries to synthesize as outlier probes."""
    return float(get_workload_config(config).get("outlier_ratio", 0.05))


def get_k_boundary(config: Dict[str, Any]) -> int:
    """Get k for NPOS boundary scoring and best-of-p selection (Eq. 4)."""
    return int(get_workload_config(config).get("k_boundary", 300))


def get_boundary_ratio(config: Dict[str, Any]) -> float:
    """Get fraction of boundary_sample_size kept as boundary vectors."""
    return float(get_workload_config(config).get("boundary_ratio", 0.1))


def get_sigma(config: Dict[str, Any]) -> float:
    """Get sigma for isotropic Gaussian N(h(x_i), sigma^2 I) in NPOS (Eq. 5)."""
    return float(get_workload_config(config).get("sigma", 0.316))


def get_p_candidates(config: Dict[str, Any]) -> int:
    """Get candidates drawn per boundary sample before best-of-p selection."""
    return int(get_workload_config(config).get("p_candidates", 1000))


def get_boundary_sample_size(config: Dict[str, Any]) -> int:
    """Get number of base vectors sampled for NPOS boundary scoring."""
    return int(get_workload_config(config).get("boundary_sample_size", 5000))


# =============================================================================
# TEMPORAL FRESHNESS WORKLOAD HELPERS
# =============================================================================

def get_k_prime(config: Dict[str, Any]) -> int:
    """Get number of initial ANN candidates to fetch before re-ranking."""
    return int(get_workload_config(config).get("k_prime", 50))


def get_alpha(config: Dict[str, Any]) -> float:
    """Get freshness/causal weight (0=pure semantic, 1=pure freshness)."""
    return float(get_workload_config(config).get("alpha", 0.3))


def get_decay_lambda(config: Dict[str, Any]) -> float:
    """Get exponential decay rate for the causal score."""
    return float(get_workload_config(config).get("decay_lambda", 0.01))


def get_similarity_threshold(config: Dict[str, Any]) -> float:
    """Get minimum semantic inner-product score for re-ranking."""
    return float(get_workload_config(config).get("similarity_threshold", 0.0))


# =============================================================================
# COLD-START WORKLOAD HELPERS
# =============================================================================

def get_restart_mode(config: Dict[str, Any]) -> str:
    """Get restart mode ('docker' or 'sleep')."""
    return get_workload_config(config).get("restart_mode", "sleep")


def get_docker_containers(config: Dict[str, Any]) -> list:
    """Get list of Docker container names/IDs to restart."""
    wc = get_workload_config(config)
    containers = wc.get("docker_containers", None)
    if containers:
        return list(containers)
    # Backward compat: fall back to singular field
    singular = wc.get("docker_container", "")
    return [singular] if singular else []


def get_docker_container(config: Dict[str, Any]) -> str:
    """DEPRECATED: Use get_docker_containers(). Returns first container name."""
    containers = get_docker_containers(config)
    return containers[0] if containers else ""


def get_docker_restart_timeout(config: Dict[str, Any]) -> float:
    """Get timeout (seconds) for waiting after docker restart."""
    return float(get_workload_config(config).get("docker_restart_timeout_s", 120.0))


def get_sleep_duration_seconds(config: Dict[str, Any]) -> float:
    """Get sleep duration for serverless cold-start simulation."""
    return float(get_workload_config(config).get("sleep_duration_seconds", 300.0))


def get_queries_per_cycle(config: Dict[str, Any]) -> int:
    """Get number of queries per active burst after each restart."""
    return int(get_workload_config(config).get("queries_per_cycle", 50))


def get_num_cycles(config: Dict[str, Any]) -> int:
    """Get total number of restart-query cycles."""
    return int(get_workload_config(config).get("num_cycles", 5))


def get_warmup_track_n(config: Dict[str, Any]) -> int:
    """Get number of queries to track in warmup curve per cycle."""
    return int(get_workload_config(config).get("warmup_track_n", 20))


# =============================================================================
# LID WORKLOAD HELPERS
# =============================================================================

def get_insertion_order(config: Dict[str, Any]) -> str:
    """Get LID ingestion sort order ('desc', 'asc', or 'random'). Default 'desc'."""
    return get_workload_config(config).get("insertion_order", "desc")


def get_lid_k(config: Dict[str, Any]) -> int:
    """Get k for Hill MLE LID estimation. Default 100."""
    return int(get_workload_config(config).get("lid_k", 100))


def get_lid_sample_size(config: Dict[str, Any]) -> int:
    """Get sample size for LID scoring during bulk ingestion (0 = all). Default 0."""
    return int(get_workload_config(config).get("lid_sample_size", 0))


def get_lid_drift_window(config: Dict[str, Any]) -> int:
    """Get number of write LID scores to accumulate before running KS-test. Default 200."""
    return int(get_workload_config(config).get("lid_drift_window", 200))


def get_reindex_lid_ordered(config: Dict[str, Any]) -> bool:
    """Get whether to sort live vectors by LID descending during reindex. Default True."""
    return bool(get_workload_config(config).get("reindex_lid_ordered", True))


def get_reindex_lid_k(config: Dict[str, Any]) -> int:
    """Get k for LID recomputation during reindex (can be smaller for speed). Default 50."""
    return int(get_workload_config(config).get("reindex_lid_k", 50))
