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
