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
# COMMON WORKLOAD HELPERS (both workloads)
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
