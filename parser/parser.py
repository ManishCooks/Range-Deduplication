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

def use_mmap(config: Dict[str, Any]) -> bool:
    """Get whether to use mmap from global."""
    return get_global_config(config).get("use_mmap", False)


def get_read_concurrency(config: Dict[str, Any]) -> int:
    """Get read concurrency from global."""
    global_cfg = get_global_config(config)
    return global_cfg.get("read_concurrency", 4)


def get_ingest_concurrency(config: Dict[str, Any]) -> int:
    """Get write concurrency from global."""
    global_cfg = get_global_config(config)
    return global_cfg.get("ingest_concurrency", 4)

def get_ingest_batch_size(config: Dict[str, Any]) -> int: 
    """Get ingest batch size from global."""
    return get_global_config(config).get("ingest_batch_size", 1500)


def get_query_batch_size(config: Dict[str, Any]) -> int:
    """Get query batch size from global."""
    return get_global_config(config).get("query_batch_size", 500)

def get_drop_collection_first(config: Dict[str, Any]) -> bool:
    """Get whether to drop collection first from global."""
    return get_global_config(config).get("drop_collection_first", True)

def get_monitor_system(config: Dict[str, Any]) -> bool:
    """Get whether to monitor system metrics during workload execution."""
    return get_global_config(config).get("monitor_system", False)

def get_containers(config: Dict[str, Any]) -> list:
    """Get list of Docker container names/IDs to monitor."""
    containers = get_global_config(config).get("containers", [])
    if isinstance(containers, str):
        return [containers]
    elif isinstance(containers, list):
        return containers
    else:
        raise ConfigError("Invalid 'containers' format in global config; must be string or list of strings.")

def system_metrics(config: Dict[str, Any]) -> Dict[str, list]:
    """Get dict specifying which system and docker metrics to collect."""
    return get_global_config(config).get("system_metrics", {"system": [], "docker": []})

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

def get_query_time_budget_s(config: Dict[str, Any]) -> Union[float, None]:
    """Get optional time budget for query phase in seconds."""
    val = get_workload_config(config).get("query_time_budget_s", None)
    return float(val) if val is not None else None

def get_query_vectors_path(config: Dict[str, Any]) -> str:
    """Get path to separate query vectors file."""
    return get_workload_config(config).get("query_vectors", None)

def get_query_ratio(config: Dict[str, Any]) -> float:
    """Get query ratio (fraction of dataset for queries)."""
    return get_workload_config(config).get("query_ratio", 0.01)

def get_query_vectors_path(config: Dict[str, Any]) -> str:
    """Get path to separate query vectors file."""
    return get_workload_config(config).get("query_vectors", None)

def get_query_vectors_path(config: Dict[str, Any]) -> str:
    """Get path to separate query vectors file."""
    return get_workload_config(config).get("query_vectors", None)


def get_metrics(config: Dict[str, Any]) -> list:
    """Get list of metrics to collect."""
    return get_workload_config(config).get("metrics", ["latency_p50", "latency_p95", "qps"])
# =============================================================================
# DEDUPLICATION WORKLOAD HELPERS
# =============================================================================

def get_top_k(config: Dict[str, Any]) -> int:
    """Get top-k for deduplication search."""
    return int(get_workload_config(config).get("top_k", 1))

def get_num_perm(config: Dict[str, Any]) -> int:
    """Get number of permutations for MinHash LSH."""
    return int(get_workload_config(config).get("num_perm", 128))


def get_jaccard_threshold(config: Dict[str, Any]) -> float:
    """Get Jaccard similarity threshold for deduplication."""
    return float(get_workload_config(config).get("jaccard_threshold", 0.8))


def get_bloom_capacity(config: Dict[str, Any]) -> int:
    """Get estimated number of unique items for Bloom Filter."""
    return int(get_workload_config(config).get("bloom_capacity", 100000))


def get_bloom_error_rate(config: Dict[str, Any]) -> float:
    """Get accepted false positive rate for Bloom Filter."""
    return float(get_workload_config(config).get("bloom_error_rate", 0.01))


