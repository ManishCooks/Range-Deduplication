"""
metrics.py — Metrics aggregation for deduplication workload
"""

import numpy as np
from typing import List, Dict, Any, Tuple

def collect_dedup_metrics(
    base_ingest_stats: Dict[str, Any], 
    all_pass_stats: List[Dict[str, Any]], 
    gt_max_jaccard: np.ndarray, 
    jaccard_threshold: float,
    query_time_budget_s: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Aggregates statistics across passes and calculates detailed latencies and recall.
    Returns:
        (all_query_stats, metrics)
    """
    metrics = {}
    
    # 1. Base Ingestion Stats
    metrics["ingest_total_vectors"] = base_ingest_stats.get("total_vectors", 0)
    metrics["ingest_inserted"] = base_ingest_stats.get("inserted", 0)
    metrics["ingest_failed"] = base_ingest_stats.get("failed", 0)
    metrics["ingest_time_s"] = base_ingest_stats.get("total_time_s", 0.0)
    metrics["ingest_throughput_vps"] = base_ingest_stats.get("throughput_vps", 0.0)
    metrics["ingest_avg_batch_time_ms"] = base_ingest_stats.get("avg_batch_time_ms", 0.0)
    metrics["ingest_p99_batch_time_ms"] = base_ingest_stats.get("p99_batch_time_ms", 0.0)

    if not all_pass_stats:
        return {"passes": []}, metrics

    # 2. Dedup Pipeline Ground Truth (Pass 1 only)
    true_duplicate_mask = gt_max_jaccard >= jaccard_threshold
    num_true_duplicates = np.sum(true_duplicate_mask)
    num_true_uniques = len(gt_max_jaccard) - num_true_duplicates

    pass1 = all_pass_stats[0]
    pipeline_rejected = pass1.get("pipeline_rejected", np.zeros(len(gt_max_jaccard), dtype=bool))
    
    true_positives = np.sum(pipeline_rejected & true_duplicate_mask)
    false_positives = np.sum(pipeline_rejected & ~true_duplicate_mask)
    false_negatives = np.sum(~pipeline_rejected & true_duplicate_mask)
    metrics["dedup_recall"] = float(true_positives / num_true_duplicates) if num_true_duplicates > 0 else 1.0
    metrics["false_negative_rate"] = 1.0 - metrics["dedup_recall"]
    metrics["dedup_precision"] = float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 1.0
    metrics["dedup_f1"] = 2 * (metrics["dedup_recall"] * metrics["dedup_precision"]) / (metrics["dedup_recall"] + metrics["dedup_precision"]) if (metrics["dedup_recall"] + metrics["dedup_precision"]) > 0 else 0.0
    metrics["gt_positive_rate"] = float(num_true_duplicates / len(gt_max_jaccard)) if len(gt_max_jaccard) > 0 else 0.0
    metrics["gt_max_jaccard_p50"] = float(np.percentile(gt_max_jaccard, 50))
    metrics["gt_max_jaccard_p95"] = float(np.percentile(gt_max_jaccard, 95))
    metrics["gt_max_jaccard_p99"] = float(np.percentile(gt_max_jaccard, 99))
    metrics["_gt_max_jaccard"] = gt_max_jaccard.tolist()

    # 3. Aggregate Latencies and Pass Stats
    all_query_stats = {"passes": []}
    
    total_processed = 0
    total_inserted = 0
    total_lsh_rejected = 0
    total_bloom_positives = 0
    total_time = 0.0
    
    global_hashing = []
    global_bf_search = []
    global_lsh_search = []
    global_insertion = []

    for stat in all_pass_stats:
        total_processed += stat.get("total_processed", 0)
        total_inserted += stat.get("inserted", 0)
        total_lsh_rejected += stat.get("lsh_rejected", 0)
        total_bloom_positives += stat.get("bloom_positives", 0)
        total_time += stat.get("total_time", 0.0)
        
        global_hashing.extend(stat.get("hashing_latencies", []))
        global_bf_search.extend(stat.get("bf_search_latencies", []))
        global_lsh_search.extend(stat.get("lsh_search_latencies", []))
        global_insertion.extend(stat.get("insertion_latencies", []))
        
        # Calculate per-pass throughput and latencies
        pass_time_s = stat.get("total_time", 0.0)
        pass_processed = stat.get("total_processed", 0)
        
        
        # Calculate recall for this specific pass
        pass_pipeline_rejected = stat.get("pipeline_rejected", np.zeros(len(gt_max_jaccard), dtype=bool))
        pass_tp = np.sum(pass_pipeline_rejected & true_duplicate_mask)
        pass_recall = float(pass_tp / num_true_duplicates) if num_true_duplicates > 0 else 1.0
        
        per_pass = {
            "pass": stat.get("pass", 1),
            "throughput": pass_processed / pass_time_s if pass_time_s > 0 else 0,
            "dedup_recall": pass_recall,
            "_query_time": pass_time_s * 1000.0,
            "_raw_hashing": [l * 1000.0 for l in stat.get("hashing_latencies", [])],
            "_raw_bf_search": [l * 1000.0 for l in stat.get("bf_search_latencies", [])],
            "_raw_lsh_search": [l * 1000.0 for l in stat.get("lsh_search_latencies", [])],
            "_raw_insertion": [l * 1000.0 for l in stat.get("insertion_latencies", [])],
            "_raw_bloom_positives": stat.get("bloom_positives_per_batch", []),
            "_raw_lsh_rejected": stat.get("lsh_rejected_per_batch", []),
        }
        all_query_stats["passes"].append(per_pass)

    metrics["total_processed"] = total_processed
    metrics["inserted"] = total_inserted
    metrics["lsh_rejected"] = total_lsh_rejected
    metrics["duplicate_rate"] = total_lsh_rejected / total_processed if total_processed > 0 else 0.0
    metrics["bloom_false_positive_rate"] = (total_bloom_positives - total_lsh_rejected) / total_processed if total_processed > 0 else 0.0
    metrics["bloom_amplification"] = total_bloom_positives / total_lsh_rejected if total_lsh_rejected > 0 else 1.0
    metrics["throughput_vps"] = total_processed / total_time if total_time > 0 else 0.0

    def _add_lat_metrics(prefix, lat_list):
        if lat_list:
            arr = np.array(lat_list) * 1000.0 # Convert to ms
            metrics[f"{prefix}_latency_avg"] = float(np.mean(arr))
            metrics[f"{prefix}_latency_p50"] = float(np.percentile(arr, 50))
            metrics[f"{prefix}_latency_p95"] = float(np.percentile(arr, 95))
            metrics[f"{prefix}_latency_p99"] = float(np.percentile(arr, 99))
        else:
            metrics[f"{prefix}_latency_avg"] = 0.0
            metrics[f"{prefix}_latency_p50"] = 0.0
            metrics[f"{prefix}_latency_p95"] = 0.0
            metrics[f"{prefix}_latency_p99"] = 0.0

    _add_lat_metrics("hashing", global_hashing)
    _add_lat_metrics("bf_search", global_bf_search)
    _add_lat_metrics("lsh_search", global_lsh_search)
    _add_lat_metrics("insertion", global_insertion)

    metrics["_query_time"] = query_time_budget_s if query_time_budget_s is not None else total_time
    metrics["_query_passes"] = all_query_stats["passes"]

    return all_query_stats, metrics
