"""
dedup_query.py — Deduplication Pipeline (Hashing -> Bloom -> LSH -> DB Insert)
"""

import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Dict, Callable

def run_dedup_pipeline(
    adapter: Any,
    query_vectors: np.ndarray,
    query_ids: np.ndarray,
    minhash: Any,
    bloom: Any,
    pad_fn: Callable[[bytes, int], bytes],
    bit_width: int,
    sig_dim: int,
    search_radius: float,
    top_k: int,
    batch_size: int,
    concurrency: int,
    start_offset: int = 0,
    search_params: Dict[str, Any] = None,
    deadline: float = None,
) -> Dict[str, Any]:
    """
    Runs the complete deduplication pipeline concurrently.
    """
    n_vectors = len(query_vectors)
    
    batches = []
    for i in range(0, n_vectors, batch_size):
        b_vecs = query_vectors[i:i + batch_size]
        b_ids = query_ids[i:i + batch_size].tolist()
        batches.append((b_ids, b_vecs))

    bloom_lock = threading.Lock()
    
    # Metrics
    stats = {
        "total_processed": 0,
        "bloom_positives": 0,
        "lsh_rejected": 0,
        "inserted": 0,
        "hashing_latencies": [],
        "bf_search_latencies": [],
        "lsh_search_latencies": [],
        "insertion_latencies": [],
        "bloom_positives_per_batch": [],
        "lsh_rejected_per_batch": [],
        "pipeline_rejected": np.zeros(n_vectors, dtype=bool)
    }
    
    start_time = time.perf_counter()
    checkpoint_interval = 5000
    last_checkpoint = 0

    print(f"[Dedup] Starting streaming pipeline: {n_vectors:,} vectors in {len(batches)} batches (batch_size={batch_size}, concurrency={concurrency})")

    def process_batch(batch_tuple, batch_index):
        b_ids, b_vecs = batch_tuple
        local_metrics = {
            "processed": len(b_ids),
            "bloom_positives": 0,
            "lsh_rejected": 0,
            "inserted": 0,
            "hashing_lat": 0.0,
            "bf_search_lat": 0.0,
            "lsh_search_lat": 0.0,
            "insertion_lat": 0.0,
            "rejected_flags": [] # tuples of (global_index, is_rejected)
        }
        
        # 1. Hashing
        t0 = time.perf_counter()
        b_sigs = []
        for vec in b_vecs:
            sig = minhash.compute_signature(vec, bit_width=bit_width)
            sig = pad_fn(sig, sig_dim)
            b_sigs.append(sig)
        local_metrics["hashing_lat"] = time.perf_counter() - t0
        
        # 2. Bloom Filter Check
        t0 = time.perf_counter()
        possibly_seen = []
        with bloom_lock:
            for i, sig in enumerate(b_sigs):
                if sig in bloom:
                    possibly_seen.append(i)
        local_metrics["bf_search_lat"] = time.perf_counter() - t0
        local_metrics["bloom_positives"] = len(possibly_seen)
        
        # 3. LSH Search (only for Bloom positives)
        lsh_time = 0.0
        true_duplicates = set()
        if possibly_seen:
            candidate_sigs = [b_sigs[j] for j in possibly_seen]
            try:
                batch_hits, lsh_time = adapter.search_dedup_batch(
                    query_signatures=candidate_sigs,
                    radius=search_radius,
                    top_k=top_k,
                    params=search_params
                )
                
                # Map hits back to local indices
                for j_local, hits in zip(possibly_seen, batch_hits):
                    if len(hits) > 0:
                        true_duplicates.add(j_local)
            except Exception as e:
                print(f"[Warning] LSH search failed for batch: {e}")
        local_metrics["lsh_search_lat"] = lsh_time
        local_metrics["lsh_rejected"] = len(true_duplicates)
        
        # Populate rejected flags for ground truth recall mapping
        for local_idx, global_id in enumerate(b_ids):
            is_rejected = local_idx in true_duplicates
            local_metrics["rejected_flags"].append((global_id, is_rejected))
        
        # 4. DB & Bloom Insertion (for surviving unique vectors)
        insert_time = 0.0
        insert_ids = []
        insert_vecs = []
        insert_sigs = []
        for i in range(len(b_ids)):
            if i not in true_duplicates:
                insert_ids.append(b_ids[i])
                insert_vecs.append(b_vecs[i])
                insert_sigs.append(b_sigs[i])
                
        if insert_ids:
            # Add to Bloom filter
            with bloom_lock:
                for sig in insert_sigs:
                    bloom.add(sig)
            # Insert into database
            try:
                insert_time = adapter.insert_dedup(insert_ids, [v.tolist() for v in insert_vecs], insert_sigs)
                local_metrics["inserted"] = len(insert_ids)
            except Exception as e:
                print(f"[Warning] DB insert failed for batch: {e}")
        local_metrics["insertion_lat"] = insert_time
        
        return local_metrics

    # Execute Concurrently
    processed_count = 0
    
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for i, batch in enumerate(batches):
                futures[executor.submit(process_batch, batch, i)] = batch
                
            for future in as_completed(futures):
                if deadline and time.perf_counter() > deadline:
                    break
                try:
                    res = future.result()
                    stats["total_processed"] += res["processed"]
                    stats["bloom_positives"] += res["bloom_positives"]
                    stats["lsh_rejected"] += res["lsh_rejected"]
                    stats["inserted"] += res["inserted"]
                    stats["hashing_latencies"].append(res["hashing_lat"])
                    stats["bf_search_latencies"].append(res["bf_search_lat"])
                    stats["lsh_search_latencies"].append(res["lsh_search_lat"])
                    stats["insertion_latencies"].append(res["insertion_lat"])
                    stats["bloom_positives_per_batch"].append(res["bloom_positives"])
                    stats["lsh_rejected_per_batch"].append(res["lsh_rejected"])
                    
                    for global_id, is_rejected in res["rejected_flags"]:
                        stats["pipeline_rejected"][global_id - start_offset] = is_rejected
                        
                    processed_count += res["processed"]
                    if processed_count // checkpoint_interval > last_checkpoint:
                        last_checkpoint = processed_count // checkpoint_interval
                        print(f"[Dedup] Checkpoint: {processed_count:,} vectors processed ({processed_count*100/n_vectors:.1f}%) - Duplicates found: {stats['lsh_rejected']}")
                except Exception as e:
                    print(f"[Warning] Thread execution failed: {e}")
    else:
        for i, batch in enumerate(batches):
            if deadline and time.perf_counter() > deadline:
                break
            try:
                res = process_batch(batch, i)
                stats["total_processed"] += res["processed"]
                stats["bloom_positives"] += res["bloom_positives"]
                stats["lsh_rejected"] += res["lsh_rejected"]
                stats["inserted"] += res["inserted"]
                stats["hashing_latencies"].append(res["hashing_lat"])
                stats["bf_search_latencies"].append(res["bf_search_lat"])
                stats["lsh_search_latencies"].append(res["lsh_search_lat"])
                stats["insertion_latencies"].append(res["insertion_lat"])
                stats["bloom_positives_per_batch"].append(res["bloom_positives"])
                stats["lsh_rejected_per_batch"].append(res["lsh_rejected"])
                
                for global_id, is_rejected in res["rejected_flags"]:
                    stats["pipeline_rejected"][global_id - start_offset] = is_rejected
                    
                processed_count += res["processed"]
                if processed_count // checkpoint_interval > last_checkpoint:
                    last_checkpoint = processed_count // checkpoint_interval
                    print(f"[Dedup] Checkpoint: {processed_count:,} vectors processed ({processed_count*100/n_vectors:.1f}%) - Duplicates found: {stats['lsh_rejected']}")
            except Exception as e:
                print(f"[Warning] Sequential execution failed: {e}")

    stats["total_time"] = time.perf_counter() - start_time
    return stats
