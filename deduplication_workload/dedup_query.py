"""
dedup_query.py — Deduplication Pipeline (Hashing -> Range Search -> DB Insert)

Design Decisions:
- Concurrency Branching: Dynamically branches based on the `concurrency` setting. 
  - When `concurrency <= 1`: Executes single-vector per-query adapter calls (`search_dedup`/`search_range`) to capture authentic per-vector execution latencies straight from the database.
  - When `concurrency > 1`: Uses batch search calls (`search_dedup_batch`/`search_range_batch`) to minimize network overhead, capturing aggregated batch wall-clock latency.
- Latency Aggregation: Hashing, search, and insertion latencies are collected as discrete arrays instead of summed scalars, permitting accurate percentile metrics across the run.
- Mode Gating: 
  - LSH Mode: Computes MinHash signatures on the fly and utilizes Jaccard-based radius searches.
  - Dense Mode: Skips hashing entirely and executes standard L2/Cosine range searches.
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
    lsh_mode: bool = True,
) -> Dict[str, Any]:
    """
    Runs the complete deduplication pipeline. 
    Divides the query set into batches, executing the hashing, exact match lookup, and 
    conditional insertion steps. Branches to parallelized thread-pool execution if concurrency > 1, 
    otherwise runs fully sequentially to capture pristine latency traces.
    """
    n_vectors = len(query_vectors)
    
    batches = []
    for i in range(0, n_vectors, batch_size):
        b_vecs = query_vectors[i:i + batch_size]
        b_ids = query_ids[i:i + batch_size].tolist()
        batches.append((b_ids, b_vecs))

    stats = {
        "total_processed": 0,
        "lsh_rejected": 0,
        "inserted": 0,
        "hashing_latencies": [],
        "lsh_search_latencies": [],
        "insertion_latencies": [],
        "lsh_rejected_per_batch": [],
        "pipeline_rejected": np.zeros(n_vectors, dtype=bool)
    }
    
    start_time = time.perf_counter()
    checkpoint_interval = 5000
    last_checkpoint = 0

    print(f"[Dedup] Starting streaming pipeline: {n_vectors:,} vectors in {len(batches)} batches (batch_size={batch_size}, concurrency={concurrency})")

    if lsh_mode:
        if not hasattr(adapter, "search_dedup_batch"):
            raise AttributeError(
                "LSH mode requires adapter.search_dedup_batch(). "
                f"Adapter '{type(adapter).__name__}' does not support it."
            )
    else:
        if not hasattr(adapter, "search_range_batch"):
            raise AttributeError(
                "Dense mode requires adapter.search_range_batch(). "
                f"Adapter '{type(adapter).__name__}' does not support it."
            )

    def process_batch(batch_tuple, batch_index):
        b_ids, b_vecs = batch_tuple
        local_metrics = {
            "processed": len(b_ids),
            "lsh_rejected": 0,
            "inserted": 0,
            "hashing_lat": [],
            "lsh_search_lat": [],
            "insertion_lat": [],
            "rejected_flags": [] # tuples of (global_index, is_rejected)
        }
        true_duplicates = set()

        if lsh_mode:
            # ---- Step 1: MinHash ----
            t0 = time.perf_counter()
            b_sigs = []
            for vec in b_vecs:
                sig = minhash.compute_signature(vec, bit_width=bit_width)
                sig = pad_fn(sig, sig_dim)
                b_sigs.append(sig)
            local_metrics["hashing_lat"].append((time.perf_counter() - t0) * 1000)

            # ---- Step 2: LSH Search ----
            if b_sigs:
                try:
                    if concurrency <= 1:
                        # Serial mode: query one-by-one to get per-query DB latencies
                        batch_hits = []
                        for sig in b_sigs:
                            hits, q_lat = adapter.search_dedup(
                                query_signature=sig,
                                radius=search_radius,
                                top_k=top_k,
                                params=search_params
                            )
                            batch_hits.append(hits)
                            local_metrics["lsh_search_lat"].append(q_lat)
                    else:
                        # Batch mode: single network call
                        batch_hits, lsh_time = adapter.search_dedup_batch(
                            query_signatures=b_sigs,
                            radius=search_radius,
                            top_k=top_k,
                            params=search_params
                        )
                        local_metrics["lsh_search_lat"].append(lsh_time)

                    # Map hits back to local indices
                    if b_ids[0] == 0:  # Print for the very first query to debug
                        print(f"\n[DEBUG LSH] Top {len(batch_hits[0])} hits for first query:")
                        for idx_debug, hit_debug in enumerate(batch_hits[0][:5]):
                            print(f"   -> Hit {idx_debug}: ID={hit_debug['id']}, Distance={hit_debug['distance']}")

                    for j_local, hits in enumerate(batch_hits):
                        if len(hits) > 0:
                            true_duplicates.add(j_local)
                except Exception as e:
                    print(f"[Warning] LSH search failed for batch {batch_index}: {e}")

            # ---- Step 3: Insert (LSH path: float vec + signature) ----
            insert_ids, insert_vecs, insert_sigs = [], [], []
            for i in range(len(b_ids)):
                if i not in true_duplicates:
                    insert_ids.append(b_ids[i])
                    insert_vecs.append(b_vecs[i])
                    insert_sigs.append(b_sigs[i])
            if insert_ids:
                try:
                    insert_time = adapter.insert_dedup(
                        insert_ids, [v.tolist() for v in insert_vecs], insert_sigs
                    )
                    local_metrics["inserted"] = len(insert_ids)
                    local_metrics["insertion_lat"].append(insert_time)
                except Exception as e:
                    print(f"[Warning] DB insert failed for batch {batch_index}: {e}")

        else:
            # ---- Dense Mode: no hashing ----
            # Step 1 + 2: Range search on raw float vectors
            try:
                if concurrency <= 1:
                    batch_hits = []
                    for vec in b_vecs:
                        hits, q_lat = adapter.search_range(
                            query_vector=vec.tolist(),
                            radius=search_radius,
                            top_k=top_k,
                            params=search_params
                        )
                        batch_hits.append(hits)
                        local_metrics["lsh_search_lat"].append(q_lat)
                else:
                    batch_hits, lsh_time = adapter.search_range_batch(
                        query_vectors=[v.tolist() for v in b_vecs],
                        radius=search_radius,
                        top_k=top_k,
                        params=search_params,
                    )
                    local_metrics["lsh_search_lat"].append(lsh_time)
                    
                for j_local, hits in enumerate(batch_hits):
                    if len(hits) > 0:
                        true_duplicates.add(j_local)
            except Exception as e:
                print(f"[Warning] Dense range search failed for batch {batch_index}: {e}")

            # ---- Step 3: Insert (Dense path: float vec only) ----
            insert_ids, insert_vecs = [], []
            for i in range(len(b_ids)):
                if i not in true_duplicates:
                    insert_ids.append(b_ids[i])
                    insert_vecs.append(b_vecs[i])
            if insert_ids:
                try:
                    insert_time = adapter.insert_dense_dedup(
                        insert_ids, [v.tolist() for v in insert_vecs]
                    )
                    local_metrics["inserted"] = len(insert_ids)
                    local_metrics["insertion_lat"].append(insert_time)
                except Exception as e:
                    print(f"[Warning] Dense insert failed for batch {batch_index}: {e}")

        local_metrics["lsh_rejected"] = len(true_duplicates)
        # Populate rejected flags for ground truth recall mapping
        for local_idx, global_id in enumerate(b_ids):
            local_metrics["rejected_flags"].append((global_id, local_idx in true_duplicates))

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
                    stats["lsh_rejected"] += res["lsh_rejected"]
                    stats["inserted"] += res["inserted"]
                    stats["hashing_latencies"].extend(res["hashing_lat"])
                    stats["lsh_search_latencies"].extend(res["lsh_search_lat"])
                    stats["insertion_latencies"].extend(res["insertion_lat"])
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
                stats["lsh_rejected"] += res["lsh_rejected"]
                stats["inserted"] += res["inserted"]
                stats["hashing_latencies"].extend(res["hashing_lat"])
                stats["lsh_search_latencies"].extend(res["lsh_search_lat"])
                stats["insertion_latencies"].extend(res["insertion_lat"])
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
