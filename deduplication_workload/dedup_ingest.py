"""
dedup_ingest.py — Base Vector Ingestion for Deduplication Workload
"""

import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple, Callable

def concurrent_base_ingest(
    adapter: Any,
    base_vectors: np.ndarray,
    minhash: Any,
    pad_fn: Callable[[bytes, int], bytes],
    bit_width: int,
    sig_dim: int,
    ingest_batch_size: int,
    ingest_concurrency: int,
    lsh_mode: bool = True,
) -> Tuple[int, int, float, List[float]]:
    """
    Ingest base vectors concurrently into the deduplication index.
    In LSH Mode, computes MinHash signatures on the fly. 
    Returns the exact array of raw batch insertion latencies (ms) for accurate percentile tracking.
    """
    n_vectors = len(base_vectors)
    
    # Create batches upfront (only ids and vectors, signatures computed in threads)
    batches = []
    for i in range(0, n_vectors, ingest_batch_size):
        batch_vectors = base_vectors[i:i + ingest_batch_size]
        batch_ids = list(range(i, i + len(batch_vectors)))
        batches.append((batch_ids, batch_vectors))
        
    start_time = time.perf_counter()
    inserted_total = 0
    failed_total = 0
    batch_times = []
    
    checkpoint_interval = ingest_batch_size
    last_checkpoint = 0
    
    if lsh_mode:
        print(f"[Ingest] Computing MinHash signatures on the fly...")
    else:
        print(f"[Ingest] Preparing raw vectors for ingestion...")

    def serial_insert(batch):
        b_ids, b_vecs = batch
        t0 = time.perf_counter()
        
        # 1. Compute signatures on the fly (CPU intensive, now parallelized)
        if lsh_mode:
            b_sigs = []
            for vec in b_vecs:
                sig = minhash.compute_signature(vec, bit_width=bit_width)
                sig = pad_fn(sig, sig_dim)
                b_sigs.append(sig)
            # 2. Insert into the database
            elapsed = 0.0
            try:
                insert_time_ms = adapter.insert_dedup(b_ids, b_vecs.tolist(), b_sigs)
                elapsed = insert_time_ms / 1000.0  # Keep it in seconds for batch_times
                inserted_count = len(b_ids)
                failed_count = 0
            except Exception as e:
                print(f"[Warning] Failed to insert base batch: {e}")
                inserted_count = 0
                failed_count = len(b_ids)
        else:
            elapsed = 0.0
            try:
                insert_time_ms = adapter.insert_dense_dedup(b_ids, b_vecs.tolist())
                elapsed = insert_time_ms / 1000.0
                inserted_count = len(b_ids)
                failed_count = 0
            except Exception as e:
                print(f"[Warning] Failed to dense-insert base batch: {e}")
                inserted_count = 0
                failed_count = len(b_ids)
                
        return inserted_count, failed_count, elapsed

    if ingest_concurrency > 1:
        with ThreadPoolExecutor(max_workers=ingest_concurrency) as executor:
            futures = {executor.submit(serial_insert, batch): batch for batch in batches}
            for future in as_completed(futures):
                inserted, failed, elapsed = future.result()
                batch_times.append(elapsed)
                inserted_total += inserted
                failed_total += failed
                
                if inserted_total // checkpoint_interval > last_checkpoint:
                    last_checkpoint = inserted_total // checkpoint_interval
                    elapsed_total = time.perf_counter() - start_time
                    print(f"[Ingest] Checkpoint: {inserted_total:,} vectors inserted ({inserted_total*100/n_vectors:.1f}%) - Elapsed: {elapsed_total:.2f}s")
    else:
        for batch in batches:
            inserted, failed, elapsed = serial_insert(batch)
            batch_times.append(elapsed)
            inserted_total += inserted
            failed_total += failed
            
            if inserted_total // checkpoint_interval > last_checkpoint:
                last_checkpoint = inserted_total // checkpoint_interval
                elapsed_total = time.perf_counter() - start_time
                print(f"[Ingest] Checkpoint: {inserted_total:,} vectors inserted ({inserted_total*100/n_vectors:.1f}%) - Elapsed: {elapsed_total:.2f}s")

    total_time = time.perf_counter() - start_time
    batch_times_ms = [t * 1000.0 for t in batch_times]
    
    return inserted_total, failed_total, total_time, batch_times_ms
