"""
Two-Phase Vector Deduplication Workload

Flow:
1. Load dataset & normalize
2. Create initial index
3. Ingest base vectors
4. Deduplication phase:
   - Phase 1: Bloom Filter (probabilistic pre-filter)
   - Phase 2: Milvus LSH Search (near-duplicate confirmation)
5. Collect metrics
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_hdf5_dataset, normalize_vectors
from orchestrator.operations.metrics import collect_metrics
from workloads.deduplication_workload.bloom_filter import BloomFilter, SignatureMinHash


def run_workload(config: Dict[str, Any], adapter) -> Dict[str, Any]:
    # --- Config Extraction ---
    global_config   = config["global"]
    workload_config = config.get("workload",{})
    index_config    = config.get("index",{})

    dataset_name = global_config["dataset"]
    seed         = global_config.get("seed",42)
    batch_size   = global_config.get("batch_size",1000)

    drop_collection_first = workload_config.get("drop_collection_first",True)
    initial_ingest_ratio  = workload_config.get("initial_ingest_ratio",0.5)

    # Phase 1 — Bloom Filter
    bloom_capacity   = workload_config.get("bloom_capacity",100_000)
    bloom_error_rate = workload_config.get("bloom_error_rate",0.01)

    # Phase 2 — MinHash LSH
    top_k             = workload_config.get("top_k",1)
    num_perm          = workload_config.get("num_perm",128)
    jaccard_threshold = workload_config.get("jaccard_threshold",0.8)

    default_metrics = [
        "total_processed",
        "bloom_rejected",
        "lsh_rejected",
        "inserted",
        "duplicate_rate",
        "bloom_false_positive_rate",
        "ingest_latency_p50",
        "ingest_latency_p95",
        "ingest_latency_p99",
        "phase1_latency_avg",
        "phase2_latency_avg",
        "throughput_vps",
    ]
    metrics_to_collect = workload_config.get("metrics", default_metrics)

    index_type = index_config.get("type",   "BIN_FLAT")
    metric     = index_config.get("metric", "JACCARD")
    params     = index_config.get("params", {})

    print(f"\n{'='*60}")
    print("WORKLOAD: Two-Phase Vector Deduplication")
    print(f"{'='*60}")

    # =========================================================================
    # 1. Load & Prepare Data
    # =========================================================================
    print("[1/5] Loading dataset...")
    data    = load_hdf5_dataset(dataset_name, load_train=True, load_test=False)
    vectors = data["train"].astype(np.float32)

    if len(vectors) == 0:
        raise ValueError(f"Dataset '{dataset_name}' has no train vectors.")

    split_idx     = int(len(vectors) * initial_ingest_ratio)
    base_vectors  = vectors[:split_idx]
    dedup_vectors = vectors[split_idx:]

    if metric in ("cosine", "inner_product"):
        print("[2/5] Normalizing vectors...")
        base_vectors  = normalize_vectors(base_vectors)
        dedup_vectors = normalize_vectors(dedup_vectors)
    else:
        print("[2/5] Skipping normalization (non-cosine metric)")

    # =========================================================================
    # 3. Create Index
    # =========================================================================
    print("[3/5] Creating index...")
    vector_dim    = base_vectors.shape[1]
    signature_dim = (num_perm + 7) // 8 * 8  # round up to multiple of 8

    adapter.create_dedup_collection(
        vector_dim=vector_dim,
        signature_dim=signature_dim,
        drop_existing=drop_collection_first,
    )
    
    try:
        from pymilvus import Collection
        coll = Collection(adapter._config.collection, using=adapter._connection_alias)
        
        # 1. Create a dummy index on the main vector field so Milvus allows us to load the collection
        coll.create_index(
            field_name="vector",
            index_params={
                "index_type": "FLAT",
                "metric_type": "COSINE",
                "params": {}
            }
        )
        
        # 2. Create the actual index on the signature field
        coll.create_index(
            field_name="signature",
            index_params={
                "index_type":  index_type,
                "metric_type": metric,
                "params":      params,
            }
        )
        coll.load()
        print(f"[Milvus] Signature index ({index_type}) created and collection loaded.")
    except Exception as e:
        print(f"[Warning] Failed to create index on signature: {e}. Will rely on brute-force.")

    # =========================================================================
    # 4. Ingest Base Vectors
    # =========================================================================
    print(f"\n{'='*60}")
    print("[4/5] INITIAL INGESTION PHASE STARTING")
    print(f"[Ingest] Inserting {len(base_vectors):,} base vectors (batch_size={batch_size})")

    bloom  = BloomFilter(bloom_capacity, bloom_error_rate)
    minhash = SignatureMinHash(num_perm=signature_dim, seed=seed)

    # Pre-compute signatures for base vectors and populate bloom filter
    base_signatures = []
    for vec in base_vectors:
        sig = minhash.compute_signature(vec)
        sig = _pad_signature(sig, signature_dim)
        base_signatures.append(sig)
        bloom.add(sig)

    ingest_start_time = time.perf_counter()
    inserted_base = 0
    failed_base = 0
    batch_times = []

    for i in range(0, len(base_vectors), batch_size):
        batch = base_vectors[i:i + batch_size]
        batch_sig = base_signatures[i:i + batch_size]
        batch_ids = list(range(i, i + len(batch)))
        t0 = time.perf_counter()
        try:
            adapter.insert_dedup(batch_ids, batch.tolist(), batch_sig)
            inserted_base += len(batch_ids)
        except Exception as e:
            print(f"[Warning] Failed to insert base batch: {e}")
            failed_base += len(batch_ids)
        batch_times.append(time.perf_counter() - t0)

    total_time = time.perf_counter() - ingest_start_time
    base_ingest_stats = {
        "total_vectors": len(base_vectors),
        "inserted": inserted_base,
        "failed": failed_base,
        "total_time_s": total_time,
        "throughput_vps": inserted_base / total_time if total_time > 0 else 0,
        "avg_batch_time_ms": float(np.mean(batch_times)) * 1000 if batch_times else 0.0,
        "p99_batch_time_ms": float(np.percentile(batch_times, 99)) * 1000 if batch_times else 0.0
    }
    print(f"[Ingest] Done: {inserted_base} inserted, {failed_base} failed in {total_time:.2f}s ({base_ingest_stats['throughput_vps']:.0f} vec/s)")

    # =========================================================================
    # 5. Deduplication Phase
    # =========================================================================
    print(f"\n{'='*60}")
    print("[5/5] DEDUPLICATION PHASE STARTING")
    print(f"[Dedup] Processing {len(dedup_vectors):,} remaining vectors")

    print("Precomputing MinHash signatures...")
    #batch computing minhash for dedup vectors
    all_sigs = []
    for vec in dedup_vectors:
        sig = minhash.compute_signature(vec)
        sig = _pad_signature(sig, signature_dim)
        all_sigs.append(sig)
    print(f"[Dedup] Signatures precomputed for {len(dedup_vectors):,} vectors")

    state = {
        "total_processed":  0,
        "bloom_positives":  0,
        "lsh_rejected":     0,
        "inserted":         0,
        "phase1_latencies": [],
        "phase2_latencies": [],
        "ingest_latencies": [],
        "dedup_progress":   [],
    }

    checkpoint_interval    = 5000
    last_checkpoint        = 0
    dedup_start_time       = time.perf_counter()
    offset                 = split_idx

    for i in range(0, len(dedup_vectors), batch_size):
        batch      = dedup_vectors[i:i + batch_size]
        batch_ids  = list(range(offset + i, offset + i + len(batch)))
        batch_sigs = [_pad_signature(s, signature_dim) for s in all_sigs[i:i + batch_size]]

        batch_t0 = time.perf_counter()

        # ── Phase 1: Bloom Filter (entire batch) ──────────────────────────
        t0 = time.perf_counter()
        bloom_results   = [sig in bloom for sig in batch_sigs]
        possibly_seen   = [i for i, hit in enumerate(bloom_results) if hit]
        definitely_new  = [i for i, hit in enumerate(bloom_results) if not hit]
        t1 = time.perf_counter()
        state["phase1_latencies"].append((t1 - t0) * 1000.0)
        state["bloom_positives"] += len(possibly_seen)
        state["total_processed"] += len(batch)

        # ── Phase 2: Milvus LSH Batch Search (bloom positives only) ───────
        lsh_duplicate_indices = set()
        if possibly_seen:
            t2 = time.perf_counter()
            candidate_sigs = [batch_sigs[j] for j in possibly_seen]
            radius         = 1.0 - jaccard_threshold
            batch_hits     = adapter.search_dedup_batch(
                query_signatures=candidate_sigs,
                radius=radius,
                top_k=top_k,
            )  # returns list of hit lists, one per query
            t3 = time.perf_counter()
            state["phase2_latencies"].append((t3 - t2) * 1000.0)

            for local_idx, (orig_idx, hits) in enumerate(zip(possibly_seen, batch_hits)):
                is_duplicate = any(hit["distance"] <= radius for hit in hits)
                state["dedup_progress"].append((state["total_processed"], is_duplicate))
                if is_duplicate:
                    state["lsh_rejected"] += 1
                    lsh_duplicate_indices.add(orig_idx)

        # ── Queue for insert: definitely new + bloom FP ───────────────────
        insert_ids        = []
        insert_vecs       = []
        insert_signatures = []

        # definitely_new — skipped bloom, go straight to insert
        for j in definitely_new:
            insert_ids.append(batch_ids[j])
            insert_vecs.append(batch[j].tolist())
            insert_signatures.append(batch_sigs[j])
            bloom.add(batch_sigs[j])

        # bloom positives that were NOT confirmed as duplicates by LSH (FPs)
        for j in possibly_seen:
            if j not in lsh_duplicate_indices:
                insert_ids.append(batch_ids[j])
                insert_vecs.append(batch[j].tolist())
                insert_signatures.append(batch_sigs[j])
                bloom.add(batch_sigs[j])

        # ── Batch Insert ──────────────────────────────────────────────────
        if insert_ids:
            try:
                adapter.insert_dedup(insert_ids, insert_vecs, insert_signatures)
                state["inserted"] += len(insert_ids)
            except Exception as e:
                print(f"[Warning] Failed to insert batch: {e}")

        batch_t1 = time.perf_counter()
        state["ingest_latencies"].append((batch_t1 - batch_t0) * 1000.0)

        processed = state["total_processed"]
        if processed // checkpoint_interval > last_checkpoint:
            last_checkpoint = processed // checkpoint_interval
            elapsed  = time.perf_counter() - dedup_start_time
            pct      = processed * 100 / len(dedup_vectors)
            dup_rate = state["lsh_rejected"] / processed if processed > 0 else 0.0
            print(
                f"[Dedup] Checkpoint: {processed:,} vectors ({pct:.1f}%) — "
                f"inserted={state['inserted']:,}  duplicates={state['lsh_rejected']:,}  "
                f"dup_rate={dup_rate:.3f}  elapsed={elapsed:.2f}s"
            )

    dedup_total_s = time.perf_counter() - dedup_start_time

    # =========================================================================
    # 6. Metrics
    # =========================================================================
    print("[6/5] Collecting metrics...")

    total          = state["total_processed"]
    lsh_rejected   = state["lsh_rejected"]
    bloom_pos      = state["bloom_positives"]
    bloom_fp       = bloom_pos - lsh_rejected
    total_neg      = total - lsh_rejected

    ingest_lat_arr  = np.array(state["ingest_latencies"],  dtype=np.float32)
    phase1_lat_arr  = np.array(state["phase1_latencies"],  dtype=np.float32)
    phase2_lat_arr  = np.array(state["phase2_latencies"],  dtype=np.float32)

    metrics = collect_metrics(
        ingest_stats=base_ingest_stats,
        query_stats={},
        monitor_stats={},
        ground_truth=None,
        k=None,
        metrics_to_collect=metrics_to_collect,
    )

    metrics.update({
        "workload":                  "ood_workload",
        "total_processed":           total,
        "bloom_rejected":            0,          # bloom only routes, never hard-rejects
        "lsh_rejected":              lsh_rejected,
        "inserted":                  state["inserted"],
        "duplicate_rate":            lsh_rejected / total if total > 0 else 0.0,
        "bloom_false_positive_rate": bloom_fp / total_neg if total_neg > 0 else 0.0,

        "ingest_latency_p50": float(np.percentile(ingest_lat_arr, 50)) if len(ingest_lat_arr) else None,
        "ingest_latency_p95": float(np.percentile(ingest_lat_arr, 95)) if len(ingest_lat_arr) else None,
        "ingest_latency_p99": float(np.percentile(ingest_lat_arr, 99)) if len(ingest_lat_arr) else None,

        "phase1_latency_avg": float(np.mean(phase1_lat_arr)) if len(phase1_lat_arr) else None,
        "phase2_latency_avg": float(np.mean(phase2_lat_arr)) if len(phase2_lat_arr) else None,
        "throughput_vps":     total / dedup_total_s if dedup_total_s > 0 else None,

        "final_collection_size": split_idx + state["inserted"],

        # Private — not plotted
        "_phase1_latencies": state["phase1_latencies"],
        "_phase2_latencies": state["phase2_latencies"],
        "_ingest_latencies": state["ingest_latencies"],

        "_dedup_progress":   state["dedup_progress"],     # feeds duplicate rate plot
        "bloom_capacity":    bloom_capacity,              # feeds bloom FP curve
        "bloom_error_rate":  bloom_error_rate,            # feeds bloom FP curve
    })

    adapter.flush()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    return metrics


# =============================================================================
# Helpers
# =============================================================================

def _pad_signature(sig: bytes, signature_dim: int) -> bytes:
    expected = signature_dim // 8
    if len(sig) < expected:
        return sig + b'\x00' * (expected - len(sig))
    return sig[:expected]