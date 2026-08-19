"""
Plotting - Automated benchmark result plots.
Sits alongside query.py so any workload can call generate_plots()
after a run.

Plots generated (depending on which keys are present in `stats`):
    latency_cdf.png          — CDF with p50/p95/p99/p99.9/p99.99 markers
    latency_per_modality.png — Mean + P99 bars per query modality
    size_per_partition.png   — Entity count per partition
    system_metrics.png       — Dual-axis CPU% + RAM timeline
    sparse_timeline.png      — Wall-clock seconds vs query latency (sparse workload)
    warmup_curve.png         — Per-cycle latency decay after cold start
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive, works headless
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import numpy as np
# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_plots(
    stats:            Dict[str, Any],
    latencies:        List[float],
    output_dir:       Path,
    monitor_timeline: Optional[Dict[str, List[float]]] = None,
) -> List[str]:
    """
    Generate all applicable plots and save PNGs to `output_dir`.

    Args:
        stats:            The stats dict returned by run_workload / run_queries.
        latencies:        Raw per-query latency list in ms.
        output_dir:       Directory to write PNGs into (created if absent).
        monitor_timeline: Optional {"cpu_sys", "mem_sys", "cpu_db", "mem_db"}
                          time-series lists from SystemMonitor.
    Returns:
        List of absolute paths to generated PNG files.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Plots] matplotlib not installed — skipping. Run: pip install matplotlib")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[str] = []
    #complete ingestion workload
    # Per-pass CDF: triggered by _raw_latencies present in per-pass stats.
    # Global CDF: triggered by the latencies list passed in directly.
    raw_lats = stats.get("_raw_latencies")
    pass_num = stats.get("pass")        # present in per-pass stats, None otherwise
    cdf_source = raw_lats if raw_lats else (latencies if latencies else None)
    if cdf_source:
        p = _plot_latency_cdf(cdf_source, output_dir, pass_num=pass_num)
        if p:
            generated.append(p)
    if "_query_passes" in stats:
        p = _plot_pass_qps_vs_recall(stats, output_dir)
        if p: generated.append(p)
    
    # RWD Based workload plots
    if stats.get("workload") in ["rwd_workload", "streaming_ingestion_workload", "burst_rwd_workload" ,"lid_workload"]:
        if "_raw_latencies" in stats:
            p = _plot_rwd_latency_over_time(stats, output_dir)
            if p: generated.append(p)
        if "_monitor_events" in stats:
            p = _plot_rwd_drift_over_time(stats, output_dir)
            if p: generated.append(p)
        if "_query_passes" in stats:
            p = _plot_rwd_recall_over_time(stats, output_dir)
            if p: generated.append(p)
            p = _plot_rwd_query_percentiles_over_time(stats, output_dir)
            if p: generated.append(p)
        if "_mutation_events" in stats:
            p = _plot_rwd_mutation_latency_over_time(stats, output_dir)
            if p: generated.append(p)
        

    if "_burst_timeline" in stats:
        p = _plot_burst_qps_vs_latency(stats, output_dir)
        if p: generated.append(p)
        p = _plot_phase_isolated_cdfs(stats, output_dir)
        if p: generated.append(p)
        p = _plot_burst_qps_over_time(stats, output_dir)
        if p: generated.append(p)
        p = _plot_burst_phase_boxplots(stats, output_dir)
        if p: generated.append(p)

    if stats.get("workload") == "deduplication_workload":
        if "_gt_nn_score" in stats:
            p = _plot_dedup_gt_histogram(stats, output_dir)
            if p: generated.append(p)
            
        if "_query_passes" in stats:
            p = _plot_dedup_recall_vs_pass(stats, output_dir)
            if p: generated.append(p)

        # Per-pass plots
        if "_raw_hashing" in stats and "_raw_lsh_search" in stats:
            p = _plot_dedup_latency_timeline(stats, output_dir)
            if p: generated.append(p)
            
            p = _plot_dedup_latency_cdfs(stats, output_dir)
            if p: generated.extend(p)
            
        if "_raw_bloom_positives" in stats:
            p = _plot_dedup_bloom_positives_timeline(stats, output_dir)
            if p: generated.append(p)

    if stats.get("workload") == "sparse_workload":
        paths = _plot_sparse_workload_metrics(stats, output_dir)
        if paths: generated.extend(paths)

        if "_per_cycle" in stats:
            p = _plot_warmup_curve(stats, output_dir)
            if p:
                generated.append(p)

    if "latency_per_modality" in stats:
        p = _plot_latency_per_modality(stats["latency_per_modality"], output_dir)
        if p:
            generated.append(p)

    if "size_per_partition" in stats:
        p = _plot_size_per_partition(stats["size_per_partition"], output_dir)
        if p:
            generated.append(p)

    
    if "_umap_events" in stats:
        p = _plot_umap_drift_events(stats["_umap_events"], output_dir)
        if p: generated.extend(p)  

    if monitor_timeline:
        p = _plot_process_timeline(monitor_timeline, output_dir)
        if p:
            generated.append(p)
        p = _plot_docker_timeline(monitor_timeline, output_dir)
        if p:
            generated.append(p)
            
        p = _plot_disk_usage(monitor_timeline, output_dir)
        if p:
            generated.append(p)
            
    if "_access_histogram" in stats:
        p = _plot_access_histogram(stats["_access_histogram"], output_dir)
        if p:
            generated.append(p)
    if "hot_latency_p50" in stats or "cold_latency_p50" in stats:
        p = _plot_hot_cold_latency(stats, output_dir)
        if p:
            generated.append(p)

    if stats.get("workload") == "hot_cold_workload":        
        if "_query_passes" in stats:
            p = _plot_hot_cold_recall_over_time(stats, output_dir)
            if p:
                generated.append(p)
                
            p = _plot_hot_cold_pgfaults_overall(stats, output_dir)
            if p:
                generated.append(p)
            p = _plot_hot_cold_pgfaults_pass(stats, output_dir)
            if p:
                generated.append(p) 

    if "ood_freq_distribution" in stats:
        p = _plot_ood_freq(stats["ood_freq_distribution"], output_dir)
        if p:
            generated.append(p)

    if "ood_latency_p50" in stats or "id_latency_p50" in stats:
        p = _plot_ood_vs_id_latency(stats, output_dir)
        if p:
            generated.append(p)

    # Outlier workload: latency comparison (normal vs outlier queries)
    if "normal_latency_p50_ms" in stats or "outlier_latency_p50_ms" in stats:
        p = _plot_id_vs_outlier_latency(stats, output_dir)
        if p:
            generated.append(p)

    # Outlier workload: recall comparison (normal vs outlier queries)
    if "normal_recall" in stats or "outlier_recall" in stats:
        p = _plot_id_vs_outlier_recall(stats, output_dir)
        if p:
            generated.append(p)

    # Temporal freshness workload plots
    if "avg_search_latency_ms" in stats and "avg_rerank_latency_ms" in stats:
        p = _plot_freshness_rerank_breakdown(stats, output_dir)
        if p: generated.append(p)

    if "_ndcg_histogram" in stats:
        p = _plot_freshness_ndcg_histogram(stats, output_dir)
        if p: generated.append(p)

    if "_age_composition_pct" in stats:
        p = _plot_freshness_age_composition(stats, output_dir)
        if p: generated.append(p)

    # Cold-start workload: timeline, warmup curve, cold-vs-warm comparison, TTFSQ
    if stats.get("workload") == "cold_start_workload":
        if "_query_timestamps" in stats and "_raw_latencies" in stats:
            p = _plot_cold_start_timeline(stats, output_dir)
            if p:
                generated.append(p)
        if "_per_cycle" in stats:
            p = _plot_cold_start_warmup_curve(stats, output_dir)
            if p:
                generated.append(p)
        if "cold_start_latency_p50" in stats and "warm_latency_p50" in stats:
            p = _plot_cold_warm_latency(stats, output_dir)
            if p:
                generated.append(p)
        if "_per_cycle" in stats:
            p = _plot_ttfsq_and_cold_latency_per_cycle(stats, output_dir)
            if p:
                generated.append(p)
    if stats.get("workload") == "deduplication":
        if "_phase1_latencies" in stats or "_phase2_latencies" in stats:
            p = _plot_dedup_phase_latency_distribution(stats, output_dir)
            if p: generated.append(p)
 
        if "_dedup_progress" in stats:
            p = _plot_dedup_duplicate_rate_progress(stats, output_dir)
            if p: generated.append(p)
 
        if "bloom_capacity" in stats and "bloom_error_rate" in stats:
            p = _plot_dedup_bloom_fp_curve(stats, output_dir)
            if p: generated.append(p)
 
        if "ingest_latency_p50" in stats:
            p = _plot_dedup_ingest_latency_percentiles(stats, output_dir)
            if p: generated.append(p)

    if stats.get("workload") == "ood_workload":
            if "_raw_id_latencies" in stats or "_raw_ood_latencies" in stats:
                p = _plot_ood_id_latency_cdf(stats, output_dir)
                if p: generated.append(p)
    
            if "_confusion_matrix" in stats:
                p = _plot_ood_confusion_heatmap(stats, output_dir)
                if p: generated.append(p)


    print(f"[Plots] {len(generated)} plot(s) saved to: {output_dir}")
    for f in generated:
        print(f"        {Path(f).name}")
    return generated


def _plot_disk_usage(timeline: Dict[str, Any], out_dir: Path) -> Optional[str]:
    """Plot disk footprint (net cumulative size) over time."""
    mode = timeline.get("mode", "process")
    times = np.array(timeline.get("sample_times_s", []))
    if len(times) == 0:
        return None

    # format: list of (label, sizes_mb_array, color)
    series_to_plot = []

    if mode == "process":
        sizes_mb = timeline.get("proc_disk_usage_mb", [])
        if sizes_mb:
            series_to_plot.append(("Process", np.array(sizes_mb), '#2563EB'))
    elif mode == "docker":
        docker_stats = timeline.get("docker_stats", {})
        colors = ['#2563EB', '#F59E0B', '#10B981', '#8B5CF6', '#EF4444', '#14B8A6']
        color_idx = 0
        for container_name, stats in docker_stats.items():
            container_sizes = np.array(stats.get("disk_usage_mb", []))
            if len(container_sizes) > 0:
                series_to_plot.append((container_name, container_sizes, colors[color_idx % len(colors)]))
                color_idx += 1

    if not series_to_plot:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    # Cumulative Disk Footprint (Net Size) per container
    for label, sizes_mb, color in series_to_plot:
        min_len = min(len(times), len(sizes_mb))
        ax.plot(times[:min_len], sizes_mb[:min_len], color=color, linewidth=1.8, marker=".", markersize=4, label=label)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Net Disk Size (MB)", fontsize=11)
    ax.set_title("Benchmark Disk Footprint over time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    path = str(out_dir / "disk_usage_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def _plot_latency_cdf(
    latencies: List[float],
    out_dir: Path,
    pass_num: Optional[int] = None,
) -> Optional[str]:
    """Plot query latency CDF.

    When *pass_num* is provided the plot is labelled "Pass N" and saved as
    ``latency_cdf_pass_N.png`` inside *out_dir* (the per-pass directory).
    """
    if not latencies:
        return None

    arr = np.sort(np.array(latencies, dtype=float))
    cdf = np.arange(1, len(arr) + 1) / len(arr)

    pcts = {
        "p50":    np.percentile(arr, 50),
        "p95":    np.percentile(arr, 95),
        "p99":    np.percentile(arr, 99),
        "p99.9":  np.percentile(arr, 99.9),
        "p99.99": np.percentile(arr, 99.99),
    }

    pass_label = f" — Pass {pass_num}" if pass_num is not None else ""
    filename   = f"latency_cdf_pass_{pass_num}.png" if pass_num is not None else "latency_cdf.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(arr, cdf, color="#3B82F6", linewidth=1.8, label="CDF")

    colors = ["#22C55E", "#F59E0B", "#EF4444", "#9333EA", "#EC4899"]
    for (label, val), color in zip(pcts.items(), colors):
        ax.axvline(val, color=color, linestyle="--", linewidth=1.2,
                   label=f"{label}={val:.2f}ms")

    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title(f"Query Latency CDF{pass_label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    path = str(out_dir / filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_recall_vs_latency(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Grouped bar chart of p50 / p95 / p99 latency with Recall@k
    overlaid on a secondary Y-axis.  Gives a single-glance
    accuracy-vs-speed tradeoff view for the run.
    """
    percentiles = ["p50", "p95", "p99"]
    lat_keys    = ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]
    lat_vals    = [stats.get(k) for k in lat_keys]

    if all(v is None for v in lat_vals):
        return None

    lat_vals = [v or 0.0 for v in lat_vals]
    recall   = stats.get("recall")          # scalar 0-1

    x     = np.arange(len(percentiles))
    width = 0.5

    fig, ax1 = plt.subplots(figsize=(7, 5))

    bars = ax1.bar(x, lat_vals, width,
                   color=["#22C55E", "#F59E0B", "#EF4444"],
                   alpha=0.85, zorder=2)
    _bar_labels(ax1, bars)

    ax1.set_xticks(x)
    ax1.set_xticklabels([p.upper() for p in percentiles], fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_ylim(bottom=0, top=max(lat_vals) * 1.35)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_title("Recall@K vs Query Latency", fontsize=14, fontweight="bold")

    # Secondary axis: recall as a horizontal band
    if recall is not None:
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 1.0)
        ax2.axhline(recall, color="#6366F1", linewidth=2.0,
                    linestyle="--", label=f"Recall@k = {recall:.4f}", zorder=3)
        ax2.set_ylabel("Recall@k", fontsize=12, color="#6366F1")
        ax2.tick_params(axis="y", labelcolor="#6366F1")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=10, loc="upper right")

        # Shade the recall band for emphasis
        ax2.axhspan(recall - 0.005, min(1.0, recall + 0.005),
                    color="#6366F1", alpha=0.12)

    path = str(out_dir / "recall_vs_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_latency_vs_collection_size(stats: dict, out_dir: Path) -> Optional[str]:
    snapshots = stats.get("_latency_snapshots", [])
    if not snapshots:
        return None

    sizes     = np.array([s for s, _ in snapshots])
    latencies = np.array([l for _, l in snapshots])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sizes, latencies, s=12, alpha=0.35, color="#3B82F6", label="query batch")

    # Rolling median trend
    order    = np.argsort(sizes)
    s_sorted = sizes[order]
    l_sorted = latencies[order]
    window   = max(1, len(s_sorted) // 20)
    trend    = np.convolve(l_sorted, np.ones(window) / window, mode="valid")
    ax.plot(s_sorted[window - 1:], trend, color="#EF4444",
            linewidth=1.8, label="rolling median")

    drift = stats.get("latency_drift")
    if drift is not None:
        ax.annotate(f"slope: {drift:+.3f} ms/window",
                    xy=(0.02, 0.93), xycoords="axes fraction",
                    fontsize=9, color="#EF4444")

    ax.set_xlabel("Collection size (vectors)", fontsize=12)
    ax.set_ylabel("Query latency (ms)", fontsize=12)
    ax.set_title("Query Latency vs Collection Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = str(out_dir / "latency_vs_collection_size.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_concurrent_latency_cdf(stats: dict, out_dir: Path) -> Optional[str]:
    query_lat  = np.array(stats.get("_raw_latencies", []),        dtype=float)
    ingest_lat = np.array(stats.get("_raw_ingest_latencies", []), dtype=float)

    if len(query_lat) == 0 and len(ingest_lat) == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    for lat, label, color in [
        (query_lat,  "query",  "#3B82F6"),
        (ingest_lat, "ingest", "#F59E0B"),
    ]:
        if len(lat) == 0:
            continue
        sorted_lat = np.sort(lat)
        cdf        = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
        ax.plot(sorted_lat, cdf, label=label, color=color, linewidth=1.8)

        for pct, ls in [(99, "--"), (95, ":")]:
            val = np.percentile(lat, pct)
            ax.axvline(val, color=color, linestyle=ls, linewidth=1.0, alpha=0.6,
                       label=f"{label} p{pct}={val:.1f}ms")

    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title("Query vs Ingest Latency CDF", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    path = str(out_dir / "concurrent_latency_cdf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_concurrent_latency_percentiles(stats: dict, out_dir: Path) -> Optional[str]:
    percentiles = ["p50", "p95", "p99"]
    query_vals  = [stats.get(f"query_latency_{p}")  for p in percentiles]
    ingest_vals = [stats.get(f"ingest_latency_{p}") for p in percentiles]

    if all(v is None for v in query_vals + ingest_vals):
        return None

    # Replace None with 0 for plotting; bar label skips zeros
    query_vals  = [v or 0 for v in query_vals]
    ingest_vals = [v or 0 for v in ingest_vals]

    x     = np.arange(len(percentiles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width / 2, query_vals,  width,
                label="Query",  color="#3B82F6", alpha=0.85)
    b2 = ax.bar(x + width / 2, ingest_vals, width,
                label="Ingest", color="#F59E0B", alpha=0.85)

    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    # Recall delta annotation
    r_init  = stats.get("recall_initial")
    r_final = stats.get("recall_final")
    r_delta = stats.get("recall_delta")
    parts   = []
    if r_init  is not None: parts.append(f"Recall initial: {r_init:.4f}")
    if r_final is not None: parts.append(f"Recall final: {r_final:.4f}")
    if r_delta is not None: parts.append(f"Δ: {r_delta:+.4f}")
    if parts:
        ax.annotate("  |  ".join(parts),
                    xy=(0.5, -0.12), xycoords="axes fraction",
                    ha="center", fontsize=9, fontstyle="italic", color="#6B7280")

    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in percentiles], fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Query vs Ingest Latency Percentiles", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    path = str(out_dir / "concurrent_latency_percentiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_rwd_latency_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    """Total vs execution vs stall latency per query batch with reindex markers."""
    total = np.array(stats.get("_raw_latencies",       []), dtype=np.float32)
    exec_ = np.array(stats.get("_raw_exec_latencies",  []), dtype=np.float32)
    stall = np.array(stats.get("_raw_stall_latencies", []), dtype=np.float32)

    if len(total) == 0:
        return None

    n = len(total)
    x = np.arange(n)

    def smooth(arr: np.ndarray, w: int = 20) -> np.ndarray:
        if w <= 1 or len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="same")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, smooth(total), color="#2563EB", linewidth=1.4, label="Total latency")

    if len(exec_) == n:
        ax.plot(x, smooth(exec_), color="#16A34A", linewidth=1.2,
                linestyle="--", label="Execution latency")

    if len(stall) == n:
        ax.fill_between(x, 0, smooth(stall), color="#DC2626",
                        alpha=0.25, label="Stall (lock wait)")
        # Mark reindex windows — any batch that waited > 0.5 ms
        reindex_qs = np.where(stall > 0.5)[0]
        first = True
        for qi in reindex_qs:
            ax.axvline(x=qi, color="#F97316", linewidth=0.8, linestyle=":",
                       alpha=0.7, label="Reindex event" if first else None)
            first = False

    # Summary stats box
    parts = []
    for key, label in [("latency_total_p50", "p50"),
                        ("latency_total_p99", "p99"),
                        ("stall_latency_max", "stall_max")]:
        v = stats.get(key)
        if v is not None:
            parts.append(f"{label}={v:.1f}ms")
    if parts:
        ax.text(0.99, 0.97, "  |  ".join(parts), transform=ax.transAxes,
                fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel("Query batch index", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("RWD: Total vs Execution vs Stall Latency", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

    path = str(out_dir / "rwd_latency_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_rwd_drift_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    """Drift score per monitor interval with threshold line and reindex markers."""
    monitor_events = stats.get("_monitor_events", [])

    drift_scores: list[float] = []
    reindex_at:   list[int]   = []

    for evt in monitor_events:
        if not evt:
            continue
        ds = evt.get("drift_score")
        if ds is not None:
            drift_scores.append(float(ds))
            if evt.get("event") == "reindex":
                reindex_at.append(len(drift_scores) - 1)

    if not drift_scores:
        return None

    drift_threshold = float(stats.get("_drift_threshold", 0.1))
    x = np.arange(len(drift_scores))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, drift_scores, color="#7C3AED", linewidth=1.6,
            marker="o", markersize=4, label="Drift score")
    ax.axhline(drift_threshold, color="#DC2626", linewidth=1.2,
               linestyle="--", label=f"Threshold ({drift_threshold})")
    ax.fill_between(x, drift_threshold, drift_scores,
                    where=np.array(drift_scores) > drift_threshold,
                    color="#DC2626", alpha=0.15, label="Above threshold")

    first = True
    for ri in reindex_at:
        ax.axvline(x=ri, color="#F97316", linewidth=1.0, linestyle=":",
                   alpha=0.8, label="Reindex triggered" if first else None)
        first = False

    ax.set_xlabel("Monitor interval", fontsize=11)
    ax.set_ylabel("Drift score", fontsize=11)
    ax.set_title("RWD: Drift Score over Monitoring Intervals", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, max(1, len(drift_scores) - 1))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = str(out_dir / "rwd_drift_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_umap_drift_events(
    umap_events: List[Dict],
    out_dir: Path,
) -> List[str]:
    try:
        from umap import UMAP
    except ImportError:
        print("[Plots] umap-learn not installed — skipping UMAP plots. Run: pip install umap-learn")
        return []

    if not umap_events:
        return []

    generated = []

    for evt in umap_events:
        baseline = evt.get("baseline")
        drift    = evt.get("drift")
        idx      = evt.get("event_idx", 0)
        reason   = evt.get("reason", "unknown")
        drift_score  = evt.get("drift_score", 0.0)
        zombie_ratio = evt.get("zombie_ratio", 0.0)

        if baseline is None or drift is None:
            continue
        if len(baseline) == 0 or len(drift) == 0:
            continue

        # Fit UMAP on baseline, transform both — shared coordinate space
        combined = np.vstack([baseline, drift])
        reducer  = UMAP(n_components=2, random_state=42, verbose=False)
        embedded = reducer.fit_transform(combined)

        n_base  = len(baseline)
        emb_base  = embedded[:n_base]
        emb_drift = embedded[n_base:]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            emb_base[:, 0], emb_base[:, 1],
            s=8, alpha=0.5, color="#3B82F6",
            label=f"Baseline (n={n_base})",
        )
        ax.scatter(
            emb_drift[:, 0], emb_drift[:, 1],
            s=8, alpha=0.5, color="#EF4444",
            label=f"Drift buffer (n={len(drift)})",
        )

        ax.set_title(
            f"UMAP — Reindex Event {idx}  |  reason: {reason}\n"
            f"drift score: {drift_score:.4f}   zombie ratio: {zombie_ratio:.2%}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("UMAP-1", fontsize=10)
        ax.set_ylabel("UMAP-2", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2)
        
        path = str(out_dir / f"umap_drift_event_{idx}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)
        print(f"[Plots] UMAP event {idx} saved: {Path(path).name}")

    return generated

def _plot_rwd_query_percentiles_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    passes = stats.get("_query_passes", [])
    p95s = [p.get("query_latency_p95") for p in passes if p.get("query_latency_p95") is not None]
    
    if not p95s :
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    x = [p.get("time_s", i) for i, p in enumerate(passes)]

    def smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
        if w <= 1 or len(arr) < w: return arr
        return np.convolve(arr, np.ones(w)/w, mode="same")

    # Plot the smoothed median and tail percentile
    ax.plot(x, smooth(np.array(p95s)), color="#3B82F6", linewidth=1.5, label="p95 Latency (Tail)")

    # Overlay vertical lines for Reindex Events
    monitor_events = stats.get("_monitor_events", [])
    for evt in monitor_events:
        if evt and evt.get("event") == "reindex" and evt.get("time_s") is not None:
            ax.axvline(x=evt["time_s"], color="#F97316", linewidth=1.0, linestyle=":", alpha=0.8)

    ax.set_xlabel("Workload Time (s)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    _overlay_burst_phases(ax, stats)
    ax.set_title("RWD: Query p95 Latency Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    path = str(out_dir / "rwd_query_percentiles_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_rwd_recall_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Redesigned recall-over-time plot.
    
    Layout:
    - Top panel: Recall@K step line with pass annotations + burst phase bands
    - Bottom panel: p50 / p95 / p99 query latency fan + zombie-ratio area fill
    
    Burst phases (baseline/burst/cooldown/recovery) are overlaid via
    coloured x-spans on both panels so the reader can immediately see
    what was happening when recall dropped or latency spiked.
    """
    passes = stats.get("_query_passes", [])
    if not passes:
        return None

    # ── gather per-pass series (filter to passes that have time_s + recall) ──
    valid = [
        p for p in passes
        if p.get("time_s") is not None and p.get("recall") is not None
    ]
    if not valid:
        return None

    xs        = [p["time_s"] for p in valid]
    recalls   = [p["recall"] for p in valid]
    zombies   = [p.get("zombie_ratio_at_query", 0.0) for p in valid]
    p50s      = [p.get("query_latency_p50") for p in valid]
    p95s      = [p.get("query_latency_p95") for p in valid]
    p99s      = [p.get("query_latency_p99") for p in valid]
    pass_nums = [p.get("pass", i + 1) for i, p in enumerate(valid)]

    have_lat = any(v is not None for v in p50s)

    n_rows = 2 if have_lat else 1
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, 4 * n_rows + 1),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.4] if n_rows == 2 else [1]},
    )
    if n_rows == 1:
        axes = [axes]
    ax_recall, ax_lat = (axes[0], axes[1]) if n_rows == 2 else (axes[0], None)

    PHASE_COLORS = {
        "baseline":  ("#3B82F6", 0.08),
        "burst":     ("#EF4444", 0.14),
        "cooldown":  ("#10B981", 0.10),
        "recovery":  ("#EAB308", 0.12),
    }

    def _shade_phases(ax_):
        """Draw phase background bands on ax_ from the burst timeline."""
        tl = stats.get("_burst_timeline") or stats.get("burst_timeline", [])
        seen = set()
        for seg in tl:
            phase  = seg.get("phase")
            t0     = seg.get("start_time_s")
            t1     = seg.get("end_time_s")
            if phase is None or t0 is None or t1 is None or phase == "done":
                continue
            color, alpha = PHASE_COLORS.get(phase, ("#9CA3AF", 0.08))
            label = phase.capitalize() if phase not in seen else "_nolegend_"
            seen.add(phase)
            ax_.axvspan(t0, t1, color=color, alpha=alpha, zorder=0, label=label)

    # ── TOP: recall ──────────────────────────────────────────────────────────
    _shade_phases(ax_recall)

    ax_recall.plot(
        xs, recalls,
        color="#1D4ED8", linewidth=2, marker="o", markersize=5,
        zorder=3, label="Recall@K",
    )

    # Annotate pass number at each point (every 2nd to avoid clutter)
    step = max(1, len(valid) // 20)
    for i, (xi, ri, pn) in enumerate(zip(xs, recalls, pass_nums)):
        if i % step == 0:
            ax_recall.annotate(
                f"P{pn}",
                (xi, ri),
                xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=7, color="#374151",
                clip_on=True,
            )

    # Zombie ratio as a semi-transparent area behind recall
    if any(z > 0 for z in zombies):
        ax_zom = ax_recall.twinx()
        ax_zom.fill_between(
            xs, zombies,
            color="#DC2626", alpha=0.15, zorder=1, label="Zombie Ratio",
        )
        ax_zom.set_ylim(0, max(max(zombies) * 1.4, 0.05))
        ax_zom.set_ylabel("Zombie Ratio", fontsize=9, color="#DC2626")
        ax_zom.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax_zom.tick_params(axis='y', labelcolor="#DC2626", labelsize=8)
        # merge into one legend
        h1, l1 = ax_recall.get_legend_handles_labels()
        h2, l2 = ax_zom.get_legend_handles_labels()
        ax_recall.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, framealpha=0.9)
    else:
        ax_recall.legend(loc="lower left", fontsize=8, framealpha=0.9)

    ax_recall.set_ylabel("Recall@K", fontsize=11)
    ax_recall.set_ylim(-0.02, 1.08)
    ax_recall.grid(True, alpha=0.25, linestyle="--")
    ax_recall.set_title(
        "Burst RWD: Recall, Latency & Phase Timeline",
        fontsize=13, fontweight="bold",
    )

    # Reindex vertical markers
    monitor_events = stats.get("_monitor_events", [])
    first_ri = True
    for evt in monitor_events:
        if evt and evt.get("event") == "reindex":
            t = evt.get("time_s")
            if t is not None:
                ax_recall.axvline(
                    x=t, color="#F97316", linewidth=1.2, linestyle=":",
                    alpha=0.9, label="Reindex" if first_ri else "_nolegend_",
                    zorder=5,
                )
                first_ri = False

    # ── BOTTOM: latency fan ───────────────────────────────────────────────────
    if ax_lat is not None and have_lat:
        _shade_phases(ax_lat)

        # filter Nones
        lat_xs  = [xi for xi, v in zip(xs, p50s) if v is not None]
        l50     = [v  for v        in p50s        if v is not None]
        l95     = [p.get("query_latency_p95") or 0 for p in valid if p.get("query_latency_p50") is not None]
        l99     = [p.get("query_latency_p99") or 0 for p in valid if p.get("query_latency_p50") is not None]

        ax_lat.fill_between(lat_xs, l50, l99, alpha=0.18, color="#6366F1", label="p50–p99 band")
        ax_lat.plot(lat_xs, l50, color="#6366F1", linewidth=1.8, label="p50")
        ax_lat.plot(lat_xs, l95, color="#A78BFA", linewidth=1.2, linestyle="--", label="p95")
        ax_lat.plot(lat_xs, l99, color="#EF4444", linewidth=1.0, linestyle=":",  label="p99")

        # Reindex markers on latency panel too
        for evt in monitor_events:
            if evt and evt.get("event") == "reindex":
                t = evt.get("time_s")
                if t is not None:
                    ax_lat.axvline(x=t, color="#F97316", linewidth=1.0,
                                   linestyle=":", alpha=0.8)

        ax_lat.set_ylabel("Query Latency (ms)", fontsize=10)
        ax_lat.set_xlabel("Workload Time (s)", fontsize=10)
        ax_lat.set_ylim(bottom=0)
        ax_lat.grid(True, alpha=0.25, linestyle="--")
        ax_lat.legend(loc="upper left", fontsize=8, framealpha=0.9)
    elif ax_lat is None:
        ax_recall.set_xlabel("Workload Time (s)", fontsize=10)

    plt.tight_layout()
    path = str(out_dir / "rwd_recall_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_rwd_query_percentiles_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Per-pass query latency percentiles over workload time.

    Shows p50 / p95 / p99 / p99.9 lines on a single axis so you can see
    exactly how tail latency evolves across bursts.
    Burst phase bands are overlaid if a burst timeline is present.
    Each pass is annotated with its pass number (every Nth point to
    avoid clutter) so specific passes can be identified.
    """
    passes = stats.get("_query_passes", [])
    valid = [
        p for p in passes
        if p.get("time_s") is not None
        and p.get("query_latency_p50") is not None
    ]
    if not valid:
        return None

    xs        = [p["time_s"] for p in valid]
    p50s      = [p["query_latency_p50"]     for p in valid]
    p95s      = [p.get("query_latency_p95") or p["query_latency_p50"] for p in valid]
    p99s      = [p.get("query_latency_p99") or p["query_latency_p50"] for p in valid]
    p999s     = [p.get("query_latency_p99_9") or p.get("query_latency_p99") or p["query_latency_p50"] for p in valid]
    pass_nums = [p.get("pass", i + 1) for i, p in enumerate(valid)]

    fig, ax = plt.subplots(figsize=(16, 5))

    PHASE_COLORS = {
        "baseline":  ("#3B82F6", 0.08),
        "burst":     ("#EF4444", 0.14),
        "cooldown":  ("#10B981", 0.10),
        "recovery":  ("#EAB308", 0.12),
    }
    tl = stats.get("_burst_timeline") or stats.get("burst_timeline", [])
    seen_phases = set()
    for seg in tl:
        phase = seg.get("phase")
        t0    = seg.get("start_time_s")
        t1    = seg.get("end_time_s")
        if phase is None or t0 is None or t1 is None or phase == "done":
            continue
        color, alpha = PHASE_COLORS.get(phase, ("#9CA3AF", 0.08))
        label = phase.capitalize() if phase not in seen_phases else "_nolegend_"
        seen_phases.add(phase)
        ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0, label=label)

    ax.fill_between(xs, p50s, p999s, alpha=0.10, color="#6366F1")
    ax.plot(xs, p50s,  color="#6366F1", linewidth=2.0, marker="o", markersize=3, label="p50")
    ax.plot(xs, p95s,  color="#A78BFA", linewidth=1.5, linestyle="--",            label="p95")
    ax.plot(xs, p99s,  color="#F59E0B", linewidth=1.2, linestyle="-.",            label="p99")
    ax.plot(xs, p999s, color="#EF4444", linewidth=1.0, linestyle=":",             label="p99.9")

    # Annotate pass numbers (every Nth to avoid clutter)
    step = max(1, len(valid) // 20)
    for i, (xi, yi, pn) in enumerate(zip(xs, p95s, pass_nums)):
        if i % step == 0:
            ax.annotate(
                f"P{pn}",
                (xi, yi),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=7, color="#374151", clip_on=True,
            )

    # Reindex events
    monitor_events = stats.get("_monitor_events", [])
    first_ri = True
    for evt in monitor_events:
        if evt and evt.get("event") == "reindex":
            t = evt.get("time_s")
            if t is not None:
                ax.axvline(
                    x=t, color="#F97316", linewidth=1.2, linestyle=":",
                    alpha=0.9, label="Reindex" if first_ri else "_nolegend_",
                    zorder=5,
                )
                first_ri = False

    ax.set_xlabel("Workload Time (s)", fontsize=11)
    ax.set_ylabel("Query Latency (ms)", fontsize=11)
    ax.set_title("RWD: Query Latency Percentiles Over Time", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = str(out_dir / "rwd_query_percentiles_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pass_qps_vs_recall(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Dual Y-axis time-series: QPS (left axis, blue) and Recall@K (right axis, orange)
    over query passes, with optional vertical reindex event markers.

    Replaces the old scatter/bar chart. For pass-based workloads the natural dimension
    is time (pass index), not a Pareto tradeoff curve — both metrics evolve together so
    you can observe them diverge and recover around reindex events.

    Works for all workloads that populate _query_passes (rwd, streaming_ingestion,
    lid, burst_rwd, hot_cold, deduplication, etc.).

    Reindex markers are drawn only when _monitor_events is present in stats.
    latency_drift_ms on each reindex event is annotated on the marker when available.
    """
    passes = stats.get("_query_passes", [])
    if not passes:
        return None

    use_bps = all(p.get("bps") is not None for p in passes)
    metric_key   = "bps"  if use_bps else "qps"
    metric_label = "BPS"  if use_bps else "QPS"

    valid = [
        p for p in passes
        if p.get(metric_key) is not None and p.get("recall") is not None
    ]
    if not valid:
        return None

    pass_nums   = [p.get("pass", i + 1) for i, p in enumerate(valid)]
    metric_vals = [p[metric_key]  for p in valid]
    recall_vals = [p["recall"]    for p in valid]
    xs          = np.array(pass_nums, dtype=float)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax_qps = plt.subplots(figsize=(max(10, len(valid) * 0.7 + 2), 5))
    ax_rec = ax_qps.twinx()

    # ── QPS / BPS line (left Y-axis, blue) ───────────────────────────────────
    qps_color = "#2563EB"
    ax_qps.plot(xs, metric_vals,
                color=qps_color, linewidth=2.2, marker="o", markersize=5,
                zorder=3, label=metric_label)
    ax_qps.fill_between(xs, metric_vals, alpha=0.07, color=qps_color, zorder=1)

    ax_qps.set_xlabel("Query Pass", fontsize=11)
    ax_qps.set_ylabel(f"{metric_label} (Throughput)", fontsize=11, color=qps_color)
    ax_qps.tick_params(axis="y", labelcolor=qps_color)
    ax_qps.set_ylim(bottom=0, top=max(metric_vals) * 1.25 if metric_vals else 1)
    ax_qps.set_xticks(xs)
    ax_qps.set_xticklabels([f"P{int(p)}" for p in xs], fontsize=9)
    ax_qps.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # ── Recall line (right Y-axis, orange) ────────────────────────────────────
    rec_color = "#F97316"
    ax_rec.plot(xs, recall_vals,
                color=rec_color, linewidth=2.2, marker="s", markersize=5,
                linestyle="--", zorder=3, label="Recall@K")
    ax_rec.set_ylabel("Recall@K", fontsize=11, color=rec_color)
    ax_rec.tick_params(axis="y", labelcolor=rec_color)
    ax_rec.set_ylim(
        max(0.0, min(recall_vals) - 0.08),
        min(1.0, max(recall_vals) + 0.06),
    )
    ax_rec.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ── Reindex markers (only for workloads that have _monitor_events) ────────
    monitor_events = stats.get("_monitor_events", [])
    reindex_events = [e for e in monitor_events if e and e.get("event") == "reindex"]

    for i, evt in enumerate(reindex_events):
        r_time = evt.get("time_s")
        if r_time is None:
            continue

        # Map reindex wall-clock time to nearest pass by time_s.
        times_s = [p.get("time_s") for p in valid]
        has_times = all(t is not None for t in times_s)

        if has_times:
            closest_idx = min(range(len(times_s)),
                              key=lambda j: abs(times_s[j] - r_time))
        else:
            # Fall back to pass midpoint (no time_s available)
            closest_idx = min(len(valid) - 1, i * max(1, len(valid) // max(1, len(reindex_events))))

        vx = xs[closest_idx]

        drift_ms = evt.get("latency_drift_ms")
        marker_label = f"Reindex {i + 1}"
        if drift_ms is not None:
            sign = "+" if drift_ms >= 0 else ""
            marker_label += f"\nΔlat={sign}{drift_ms:.0f}ms"

        ax_qps.axvline(x=vx, color="#DC2626", linewidth=1.3,
                       linestyle=":", alpha=0.85, zorder=4)
        ax_qps.annotate(
            marker_label,
            xy=(vx, max(metric_vals) * 1.18),
            xytext=(4, 0), textcoords="offset points",
            fontsize=7, color="#DC2626", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75,
                      ec="#DC2626", lw=0.6),
            clip_on=False,
        )

    # ── Title, grid, combined legend ──────────────────────────────────────────
    ax_qps.set_title(
        f"{metric_label} & Recall@K over Query Passes",
        fontsize=13, fontweight="bold",
    )
    ax_qps.grid(True, alpha=0.25, linestyle="--", zorder=0)

    h1, l1 = ax_qps.get_legend_handles_labels()
    h2, l2 = ax_rec.get_legend_handles_labels()
    ax_qps.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    filename = "bps_recall_over_passes.png" if use_bps else "qps_recall_over_passes.png"
    path = str(out_dir / filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_qps_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    window_qps = stats.get("_window_qps", [])
    if not window_qps:
        return None

    qps_arr  = np.array(window_qps, dtype=float)
    mean_qps = float(np.mean(qps_arr))
    stability = stats.get("qps_stability")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(qps_arr))

    ax.plot(x, qps_arr, color="#3B82F6", linewidth=1.5,
            marker="o", markersize=4, label="window QPS")
    ax.fill_between(x, qps_arr, alpha=0.1, color="#3B82F6")
    ax.axhline(mean_qps, color="#EF4444", linestyle="--", linewidth=1.2,
               label=f"mean: {mean_qps:.1f} QPS")

    if stability is not None:
        ax.annotate(f"QPS stability (CV): {stability:.3f}",
                    xy=(0.02, 0.93), xycoords="axes fraction",
                    fontsize=9, color="#6B7280")

    ax.set_xlabel("Window index", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title("QPS Over Time Windows", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    path = str(out_dir / "qps_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_rwd_mutation_latency_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    events = stats.get("_mutation_events", [])
    if not events: return None
    
    write_t, write_lat = [], []
    del_t, del_lat = [], []
    
    for evt in events:
        if not evt: continue
        op = evt.get("op")
        t = evt.get("time_s")
        lat = evt.get("latency_ms") or evt.get("stats", {}).get("total_time_ms") or evt.get("stats", {}).get("execution_time_ms")
        if t is None or lat is None: continue
        
        if op == "write":
            write_t.append(t)
            write_lat.append(lat)
        elif op == "delete":
            del_t.append(t)
            del_lat.append(lat)
            
    if not write_t and not del_t: return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    if write_t:
        ax1.plot(write_t, write_lat, color="#10B981", marker="o", markersize=3, linewidth=1.2, label="Write Latency")
        ax1.set_ylabel("Latency (ms)")
        _overlay_burst_phases(ax1, stats)
        ax1.set_title("Write Latency over Time", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")
        ax1.set_ylim(bottom=0)
        
    if del_t:
        ax2.plot(del_t, del_lat, color="#EF4444", marker="o", markersize=4, linewidth=1.5, label="Delete Latency")
        ax2.set_ylabel("Latency (ms)")
        _overlay_burst_phases(ax2, stats)
        ax2.set_title("Delete Latency over Time", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")
        ax2.set_ylim(bottom=0)
        
    ax2.set_xlabel("Workload Time (s)", fontsize=11)
    
    # Mark reindex events
    monitor_events = stats.get("_monitor_events", [])
    for evt in monitor_events:
        if evt and evt.get("event") == "reindex":
            t = evt.get("time_s")
            if t is not None:
                for ax in (ax1, ax2):
                    ax.axvline(x=t, color="#F97316", linewidth=1.0, linestyle=":", alpha=0.8)
                    
    plt.tight_layout()
    path = str(out_dir / "rwd_mutation_latency_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_latency_per_modality(
    latency_per_mod: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> Optional[str]:
    modalities = list(latency_per_mod.keys())
    means = [latency_per_mod[m]["mean_ms"] for m in modalities]
    p99s  = [latency_per_mod[m]["p99_ms"]  for m in modalities]

    x     = np.arange(len(modalities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(modalities) * 2), 5))
    b1 = ax.bar(x - width / 2, means, width, label="Mean", color="#3B82F6", alpha=0.85)
    b2 = ax.bar(x + width / 2, p99s,  width, label="P99",  color="#EF4444", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=11)
    ax.set_xlabel("Query Modality", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency per Query Modality", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    path = str(out_dir / "latency_per_modality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_size_per_partition(
    size_per_part: Dict[str, int],
    out_dir: Path,
) -> Optional[str]:
    parts  = list(size_per_part.keys())
    counts = [size_per_part[p] for p in parts]

    fig, ax = plt.subplots(figsize=(7, max(3, len(parts))))
    bars = ax.barh(parts, counts, color="#6366F1", alpha=0.85)
    ax.set_xlabel("Entity Count", fontsize=12)
    ax.set_title("Entities per Partition", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    _hbar_labels(ax, bars)

    path = str(out_dir / "size_per_partition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_process_timeline(
    timeline: Dict[str, List[float]],
    out_dir: Path,
    poll_interval_s: float = 0.2,
) -> Optional[str]:

    cpu     = timeline.get("proc_cpu", [])
    mem     = timeline.get("proc_mem_mb", [])
    disk_r  = timeline.get("proc_read_mbps", [])
    disk_w  = timeline.get("proc_write_mbps", [])

    if not cpu and not mem and not disk_r and not disk_w:
        return None

    t = timeline.get("sample_times_s") or \
    np.arange(len(cpu)) * poll_interval_s 

    fig, (ax_cr, ax_disk) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Process Resource Usage During Workload", fontsize=13, fontweight="bold")

    # --- CPU + RAM ---
    ax_ram = ax_cr.twinx()
    if cpu:
        m = min(len(t), len(cpu))
        ax_cr.plot(t[:m], cpu[:m], color="#3B82F6", linewidth=1.5, label="CPU %")
    if mem:
        m = min(len(t), len(mem))
        ax_ram.plot(t[:m], mem[:m], color="#EF4444", linewidth=1.5, label="RAM (MB)")
    ax_cr.set_xlabel("Time (s)",    fontsize=10)
    ax_cr.set_ylabel("CPU %",       fontsize=10, color="#3B82F6")
    ax_ram.set_ylabel("Memory (MB)",fontsize=10, color="#EF4444")
    ax_cr.set_title("CPU & RAM",    fontsize=11)
    ax_cr.grid(True, alpha=0.25)
    lines1, labels1 = ax_cr.get_legend_handles_labels()
    lines2, labels2 = ax_ram.get_legend_handles_labels()
    if lines1 or lines2:
        ax_cr.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    # --- Disk ---
    ax_dw = ax_disk.twinx()
    if disk_r:
        m = min(len(t), len(disk_r))
        ax_disk.plot(t[:m], disk_r[:m], color="#10B981", linewidth=1.5, label="Read (MB/s)")
    if disk_w:
        m = min(len(t), len(disk_w))
        ax_dw.plot(t[:m], disk_w[:m], color="#F59E0B", linewidth=1.2,
                     linestyle="--", label="Write (MB/s)")
    ax_disk.set_xlabel("Time (s)",       fontsize=10)
    ax_disk.set_ylabel("Read (MB/s)",    fontsize=10, color="#10B981")
    ax_dw.set_ylabel("Write (MB/s)",     fontsize=10, color="#F59E0B")
    ax_disk.set_title("Disk Throughput", fontsize=11)
    ax_disk.grid(True, alpha=0.25)
    lines1, labels1 = ax_disk.get_legend_handles_labels()
    lines2, labels2 = ax_dw.get_legend_handles_labels()
    if lines1 or lines2:
        ax_disk.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    plt.tight_layout()
    path = str(out_dir / "process_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_docker_timeline(
    timeline: Dict[str, List[float]],
    out_dir: Path,
    poll_interval_s: float = 0.2,
) -> Optional[str]:

    docker_stats = timeline.get("docker_stats", {})
    if not docker_stats:
        return None

    containers   = list(docker_stats.keys())
    n_containers = len(containers)

    fig, axes = plt.subplots(n_containers, 2, figsize=(14, 4 * n_containers), squeeze=False)
    fig.suptitle("Docker Container Resource Usage During Workload", fontsize=13, fontweight="bold")

    for row, name in enumerate(containers):
        series  = docker_stats[name]
        cpu     = series.get("cpu",        [])
        mem     = series.get("mem_mb",     [])
        disk_r  = series.get("read_mbps",  [])
        disk_w  = series.get("write_mbps", [])

        t = timeline.get("sample_times_s") or \
        np.arange(len(cpu)) * poll_interval_s   

        ax_cr   = axes[row][0]
        ax_disk = axes[row][1]
        ax_ram  = ax_cr.twinx()
        ax_dw   = ax_disk.twinx()

        # --- CPU + RAM ---
        if cpu:
            m = min(len(t), len(cpu))
            ax_cr.plot(t[:m], cpu[:m], color="#3B82F6", linewidth=1.5, label="CPU %")
        if mem:
            m = min(len(t), len(mem))
            ax_ram.plot(t[:m], mem[:m], color="#EF4444", linewidth=1.5, label="Mem (MB)")
        ax_cr.set_title(f"{name} — CPU & RAM", fontsize=10, fontweight="bold")
        ax_cr.set_ylabel("CPU %",       fontsize=9, color="#3B82F6")
        ax_ram.set_ylabel("Mem (MB)",   fontsize=9, color="#EF4444")
        ax_cr.grid(True, alpha=0.25)
        lines1, labels1 = ax_cr.get_legend_handles_labels()
        lines2, labels2 = ax_ram.get_legend_handles_labels()
        if lines1 or lines2:
            ax_cr.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        # --- Disk ---
        if disk_r:
            m = min(len(t), len(disk_r))
            ax_disk.plot(t[:m], disk_r[:m], color="#10B981", linewidth=1.5, label="Read (MB/s)")
        if disk_w:
            m = min(len(t), len(disk_w))
            ax_dw.plot(t[:m], disk_w[:m], color="#F59E0B", linewidth=1.2,
                       linestyle="--", label="Write (MB/s)")
        ax_disk.set_title(f"{name} — Disk Throughput", fontsize=10, fontweight="bold")
        ax_disk.set_ylabel("Read (MB/s)",  fontsize=9, color="#10B981")
        ax_dw.set_ylabel("Write (MB/s)",   fontsize=9, color="#F59E0B")
        ax_disk.grid(True, alpha=0.25)
        lines1, labels1 = ax_disk.get_legend_handles_labels()
        lines2, labels2 = ax_dw.get_legend_handles_labels()
        if lines1 or lines2:
            ax_disk.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        if row == n_containers - 1:
            ax_cr.set_xlabel("Time (s)",   fontsize=10)
            ax_disk.set_xlabel("Time (s)", fontsize=10)

    plt.tight_layout()
    path = str(out_dir / "docker_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)


def _hbar_labels(ax, bars):
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{int(w):,}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9)

def _plot_access_histogram(histogram: dict, out_dir: Path) -> Optional[str]:
    """Bar chart of per-vector access counts (hot-cold access histogram)."""
    ids    = list(histogram.keys())
    counts = [histogram[i] for i in ids]

    # Sort by count descending so hot vectors are on the left
    paired = sorted(zip(counts, ids), reverse=True)
    counts, ids = zip(*paired) if paired else ([], [])

    fig, ax = plt.subplots(figsize=(min(20, max(8, len(ids) // 50)), 4))
    x = np.arange(len(ids))
    ax.bar(x, counts, color="#6366F1", width=1.0, alpha=0.85)
    ax.set_xlabel("Vector rank (hottest → coldest)", fontsize=11)
    ax.set_ylabel("Access count", fontsize=11)
    ax.set_title("Hot-Cold Access Histogram", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)

    path = str(out_dir / "access_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hot_cold_latency(stats: dict, out_dir: Path) -> Optional[str]:
    """Grouped bar chart comparing hot vs cold latency percentiles."""
    labels = ["p50", "p95", "p99"]
    hot_vals  = [stats.get(f"hot_latency_{p}")  for p in labels]
    cold_vals = [stats.get(f"cold_latency_{p}") for p in labels]

    # Drop percentiles missing from both groups
    valid = [(l, h, c) for l, h, c in zip(labels, hot_vals, cold_vals)
             if h is not None or c is not None]
    if not valid:
        return None
    labels, hot_vals, cold_vals = zip(*valid)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - width/2, hot_vals,  width, label="Hot",  color="#EF4444", alpha=0.85)
    b2 = ax.bar(x + width/2, cold_vals, width, label="Cold", color="#3B82F6", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Hot vs Cold Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    path = str(out_dir / "hot_cold_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hot_cold_recall_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    """Line chart tracking hot vs cold query recall across passes."""
    passes = stats.get("_query_passes", [])
    valid = [p for p in passes if p.get("pass") is not None and p.get("recall") is not None]
    if not valid:
        return None

    xs = [p["pass"] for p in valid]
    recalls = [p["recall"] for p in valid]
    hot_recalls = [p.get("hot_recall") for p in valid]
    cold_recalls = [p.get("cold_recall") for p in valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(xs, recalls, color="#1D4ED8", linewidth=2, zorder=3, label="Overall Recall@K")
    
    if any(r is not None for r in hot_recalls):
        h_xs = [xi for xi, r in zip(xs, hot_recalls) if r is not None]
        h_rs = [r for r in hot_recalls if r is not None]
        ax.plot(h_xs, h_rs, color="#DC2626", linewidth=1.5, linestyle="--", zorder=4, label="Hot Recall@K")
        
    if any(r is not None for r in cold_recalls):
        c_xs = [xi for xi, r in zip(xs, cold_recalls) if r is not None]
        c_rs = [r for r in cold_recalls if r is not None]
        ax.plot(c_xs, c_rs, color="#059669", linewidth=1.5, linestyle=":", zorder=4, label="Cold Recall@K")

    ax.set_xlabel("Query Pass Number", fontsize=11)
    ax.set_ylabel("Recall@K", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Hot vs Cold Query Recall Over Time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.legend(fontsize=10, loc="lower left")

    path = str(out_dir / "hot_cold_recall_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_hot_cold_pgfaults_overall(stats: Dict[str, Any], out_dir: Path) -> Optional[str]:
    passes = stats.get("_query_passes", [])
    if not passes:
        return None
    
    # Check if pgfault data exists
    if "hot_query_pgfault" not in passes[0] and "hot_query_pgfault_total" not in stats:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Per-pass histogram (Grouped Bar Chart)
    xs = [p.get("pass", i+1) for i, p in enumerate(passes)]
    
    h_pf = [p.get("hot_query_pgfault", 0) for p in passes]
    h_mj = [p.get("hot_query_pgmajfault", 0) for p in passes]
    c_pf = [p.get("cold_query_pgfault", 0) for p in passes]
    c_mj = [p.get("cold_query_pgmajfault", 0) for p in passes]
    
    x_indices = np.arange(len(xs))
    width = 0.35
    
    ax1.bar(x_indices - width/2, h_pf, width, label="Hot Minor Faults", color="#ef4444")
    ax1.bar(x_indices - width/2, h_mj, width, bottom=h_pf, label="Hot Major Faults", color="#991b1b")
    
    ax1.bar(x_indices + width/2, c_pf, width, label="Cold Minor Faults", color="#3b82f6")
    ax1.bar(x_indices + width/2, c_mj, width, bottom=c_pf, label="Cold Major Faults", color="#1e3a8a")
    
    ax1.set_xlabel("Query Pass Number", fontsize=11)
    ax1.set_ylabel("Page Faults", fontsize=11)
    ax1.set_title("Hot vs Cold Page Faults per Pass", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(xs)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3, linestyle="--")

    # 2. Overall Totals
    h_pf_tot = stats.get("hot_query_pgfault_total", sum(h_pf))
    h_mj_tot = stats.get("hot_query_pgmajfault_total", sum(h_mj))
    c_pf_tot = stats.get("cold_query_pgfault_total", sum(c_pf))
    c_mj_tot = stats.get("cold_query_pgmajfault_total", sum(c_mj))
    
    labels = ['Hot Queries', 'Cold Queries']
    pf_tots = [h_pf_tot, c_pf_tot]
    mj_tots = [h_mj_tot, c_mj_tot]
    
    x_tot = np.arange(len(labels))
    ax2.bar(x_tot, pf_tots, 0.5, label="Minor Faults", color=["#ef4444", "#3b82f6"])
    ax2.bar(x_tot, mj_tots, 0.5, bottom=pf_tots, label="Major Faults", color=["#991b1b", "#1e3a8a"])
    
    ax2.set_ylabel("Total Page Faults", fontsize=11)
    ax2.set_title("Overall Page Faults (All Passes)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_tot)
    ax2.set_xticklabels(labels)
    
    import matplotlib.patches as mpatches
    minor_patch = mpatches.Patch(color='gray', label='Minor Faults')
    major_patch = mpatches.Patch(color='black', label='Major Faults')
    ax2.legend(handles=[minor_patch, major_patch], fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    path = str(out_dir / "hot_cold_pgfaults_overall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hot_cold_pgfaults_pass(stats: Dict[str, Any], out_dir: Path) -> Optional[str]:
    if "hot_query_pgfault" not in stats:
        return None
        
    import matplotlib.pyplot as plt
    import numpy as np
    
    pass_num = stats.get("pass", "Unknown")
    h_pf = stats.get("hot_query_pgfault", 0)
    h_mj = stats.get("hot_query_pgmajfault", 0)
    c_pf = stats.get("cold_query_pgfault", 0)
    c_mj = stats.get("cold_query_pgmajfault", 0)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    labels = ['Hot Queries', 'Cold Queries']
    pf_tots = [h_pf, c_pf]
    mj_tots = [h_mj, c_mj]
    
    x_tot = np.arange(len(labels))
    ax.bar(x_tot, pf_tots, 0.5, label="Minor Faults", color=["#ef4444", "#3b82f6"])
    ax.bar(x_tot, mj_tots, 0.5, bottom=pf_tots, label="Major Faults", color=["#991b1b", "#1e3a8a"])
    
    ax.set_ylabel("Page Faults", fontsize=11)
    ax.set_title(f"Hot vs Cold Page Faults (Pass {pass_num})", fontsize=12, fontweight="bold")
    ax.set_xticks(x_tot)
    ax.set_xticklabels(labels)
    
    import matplotlib.patches as mpatches
    minor_patch = mpatches.Patch(color='gray', label='Minor Faults')
    major_patch = mpatches.Patch(color='black', label='Major Faults')
    ax.legend(handles=[minor_patch, major_patch], fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    path = str(out_dir / f"hot_cold_pgfaults_pass_{pass_num}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path



def _plot_ood_freq(freq_dist: dict, out_dir: Path) -> Optional[str]:
    """Ranked bar chart of OOD vector query hit counts."""
    counts = sorted(freq_dist.values(), reverse=True)
    if not counts:
        return None

    fig, ax = plt.subplots(figsize=(min(20, max(8, len(counts) // 50)), 4))
    x = np.arange(len(counts))
    ax.bar(x, counts, color="#F59E0B", width=1.0, alpha=0.85)
    ax.set_xlabel("OOD vector rank (most queried -> least)", fontsize=11)
    ax.set_ylabel("Query count", fontsize=11)
    ax.set_title("OOD Query Frequency Distribution", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)

    path = str(out_dir / "ood_freq_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_ood_vs_id_latency(stats: dict, out_dir: Path) -> Optional[str]:
    """Grouped bar chart comparing ID vs OOD latency percentiles."""
    labels = ["p50", "p95", "p99"]
    id_vals  = [stats.get(f"id_latency_{p}")  for p in labels]
    ood_vals = [stats.get(f"ood_latency_{p}") for p in labels]

    valid = [(l, i, o) for l, i, o in zip(labels, id_vals, ood_vals)
             if i is not None or o is not None]
    if not valid:
        return None
    labels, id_vals, ood_vals = zip(*valid)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - width/2, id_vals,  width, label="In-Distribution", color="#3B82F6", alpha=0.85)
    b2 = ax.bar(x + width/2, ood_vals, width, label="OOD",             color="#EF4444", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("ID vs OOD Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    path = str(out_dir / "ood_vs_id_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_freshness_rerank_breakdown(stats: dict, out_dir: Path) -> Optional[str]:
    """Grouped bar chart comparing ANN search vs re-ranking average latency."""
    search_ms = stats.get("avg_search_latency_ms")
    rerank_ms = stats.get("avg_rerank_latency_ms")

    if search_ms is None or rerank_ms is None:
        return None

    labels = ["ANN Search", "Re-Ranking"]
    values = [search_ms, rerank_ms]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#3B82F6", "#F59E0B"], alpha=0.85, width=0.5)

    ax.set_ylabel("Average Latency (ms)", fontsize=11)
    ax.set_title("Search vs Re-Ranking Latency Breakdown", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, bars)

    # Add total annotation
    total = search_ms + rerank_ms
    ax.annotate(f"Total: {total:.2f} ms",
                xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", fontsize=10, fontstyle="italic", color="#6B7280")

    path = str(out_dir / "freshness_rerank_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cold_start_timeline(stats: dict, out_dir: Path) -> Optional[str]:
    """Full run timeline; red bands for restart/sleep; red diamonds for cold-start;
    blue dots for warm; dashed warm-p99 reference line."""
    latencies       = stats.get("_raw_latencies", [])
    is_cold_flags   = stats.get("_is_cold_flags", [])

    if not latencies:
        return None

    fig, ax = plt.subplots(figsize=(16, 5))

    sequence = list(range(1, len(latencies) + 1))

    # Separate cold and warm queries
    cold_t, cold_l, warm_t, warm_l = [], [], [], []
    
    for t, l, ic in zip(sequence, latencies, is_cold_flags):
        if ic:
            cold_t.append(t)
            cold_l.append(l)
        else:
            warm_t.append(t)
            warm_l.append(l)

    # Warm queries
    ax.scatter(warm_t, warm_l, c="#3B82F6", s=18, alpha=0.55,
               label="Warm query", zorder=3)
    # Connecting line to show the curve
    ax.plot(sequence, latencies, color="gray", linewidth=0.5, alpha=0.3, zorder=2)
    # Cold-start queries (prominent red diamonds)
    ax.scatter(cold_t, cold_l, c="#EF4444", s=120, alpha=0.9,
               marker="D", edgecolors="white", linewidths=1.0,
               label="Cold-start query", zorder=5)

    # Dashed warm-p99 reference line
    if warm_l:
        warm_p99 = float(np.percentile(warm_l, 99))
        ax.axhline(warm_p99, color="#6366F1", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Warm p99 = {warm_p99:.1f}ms", zorder=4)

    ax.set_xlabel("Query Sequence Index", fontsize=12)
    ax.set_ylabel("Query Latency (ms)", fontsize=12)
    ax.set_title("Cold-Start Workload Timeline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    path = str(out_dir / "cold_start_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cold_start_warmup_curve(stats: dict, out_dir: Path) -> Optional[str]:
    """One subplot per cycle; X = query number in burst; shows latency decaying
    from cold-start (red diamond) to steady-state (green square)."""
    per_cycle = stats.get("_per_cycle", [])
    cycles_with_warmup = [
        cs for cs in per_cycle
        if cs.get("warmup_latencies_ms") and len(cs["warmup_latencies_ms"]) > 1
    ]
    if not cycles_with_warmup:
        return None

    n_cycles = len(cycles_with_warmup)
    fig, axes = plt.subplots(1, n_cycles, figsize=(6 * n_cycles, 4),
                             squeeze=False, sharey=True)

    colors = ["#EF4444", "#F59E0B", "#22C55E", "#3B82F6", "#9333EA",
              "#EC4899", "#6366F1", "#14B8A6"]

    for col, cs in enumerate(cycles_with_warmup):
        ax = axes[0, col]
        wl = cs["warmup_latencies_ms"]
        x = list(range(1, len(wl) + 1))
        color = colors[col % len(colors)]

        ax.plot(x, wl, color=color, linewidth=2, marker="o", markersize=5)
        ax.fill_between(x, wl, alpha=0.1, color=color)

        # Cold-start query (red diamond)
        ax.scatter([1], [wl[0]], color="#EF4444", s=120, marker="D",
                   edgecolors="white", linewidth=1.2, zorder=5,
                   label=f"Cold-start: {wl[0]:.0f}ms")
        # Steady-state (green square)
        ax.scatter([len(wl)], [wl[-1]], color="#22C55E", s=80, marker="s",
                   edgecolors="white", linewidth=1.0, zorder=5,
                   label=f"Steady: {wl[-1]:.0f}ms")

        ax.set_xlabel("Query # in burst", fontsize=11)
        if col == 0:
            ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_title(f"Cycle {cs['cycle']}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)

    fig.suptitle("Warm-up Curve (Latency Decay After Restart)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = str(out_dir / "warmup_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cold_warm_latency(stats: dict, out_dir: Path) -> Optional[str]:
    """Grouped bar chart at p50/p95/p99 with recall degradation annotation."""
    labels = ["p50", "p95", "p99"]
    cold_vals = [
        stats.get("cold_start_latency_p50"),
        stats.get("cold_start_latency_p95"),
        stats.get("cold_start_latency_p99"),
    ]
    warm_vals = [
        stats.get("warm_latency_p50"),
        stats.get("warm_latency_p95"),
        stats.get("warm_latency_p99"),
    ]

    valid = [(l, c, w) for l, c, w in zip(labels, cold_vals, warm_vals)
             if c is not None or w is not None]
    if not valid:
        return None
    labels, cold_vals, warm_vals = zip(*valid)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width/2, cold_vals, width, label="Cold-Start", color="#EF4444", alpha=0.85)
    b2 = ax.bar(x + width/2, warm_vals, width, label="Warm",       color="#3B82F6", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Cold-Start vs Warm Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    # Recall degradation annotation at the bottom
    cold_recall = stats.get("cold_recall")
    warm_recall = stats.get("warm_recall")
    recall_deg  = stats.get("recall_degradation")
    recall_parts = []
    if cold_recall is not None:
        recall_parts.append(f"Cold Recall@k: {cold_recall:.4f}")
    if warm_recall is not None:
        recall_parts.append(f"Warm Recall@k: {warm_recall:.4f}")
    if recall_deg is not None:
        recall_parts.append(f"Degradation: {recall_deg*100:.2f}%")
    if recall_parts:
        annotation = "  |  ".join(recall_parts)
        ax.annotate(annotation,
                    xy=(0.5, -0.12), xycoords="axes fraction",
                    ha="center", fontsize=9, fontstyle="italic", color="#6B7280")

    path = str(out_dir / "cold_warm_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_id_vs_outlier_latency(stats: dict, out_dir: Path) -> Optional[str]:
    """Grouped bar chart comparing normal (ID) vs outlier query latency at
    p50 / p95 / p99, with a degradation-ratio annotation."""
    labels   = ["p50", "p95", "p99"]
    id_vals  = [
        stats.get("normal_latency_p50_ms"),
        stats.get("normal_latency_p95_ms"),
        stats.get("normal_latency_p99_ms"),
    ]
    out_vals = [
        stats.get("outlier_latency_p50_ms"),
        stats.get("outlier_latency_p95_ms"),
        stats.get("outlier_latency_p99_ms"),
    ]

    valid = [
        (l, i, o)
        for l, i, o in zip(labels, id_vals, out_vals)
        if i is not None or o is not None
    ]
    if not valid:
        return None
    labels, id_vals, out_vals = zip(*valid)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width / 2, id_vals,  width,
                label="Normal (ID)", color="#3B82F6", alpha=0.85)
    b2 = ax.bar(x + width / 2, out_vals, width,
                label="Outlier",     color="#EF4444", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Normal vs Outlier Query Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    # Per-class QPS + degradation ratio annotation
    parts: list[str] = []
    normal_qps  = stats.get("normal_qps")
    outlier_qps = stats.get("outlier_qps")
    degradation = stats.get("outlier_latency_degradation")
    if normal_qps  is not None:
        parts.append(f"Normal QPS: {normal_qps:.1f}")
    if outlier_qps is not None:
        parts.append(f"Outlier QPS: {outlier_qps:.1f}")
    if degradation is not None:
        parts.append(f"Latency degradation (p95): {degradation:.2f}×")
    if parts:
        ax.annotate("  |  ".join(parts),
                    xy=(0.5, -0.12), xycoords="axes fraction",
                    ha="center", fontsize=9, fontstyle="italic", color="#6B7280")

    path = str(out_dir / "id_vs_outlier_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_id_vs_outlier_recall(stats: dict, out_dir: Path) -> Optional[str]:
    """Side-by-side bar chart comparing recall@k for normal vs outlier queries,
    with zero-result rates and recall-degradation annotations."""
    normal_recall  = stats.get("normal_recall")
    outlier_recall = stats.get("outlier_recall")

    if normal_recall is None and outlier_recall is None:
        return None

    categories   = []
    recall_vals  = []
    bar_colors   = []

    if normal_recall is not None:
        categories.append("Normal (ID)")
        recall_vals.append(normal_recall)
        bar_colors.append("#3B82F6")
    if outlier_recall is not None:
        categories.append("Outlier")
        recall_vals.append(outlier_recall)
        bar_colors.append("#EF4444")

    fig, ax = plt.subplots(figsize=(6, 5))
    x    = np.arange(len(categories))
    bars = ax.bar(x, recall_vals, color=bar_colors, alpha=0.85, width=0.45)

    # Value labels on bars
    for bar, val in zip(bars, recall_vals):
        ax.annotate(f"{val:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Recall@k", fontsize=11)
    ax.set_ylim(0, min(1.1, max(recall_vals) * 1.25) if recall_vals else 1.1)
    ax.set_title("Normal vs Outlier Recall@k", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)

    # Annotation row: zero-result rates + recall degradation
    parts: list[str] = []
    nzr = stats.get("normal_zero_result_rate")
    ozr = stats.get("outlier_zero_result_rate")
    deg = stats.get("outlier_recall_degradation")
    if nzr is not None:
        parts.append(f"Normal zero-result rate: {nzr*100:.1f}%")
    if ozr is not None:
        parts.append(f"Outlier zero-result rate: {ozr*100:.1f}%")
    if deg is not None:
        parts.append(f"Recall degradation: {deg*100:.2f}%")
    if parts:
        ax.annotate("  |  ".join(parts),
                    xy=(0.5, -0.12), xycoords="axes fraction",
                    ha="center", fontsize=9, fontstyle="italic", color="#6B7280")

    path = str(out_dir / "id_vs_outlier_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_ttfsq_and_cold_latency_per_cycle(stats: dict, out_dir: Path) -> Optional[str]:
    """Two subplots showing TTFSQ (seconds) and Cold Latency (ms) per cycle."""
    per_cycle = stats.get("_per_cycle", [])
    if not per_cycle:
        return None

    cycles  = [cs["cycle"] for cs in per_cycle]
    ttfsqs  = [cs.get("ttfsq_s", 0) for cs in per_cycle]
    cold_lats = [cs.get("cold_start_latency_ms", 0) for cs in per_cycle]

    if not any(t > 0 for t in ttfsqs) and not any(l > 0 for l in cold_lats):
        return None

    ttfsq_mean = float(np.mean(ttfsqs)) if ttfsqs else 0
    cold_mean = float(np.mean(cold_lats)) if cold_lats else 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(7, len(cycles) * 1.5), 8), sharex=True)
    x = np.arange(len(cycles))
    width = 0.5

    # Top Plot: TTFSQ (seconds)
    bars1 = ax1.bar(x, ttfsqs, color="#6366F1", alpha=0.85, width=width, label="TTFSQ (s)")
    ax1.axhline(ttfsq_mean, color="#6366F1", linestyle="--", linewidth=1.5, alpha=0.6, label=f"Mean TTFSQ = {ttfsq_mean:.3f}s")
    ax1.set_ylabel("Time (seconds)", fontsize=11)
    ax1.set_title("Time To First Successful Query (TTFSQ) per Cycle", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    _bar_labels(ax1, bars1)

    # Bottom Plot: Cold Latency (ms)
    bars2 = ax2.bar(x, cold_lats, color="#EF4444", alpha=0.85, width=width, label="Cold Latency (ms)")
    ax2.axhline(cold_mean, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.6, label=f"Mean Cold Latency = {cold_mean:.1f}ms")
    ax2.set_ylabel("Latency (ms)", fontsize=11)
    ax2.set_title("First Query Latency (Cold Start) per Cycle", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Cycle {c}" for c in cycles], fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    _bar_labels(ax2, bars2)
    
    plt.tight_layout()
    path = str(out_dir / "ttfsq_and_cold_latency_per_cycle.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_dedup_phase_latency_distribution(stats: dict, out_dir: Path) -> Optional[str]:
    """Side-by-side violin plot comparing Phase 1 (bloom) vs Phase 2 (LSH) latency."""
    phase1 = np.array(stats.get("_phase1_latencies", []), dtype=float)
    phase2 = np.array(stats.get("_phase2_latencies", []), dtype=float)
 
    if len(phase1) == 0 and len(phase2) == 0:
        return None
 
    data   = []
    labels = []
    colors = []
 
    if len(phase1) > 0:
        data.append(phase1)
        labels.append("Phase 1\n(Bloom Filter)")
        colors.append("#3B82F6")
    if len(phase2) > 0:
        data.append(phase2)
        labels.append("Phase 2\n(LSH Search)")
        colors.append("#EF4444")
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=True)
 
    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
 
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    parts["cbars"].set_color("gray")
    parts["cmaxes"].set_color("gray")
    parts["cmins"].set_color("gray")
 
    # Annotate mean + p99 per phase
    for i, (arr, label) in enumerate(zip(data, labels)):
        mean_v = float(np.mean(arr))
        p99_v  = float(np.percentile(arr, 99))
        ax.annotate(
            f"mean={mean_v:.3f}ms\np99={p99_v:.3f}ms",
            xy=(i, float(np.percentile(arr, 75))),
            xytext=(30, 10), textcoords="offset points",
            fontsize=8, color="#374151",
            arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=0.8),
        )
 
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Phase 1 vs Phase 2 Latency Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
 
    path = str(out_dir / "dedup_phase_latency_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
def _plot_dedup_duplicate_rate_progress(stats: dict, out_dir: Path) -> Optional[str]:
    """Cumulative duplicate rate as ingestion progresses.
 
    Expects stats['_dedup_progress'] = list of (vectors_processed, is_duplicate: bool)
    emitted per-vector from the workload loop.
    """
    progress = stats.get("_dedup_progress", [])
    if not progress:
        return None
 
    processed  = np.array([p for p, _ in progress], dtype=float)
    is_dup     = np.array([d for _, d in progress], dtype=float)
    cum_dups   = np.cumsum(is_dup)
    cum_rate   = cum_dups / (np.arange(len(is_dup)) + 1)
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    ax.plot(processed, cum_rate, color="#7C3AED", linewidth=1.8,
            label="Cumulative duplicate rate")
    ax.fill_between(processed, cum_rate, alpha=0.12, color="#7C3AED")
 
    # Final rate annotation
    final_rate = float(cum_rate[-1])
    ax.axhline(final_rate, color="#EF4444", linestyle="--", linewidth=1.2,
               label=f"Final rate: {final_rate:.3f}")
 
    # Jaccard threshold annotation
    jt = stats.get("jaccard_threshold")
    if jt is not None:
        ax.annotate(f"Jaccard threshold: {jt}",
                    xy=(0.02, 0.93), xycoords="axes fraction",
                    fontsize=9, color="#6B7280")
 
    ax.set_xlabel("Vectors processed", fontsize=12)
    ax.set_ylabel("Cumulative duplicate rate", fontsize=12)
    ax.set_title("Duplicate Rate Over Ingestion Progress", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.05, final_rate * 2.5 + 0.05))
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)
 
    path = str(out_dir / "dedup_duplicate_rate_progress.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
def _plot_dedup_bloom_fp_curve(stats: dict, out_dir: Path) -> Optional[str]:
    """Theoretical bloom filter FP rate curve across capacity values,
    with the current run's operating point marked.
 
    FP rate formula: (1 - e^(-k * n / m))^k
    where k = optimal hash count, m = bit array size, n = items inserted.
    """
    import math
 
    bloom_capacity   = stats.get("bloom_capacity",   100_000)
    bloom_error_rate = stats.get("bloom_error_rate", 0.01)
    n_inserted       = stats.get("inserted", bloom_capacity)
 
    def optimal_m(n, p):
        return int(-n * math.log(p) / (math.log(2) ** 2))
 
    def optimal_k(m, n):
        return max(1, int((m / n) * math.log(2)))
 
    def theoretical_fp(capacity, n_actual, error_rate):
        m = optimal_m(capacity, error_rate)
        k = optimal_k(m, capacity)
        exponent = -k * n_actual / m
        return (1 - math.exp(exponent)) ** k
 
    # Sweep capacity from 10% to 500% of current setting
    capacities = np.linspace(
        max(1000, bloom_capacity * 0.1),
        bloom_capacity * 5,
        200,
    )
    fp_rates = [theoretical_fp(int(c), n_inserted, bloom_error_rate)
                for c in capacities]
 
    # Current operating point
    current_fp = stats.get("bloom_false_positive_rate",
                            theoretical_fp(bloom_capacity, n_inserted, bloom_error_rate))
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.plot(capacities, fp_rates, color="#3B82F6", linewidth=1.8,
            label="Theoretical FP rate")
    ax.fill_between(capacities, fp_rates, alpha=0.10, color="#3B82F6")
 
    # Mark current operating point
    ax.scatter([bloom_capacity], [current_fp], color="#EF4444", s=120,
               zorder=5, label=f"Current setting\ncapacity={bloom_capacity:,}\nFP={current_fp:.4f}")
    ax.axvline(bloom_capacity, color="#EF4444", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axhline(current_fp,     color="#EF4444", linestyle=":", linewidth=1.0, alpha=0.6)
 
    # Target error rate line
    ax.axhline(bloom_error_rate, color="#22C55E", linestyle="--", linewidth=1.2,
               label=f"Target error rate: {bloom_error_rate}")
 
    ax.set_xlabel("Bloom filter capacity (vectors)", fontsize=12)
    ax.set_ylabel("False positive rate", fontsize=12)
    ax.set_title("Bloom Filter FP Rate vs Capacity\n(theoretical, fixed n_inserted)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)
 
    path = str(out_dir / "dedup_bloom_fp_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
def _plot_dedup_ingest_latency_percentiles(stats: dict, out_dir: Path) -> Optional[str]:
    """Bar chart of ingest latency at P50 / P95 / P99."""
    percentiles = ["p50", "p95", "p99"]
    values      = [stats.get(f"ingest_latency_{p}") for p in percentiles]
 
    if all(v is None for v in values):
        return None
 
    values = [v or 0 for v in values]
 
    fig, ax = plt.subplots(figsize=(6, 5))
    x    = np.arange(len(percentiles))
    bars = ax.bar(x, values, color=["#22C55E", "#F59E0B", "#EF4444"],
                  alpha=0.85, width=0.5)
 
    _bar_labels(ax, bars)
 
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in percentiles], fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Ingest Latency Percentiles", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
 
    # Throughput annotation
    vps = stats.get("throughput_vps")
    if vps is not None:
        ax.annotate(f"Throughput: {vps:.0f} vec/s",
                    xy=(0.98, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    color="#6B7280", fontstyle="italic")
 
    path = str(out_dir / "dedup_ingest_latency_percentiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_ood_id_latency_cdf(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Overlapping CDF of ID vs OOD query latencies.
    Exposes whether OOD queries are slower/faster than ID queries.
    """
    id_lats  = np.array(stats.get("_raw_id_latencies",  []), dtype=float)
    ood_lats = np.array(stats.get("_raw_ood_latencies", []), dtype=float)
 
    if len(id_lats) == 0 and len(ood_lats) == 0:
        return None
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for lats, label, color in [
        (id_lats,  "ID queries",  "#3B82F6"),
        (ood_lats, "OOD queries", "#EF4444"),
    ]:
        if len(lats) == 0:
            continue
        sorted_lats = np.sort(lats)
        cdf         = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
        ax.plot(sorted_lats, cdf, label=label, color=color, linewidth=1.8)
 
        for pct, ls in [(50, "-"), (99, "--")]:
            val = np.percentile(lats, pct)
            ax.axvline(val, color=color, linestyle=ls, linewidth=1.0, alpha=0.6,
                       label=f"{label} p{pct}={val:.1f}ms")
 
    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title("ID vs OOD Query Latency CDF", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
 
    path = str(out_dir / "ood_id_latency_cdf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_ood_confusion_heatmap(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Confusion heatmap: rows = OOD classes, columns = ID classes.
    Cell value = how often that ID class appeared in k-NN results
    when that OOD class was queried. Normalized per row (fraction of
    total neighbor hits for that OOD class).
 
    Reveals which ID classes act as 'attractors' for each OOD class.
    """
    confusion = stats.get("_confusion_matrix", {})
    if not confusion:
        return None
 
    ood_classes = sorted(confusion.keys())
    id_classes  = sorted({cls for id_counts in confusion.values() for cls in id_counts})
 
    if not ood_classes or not id_classes:
        return None
 
    # Build matrix
    matrix = np.zeros((len(ood_classes), len(id_classes)), dtype=np.float32)
    for r, ood_cls in enumerate(ood_classes):
        row_counts = confusion[ood_cls]
        total      = sum(row_counts.values())
        if total > 0:
            for c, id_cls in enumerate(id_classes):
                matrix[r, c] = row_counts.get(id_cls, 0) / total
 
    fig_w = max(8,  len(id_classes)  * 0.6)
    fig_h = max(5,  len(ood_classes) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
 
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fraction of neighbor hits")
 
    ax.set_xticks(range(len(id_classes)))
    ax.set_yticks(range(len(ood_classes)))
    ax.set_xticklabels([str(c) for c in id_classes],  rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([str(c) for c in ood_classes], fontsize=9)
 
    ax.set_xlabel("ID class (returned neighbors)", fontsize=12)
    ax.set_ylabel("OOD class (query source)",      fontsize=12)
    ax.set_title("OOD → ID Confusion Heatmap\n(row-normalized neighbor class distribution)",
                 fontsize=13, fontweight="bold")
 
    # Annotate cells with value if matrix is small enough to read
    if len(id_classes) * len(ood_classes) <= 200:
        for r in range(len(ood_classes)):
            for c in range(len(id_classes)):
                val = matrix[r, c]
                if val > 0.01:
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if val < 0.6 else "white")
 
    plt.tight_layout()
    path = str(out_dir / "ood_confusion_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Temporal Freshness Workload plots
# ---------------------------------------------------------------------------

def _plot_freshness_rerank_breakdown(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Stacked bar: Stage 1 (ANN search) vs Stage 2 (freshness re-rank) mean latency.
    Shows where query wall-clock time is spent.
    """
    search_ms = stats.get("avg_search_latency_ms")
    rerank_ms = stats.get("avg_rerank_latency_ms")
    if search_ms is None or rerank_ms is None:
        return None

    total_ms = search_ms + rerank_ms
    pct_s    = search_ms / total_ms * 100 if total_ms > 0 else 0
    pct_r    = rerank_ms / total_ms * 100 if total_ms > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(["Mean Latency"], [search_ms],
           color="#3B82F6", alpha=0.88,
           label=f"Stage 1 · ANN search  {search_ms:.2f} ms  ({pct_s:.1f}%)")
    ax.bar(["Mean Latency"], [rerank_ms], bottom=[search_ms],
           color="#F59E0B", alpha=0.88,
           label=f"Stage 2 · Re-rank      {rerank_ms:.2f} ms  ({pct_r:.1f}%)")

    ax.text(0, total_ms + total_ms * 0.03,
            f"Total: {total_ms:.2f} ms",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Two-Stage Query Latency Breakdown", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, total_ms * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks([])

    path = str(out_dir / "freshness_rerank_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_freshness_ndcg_histogram(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Bar histogram of per-query NDCG@k scores bucketed into 10 equal bins [0, 1].
    Color ramps red → green to show quality gradient at a glance.
    """
    hist = stats.get("_ndcg_histogram", {})
    bins   = hist.get("bins", [])
    counts = hist.get("counts", [])
    if not bins or not counts or sum(counts) == 0:
        return None

    total     = sum(counts)
    fractions = [c / total for c in counts]

    # Red → yellow → green ramp matching NDCG score quality
    cmap       = plt.cm.RdYlGn
    bar_colors = [cmap(i / max(len(bins) - 1, 1)) for i in range(len(bins))]

    fig, ax = plt.subplots(figsize=(11, 5))
    x    = np.arange(len(bins))
    bars = ax.bar(x, counts, color=bar_colors, alpha=0.90,
                  edgecolor="white", linewidth=0.6)

    # Count + % label above each non-zero bar
    max_cnt = max(counts)
    for bar, cnt, frac in zip(bars, counts, fractions):
        if cnt > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_cnt * 0.012,
                f"{cnt:,}\n({frac:.1%})",
                ha="center", va="bottom", fontsize=7.5,
            )

    # Mean NDCG vertical reference line
    mean_ndcg = stats.get("ndcg_at_k")
    if mean_ndcg is not None:
        # Map mean_ndcg (0-1) to x-axis position (0 to len(bins)-1)
        mean_x = mean_ndcg * len(bins)
        ax.axvline(mean_x - 0.5, color="#6366F1", linewidth=2.0, linestyle="--",
                   label=f"Mean NDCG@k = {mean_ndcg:.4f}")
        ax.legend(fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("NDCG@k Score Bin", fontsize=12)
    ax.set_ylabel("Number of Queries", fontsize=12)
    ax.set_title("NDCG@k Score Distribution", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.grid(axis="y", alpha=0.3)

    path = str(out_dir / "freshness_ndcg_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_freshness_age_composition(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Side-by-side horizontal bar chart + donut chart showing the percentage
    breakdown of the final top-k returned documents by age bracket.
    Serves as a sanity check that the freshness decay is working as intended.
    """
    pct    = stats.get("_age_composition_pct", {})
    counts = stats.get("_age_composition_counts", {})
    if not pct:
        return None

    # Canonical bracket order oldest → freshest reversed for bar (freshest on top)
    BRACKETS = [
        ("under_7d",  "< 7 days",     "#22C55E"),
        ("8_30d",     "8 – 30 days",  "#3B82F6"),
        ("31_180d",   "31 – 180 days","#F59E0B"),
        ("over_180d", "> 180 days",   "#EF4444"),
    ]

    labels  = [b[1] for b in BRACKETS]
    values  = [pct.get(b[0], 0.0)    for b in BRACKETS]
    raw_cnt = [counts.get(b[0], 0)   for b in BRACKETS]
    colors  = [b[2] for b in BRACKETS]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Top-k Age Composition  (Freshness Sanity Check)",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Left: horizontal bar ────────────────────────────────────────────────
    y    = np.arange(len(labels))
    bars = ax_bar.barh(y, values, color=colors, alpha=0.88, height=0.55)

    for bar, val, cnt in zip(bars, values, raw_cnt):
        if val > 0:
            ax_bar.text(
                val + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%   ({cnt:,} docs)",
                va="center", fontsize=10,
            )

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=11)
    ax_bar.invert_yaxis()           # freshest age at the top
    ax_bar.set_xlabel("% of Returned Documents", fontsize=12)
    ax_bar.set_xlim(0, max(values) * 1.45 if max(values) > 0 else 100)
    ax_bar.set_title("By Age Bracket", fontsize=12)
    ax_bar.grid(axis="x", alpha=0.3)

    # ── Right: donut ────────────────────────────────────────────────────────
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if non_zero:
        pie_labels, pie_vals, pie_colors = zip(*non_zero)
        wedges, _, autotexts = ax_pie.pie(
            pie_vals,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            pctdistance=0.78,
            startangle=90,
            wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.2),
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax_pie.set_title("Distribution View", fontsize=12)
    else:
        ax_pie.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax_pie.transAxes, fontsize=12, color="#9CA3AF")

    plt.tight_layout()
    path = str(out_dir / "freshness_age_composition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path




# =============================================================================
# BURST RWD WORKLOAD PLOTS
# =============================================================================

def _plot_burst_qps_vs_latency(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Scatter plot: QPS vs p95 latency, coloured by burst phase.

    Every point is annotated with its pass number so you can trace
    which query pass corresponds to which dot (every Nth label to
    avoid overdraw when many passes overlap).
    """
    passes = stats.get("_query_passes", [])
    if not passes:
        return None

    timeline = stats.get("_burst_timeline") or stats.get("burst_timeline", [])
    phases   = ["baseline", "burst", "cooldown", "recovery"]
    colors   = {
        "baseline": "#3B82F6",
        "burst":    "#EF4444",
        "cooldown": "#10B981",
        "recovery": "#EAB308",
    }
    markers  = {
        "baseline": "o",
        "burst":    "^",
        "cooldown": "s",
        "recovery": "D",
    }

    def _pass_phase(t):
        """Map a pass time_s to its burst phase label."""
        for seg in timeline:
            t0 = seg.get("start_time_s", 0)
            t1 = seg.get("end_time_s", float("inf"))
            ph = seg.get("phase")
            if t0 <= t <= t1 and ph not in (None, "done"):
                return ph
        return "baseline"

    # Bucket each pass
    buckets: dict[str, list[tuple]] = {ph: [] for ph in phases}
    for p in passes:
        t   = p.get("time_s")
        qps = p.get("qps") or p.get("bps")
        lat = p.get("query_latency_p95")
        pn  = p.get("pass", "?")
        if t is None or qps is None or lat is None:
            continue
        ph = _pass_phase(t)
        if ph in buckets:
            buckets[ph].append((qps, lat, pn))

    plotted_any = any(len(v) > 0 for v in buckets.values())
    if not plotted_any:
        return None

    fig, ax = plt.subplots(figsize=(11, 7))

    for ph in phases:
        pts = buckets[ph]
        if not pts:
            continue
        xs_  = [pt[0] for pt in pts]
        ys_  = [pt[1] for pt in pts]
        pns_ = [pt[2] for pt in pts]

        sc = ax.scatter(
            xs_, ys_,
            color=colors[ph],
            marker=markers[ph],
            label=f"{ph.capitalize()} (n={len(pts)})",
            alpha=0.80, edgecolors="white", linewidths=0.6, s=70,
            zorder=3,
        )

        # Annotate with pass number — label every point in small groups,
        # else every other point to reduce clutter.
        step = 1 if len(pts) <= 8 else 2
        for i, (xi, yi, pn) in enumerate(zip(xs_, ys_, pns_)):
            if i % step == 0:
                ax.annotate(
                    f"P{pn}",
                    (xi, yi),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=colors[ph],
                    clip_on=True,
                )

    ax.set_xlabel("Throughput (QPS)", fontsize=12)
    ax.set_ylabel("p95 Latency (ms)", fontsize=12)
    ax.set_title(
        "Burst RWD: QPS vs p95 Latency by Phase\n"
        "(each point = one query pass; annotated with pass number)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Phase", fontsize=9, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = str(out_dir / "burst_qps_vs_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_phase_isolated_cdfs(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Per-phase latency CDF. Uses _raw_latencies stored per pass under
    the _query_passes list. Phase is determined by matching pass time_s
    against the _burst_timeline.
    """
    passes = stats.get("_query_passes", [])
    if not passes:
        return None

    timeline = stats.get("_burst_timeline") or stats.get("burst_timeline", [])

    def _pass_phase(t):
        for seg in timeline:
            t0 = seg.get("start_time_s", 0)
            t1 = seg.get("end_time_s", float("inf"))
            ph = seg.get("phase")
            if t0 <= t <= t1 and ph not in (None, "done"):
                return ph
        return "baseline"

    phase_lats = {"baseline": [], "burst": [], "cooldown": [], "recovery": []}

    for p in passes:
        t = p.get("time_s")
        # _raw_latencies is the key injected by main.py per-pass
        lats = p.get("_raw_latencies") or p.get("latencies_ms") or p.get("batch_latencies_ms", [])
        if t is None or not lats:
            continue
        ph = _pass_phase(t)
        if ph in phase_lats:
            phase_lats[ph].extend(lats)

    colors = {
        "baseline": "#3B82F6",
        "burst":    "#EF4444",
        "cooldown": "#10B981",
        "recovery": "#EAB308",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for phase in ["baseline", "burst", "cooldown", "recovery"]:
        lats = phase_lats[phase]
        if not lats:
            continue
        sorted_data = np.sort(lats)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(
            sorted_data, yvals,
            color=colors[phase], linewidth=2,
            label=f"{phase.capitalize()} (n={len(lats):,})",
        )
        # Mark p50 / p95 / p99
        for pct, ls in [(50, ":"), (95, "--"), (99, "-.")]:
            val = float(np.percentile(sorted_data, pct))
            ax.axvline(val, color=colors[phase], linewidth=0.7, linestyle=ls, alpha=0.6)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(
        "Phase-Isolated Query Latency CDFs\n"
        "(vertical lines: p50 · · ·  p95 - - -  p99 -·-·)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    all_lats = [l for lats in phase_lats.values() for l in lats]
    if all_lats and max(all_lats) > 100:
        ax.set_xscale("log")
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    path = str(out_dir / "burst_phase_cdfs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_burst_qps_over_time(stats: dict, out_dir: Path) -> Optional[str]:
    passes = stats.get("_query_passes", [])
    if not passes:
        return None
        
    xs = [p.get("time_s") for p in passes if p.get("time_s") is not None]
    ys = [p.get("qps", p.get("bps")) for p in passes if p.get("time_s") is not None]
    
    if not xs or not ys:
        return None
        
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xs, ys, color="#8B5CF6", linewidth=1.5, marker="o", markersize=4, label="Throughput")
    
    _overlay_burst_phases(ax, stats)
    
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Throughput", fontsize=11, color="#8B5CF6")
    ax.set_title("Burst RWD: Throughput Over Time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    path = str(out_dir / "burst_throughput_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_burst_phase_boxplots(stats: dict, out_dir: Path) -> Optional[str]:
    """
    Box-and-whisker plot of query latency per phase segment.
    Uses _raw_latencies stored on each pass (key injected by main.py).
    """
    passes = stats.get("_query_passes", [])
    timeline = stats.get("_burst_timeline") or stats.get("burst_timeline", [])
    if not passes or not timeline:
        return None

    def _pass_phase(t):
        for seg in timeline:
            t0 = seg.get("start_time_s", 0)
            t1 = seg.get("end_time_s", float("inf"))
            ph = seg.get("phase")
            if t0 <= t <= t1 and ph not in (None, "done"):
                return ph
        return "baseline"

    PHASE_COLOR = {
        "baseline":  "#3B82F6",
        "burst":     "#EF4444",
        "cooldown":  "#10B981",
        "recovery":  "#EAB308",
    }

    # One box per timeline segment that has data
    groups, labels, colors = [], [], []
    for b in timeline:
        start = b.get("start_time_s", 0)
        end   = b.get("end_time_s", float("inf"))
        phase = b.get("phase")
        idx   = b.get("burst_index", "")
        if phase == "done":
            continue

        lats = []
        for p in passes:
            t = p.get("time_s")
            if t is not None and start <= t <= end:
                # prefer _raw_latencies (full per-query list)
                raw = p.get("_raw_latencies") or p.get("latencies_ms") or p.get("batch_latencies_ms", [])
                lats.extend(raw)

        if lats:
            groups.append(lats)
            name = f"{phase.capitalize()}\n{idx}" if idx != "" else phase.capitalize()
            labels.append(name)
            colors.append(PHASE_COLOR.get(phase, "#6B7280"))

    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.2), 6))

    bplot = ax.boxplot(groups, patch_artist=True, tick_labels=labels, showfliers=False)

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    for median in bplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel("Query Latency (ms)", fontsize=11)
    ax.set_title(
        "Query Latency Distribution per Phase Segment\n(box = IQR, whiskers = 1.5×IQR, outliers hidden)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    path = str(out_dir / "burst_phase_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _overlay_burst_phases(ax, stats: dict) -> None:
    timeline = stats.get("_burst_timeline", [])
    first_burst = True
    first_recovery = True
    for p in timeline:
        start = p.get("start_time_s")
        end = p.get("end_time_s")
        phase = p.get("phase")
        if start is None or end is None:
            continue
        if phase == "burst":
            ax.axvspan(start, end, color="#EF4444", alpha=0.15, zorder=0, label="Burst Phase" if first_burst else "_nolegend_")
            first_burst = False
        elif phase == "recovery":
            ax.axvspan(start, end, color="#EAB308", alpha=0.15, zorder=0, label="Recovery Phase" if first_recovery else "_nolegend_")
            first_recovery = False

def _plot_sparse_workload_metrics(metrics: Dict[str, Any], out_dir: Path) -> List[str]:
    """
    1. Latency Recovery Curve (Timeline with idle bands & overlap per cycle)
    2. Page Faults vs Latency Scatter Plot
    """
    os.makedirs(out_dir, exist_ok=True)
    generated_paths = []
    
    # Extract data
    cycle_stats = metrics.get("_per_cycle", [])
    if not cycle_stats:
        print("[Plotting] No cycle stats available to plot.")
        return generated_paths

    idle_periods = metrics.get("_idle_periods", [])
    
    # -------------------------------------------------------------------------
    # Plot 1a: Timeline Latency with Idle Bands
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    all_t = []
    all_lats = []
    
    for cs in cycle_stats:
        queries = cs.get("queries", [])
        for q in queries:
            all_t.append(q["timestamp"])
            all_lats.append(q["latency_ms"])
            
    if all_t:
        plt.plot(all_t, all_lats, marker='.', linestyle='-', color='#1f77b4', markersize=4, alpha=0.8, label="Query Latency")
        
        # Draw idle bands
        for i, period in enumerate(idle_periods):
            label = "Idle Phase" if i == 0 else None
            plt.axvspan(period["start_s"], period["end_s"], color='gray', alpha=0.2, label=label)
            
        plt.title("Sparse Workload: Query Latency over Time")
        plt.xlabel("Time (seconds from workload start)")
        plt.ylabel("Latency (ms)")
        plt.yscale('log')  # Log scale often helps visualize the massive cold-start spikes
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = str(out_dir / "sparse_timeline.png")
        plt.savefig(path, dpi=300)
        plt.close()
        generated_paths.append(path)

    # -------------------------------------------------------------------------
    # Plot 1b: Latency Recovery Curve (Overlapping Cycles)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    for cs in cycle_stats:
        queries = cs.get("queries", [])
        if not queries:
            continue
        
        x_pos = [q["timestamp"] for q in queries]
        y_lat = [q["latency_ms"] for q in queries]
        
        plt.plot(x_pos, y_lat, marker='.', linestyle='-', markersize=3, alpha=0.7, 
                 label=f"Cycle {cs['cycle']} (Idle: {cs.get('idle_duration_actual_s', 0):.0f}s)")

    plt.title("Latency Recovery Curve per Active Cycle")
    plt.xlabel("Time (seconds from workload start)")
    plt.ylabel("Latency (ms)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = str(out_dir / "latency_recovery_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    generated_paths.append(path)

    return generated_paths

def _plot_warmup_curve(stats: dict, out_dir: Path) -> Optional[str]:
    """Per-cycle latency decay (warmup curve)."""
    per_cycle = stats.get("_per_cycle", [])
    cycles_with_warmup = [
        cs for cs in per_cycle
        if cs.get("warmup_latencies_ms") and len(cs["warmup_latencies_ms"]) > 1
    ]
    if not cycles_with_warmup:
        return None

    n_cycles = len(cycles_with_warmup)
    fig, axes = plt.subplots(1, n_cycles, figsize=(6 * n_cycles, 4),
                             squeeze=False, sharey=True)

    colors = ["#EF4444", "#F59E0B", "#22C55E", "#3B82F6", "#9333EA",
              "#EC4899", "#6366F1", "#14B8A6"]

    for col, cs in enumerate(cycles_with_warmup):
        ax = axes[0, col]
        wl = cs["warmup_latencies_ms"]
        x = list(range(1, len(wl) + 1))
        color = colors[col % len(colors)]

        ax.plot(x, wl, color=color, linewidth=2, marker="o", markersize=5)
        ax.fill_between(x, wl, alpha=0.1, color=color)

        # Mark the cold-start query
        ax.scatter([1], [wl[0]], color="#EF4444", s=120, marker="D",
                   edgecolors="white", linewidth=1.2, zorder=5,
                   label=f"Cold-Latency: {wl[0]:.0f}ms")

        # Mark the last query (steady-state)
        ax.scatter([len(wl)], [wl[-1]], color="#22C55E", s=80, marker="s",
                   edgecolors="white", linewidth=1.0, zorder=5,
                   label=f"Steady: {wl[-1]:.0f}ms")

        ax.set_xlabel("Query # after idle", fontsize=11)
        if col == 0:
            ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_title(f"Cycle {cs['cycle']}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)

    fig.suptitle("Warm-up Curve (Latency Decrease After idle phase)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = str(out_dir / "warmup_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# Deduplication Workload Plots
# =============================================================================

def _plot_dedup_gt_histogram(stats: Dict[str, Any], out_dir: Path) -> Optional[str]:
    """Plot distribution of GT nearest-neighbour scores."""
    gt_nn_score = np.array(stats["_gt_nn_score"])
    if len(gt_nn_score) == 0:
        return None

    is_dense = stats.get("signature_dim_bits", 1) == 0
    threshold_label = "Radius" if is_dense else "Threshold"
    threshold_val = stats.get("dense_radius", stats.get("jaccard_threshold", 0.8)) if is_dense else stats.get("jaccard_threshold", 0.8)
    xlabel = "Nearest-Neighbour Distance" if is_dense else "Max Jaccard Similarity to Base"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gt_nn_score, bins=50, color="#3498db", edgecolor="black", alpha=0.7)
    ax.axvline(threshold_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"{threshold_label} ({threshold_val})")
    ax.set_title("GT: Nearest Neighbour Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count of query vectors", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = str(out_dir / "dedup_gt_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_dedup_recall_precision_vs_threshold(stats: Dict[str, Any], out_dir: Path) -> Optional[str]:
    """Plot recall and precision across different Jaccard thresholds."""
    sweep = stats["_threshold_sweep"]
    if not sweep:
        return None

    thresholds = [row["threshold"] for row in sweep]
    recalls = [row["recall"] for row in sweep]
    precisions = [row["precision"] for row in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, recalls, marker='o', label="Recall", color="#2ecc71", linewidth=2)
    ax.plot(thresholds, precisions, marker='s', label="Precision", color="#e67e22", linewidth=2)
    
    current_threshold = stats.get("jaccard_threshold", 0.8)
    ax.axvline(current_threshold, color="#e74c3c", linestyle="--", alpha=0.5, label="Current Config")

    ax.set_title("Recall & Precision vs. Jaccard Threshold", fontsize=14, fontweight="bold")
    ax.set_xlabel("Jaccard Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = str(out_dir / "dedup_recall_precision_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_dedup_duplicate_rate_over_time(stats: Dict[str, Any], out_dir: Path) -> Optional[str]:
    """Plot the running duplicate rate as ingestion proceeds."""
    progress = stats["_dedup_progress"]
    if not progress:
        return None

    # progress is list of (processed_count, is_duplicate)
    processed = [p[0] for p in progress]
    is_dup = [p[1] for p in progress]
    
    cumulative_dups = np.cumsum(is_dup)
    running_dup_rate = cumulative_dups / np.array(processed)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(processed, running_dup_rate, color="#9b59b6", linewidth=2)
    
    ax.set_title("Running Duplicate Rate", fontsize=14, fontweight="bold")
    ax.set_xlabel("Vectors Processed", fontsize=12)
    ax.set_ylabel("Duplicate Rate", fontsize=12)
    ax.grid(True, alpha=0.3)

    path = str(out_dir / "dedup_duplicate_rate_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_dedup_phase_latencies(stats: Dict[str, Any], out_dir: Path) -> Optional[str]:
    """Boxplot comparing Phase 1 (Bloom) vs Phase 2 (LSH) latencies."""
    p1 = stats.get("_phase1_latencies", [])
    p2 = stats.get("_phase2_latencies", [])
    
    if not p1 and not p2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    
    data = []
    labels = []
    if p1:
        data.append(p1)
        labels.append(f"Phase 1 (Bloom)\nAvg: {np.mean(p1):.2f}ms")
    if p2:
        data.append(p2)
        labels.append(f"Phase 2 (LSH)\nAvg: {np.mean(p2):.2f}ms")

    ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#ecf0f1", color="#34495e"),
               medianprops=dict(color="#c0392b", linewidth=2))
               
    ax.set_title("Pipeline Latency Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    path = str(out_dir / "dedup_phase_latency_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# =============================================================================
# Deduplication Workload Plots
# =============================================================================

def _plot_dedup_gt_histogram(stats: Dict[str, Any], output_dir: Path) -> Optional[str]:
    jaccards = stats.get("_gt_nn_score", [])
    if not jaccards: return None
    
    is_dense = stats.get("signature_dim_bits", 1) == 0
    title = "Ground Truth Distance Distribution" if is_dense else "Ground Truth Max Jaccard Distribution"
    xlabel = "Distance (L2/Cosine)" if is_dense else "Max Jaccard Similarity"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(jaccards, bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    outpath = output_dir / "dedup_gt_histogram.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return str(outpath)

def _plot_dedup_recall_vs_pass(stats: Dict[str, Any], output_dir: Path) -> Optional[str]:
    passes = stats.get("_query_passes", [])
    if not passes: return None
    
    pass_nums = [p.get("pass", i+1) for i, p in enumerate(passes)]
    recalls = [p.get("dedup_recall", 0.0) for p in passes]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pass_nums, recalls, marker='o', linestyle='-', color='purple', linewidth=2)
    ax.set_title("Deduplication Recall vs Pass Number")
    ax.set_xlabel("Pass Number")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    outpath = output_dir / "dedup_recall_vs_pass.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return str(outpath)

def _plot_dedup_latency_timeline(stats: Dict[str, Any], output_dir: Path) -> Optional[str]:
    hashing = stats.get("_raw_hashing", [])
    bf = stats.get("_raw_bf_search", [])
    lsh = stats.get("_raw_lsh_search", [])
    insert = stats.get("_raw_insertion", [])
    
    if not hashing: return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    batches = range(len(hashing))
    
    ax.plot(batches, hashing, label="Hashing", alpha=0.8)
    ax.plot(batches, bf, label="Bloom Search", alpha=0.8)
    ax.plot(batches, lsh, label="LSH Search", alpha=0.8)
    ax.plot(batches, insert, label="Insertion", alpha=0.8)
    
    ax.set_title("Pipeline Stage Latency per Batch")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    outpath = output_dir / "dedup_latency_timeline.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return str(outpath)

def _plot_dedup_latency_cdfs(stats: Dict[str, Any], output_dir: Path) -> List[str]:
    lsh = stats.get("_raw_lsh_search", [])
    insert = stats.get("_raw_insertion", [])
    
    outpaths = []
    
    for name, data in [("lsh_search", lsh), ("insertion", insert)]:
        if not data: continue
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sorted_data, yvals, marker='.', linestyle='none', color='coral')
        ax.set_title(f"CDF of {name.replace('_', ' ').title()} Latency")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Cumulative Fraction")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        outpath = output_dir / f"dedup_cdf_{name}.png"
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        outpaths.append(str(outpath))
        
    return outpaths

def _plot_dedup_bloom_positives_timeline(stats: Dict[str, Any], output_dir: Path) -> Optional[str]:
    positives = stats.get("_raw_bloom_positives", [])
    lsh_rejected = stats.get("_raw_lsh_rejected", [])
    
    if not positives: return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    batches = range(len(positives))
    
    ax.plot(batches, positives, label="Bloom Positives (Candidates)", color='orange', alpha=0.9)
    ax.plot(batches, lsh_rejected, label="LSH Rejected (True Duplicates)", color='green', alpha=0.9)
    
    # Fill between to show false positives (amplification)
    ax.fill_between(batches, lsh_rejected, positives, color='red', alpha=0.2, label='False Positives (Waste)')
    
    ax.set_title("Bloom Filter Positives vs True Duplicates per Batch")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Count per Batch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    outpath = output_dir / "dedup_bloom_timeline.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return str(outpath)
