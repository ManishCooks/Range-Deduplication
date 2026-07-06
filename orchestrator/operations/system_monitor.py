"""
System Monitor - Background CPU/Memory/Disk sampler.

Two modes (auto-selected at construction time):

  Docker mode  : one or more container_names provided.
                 Each container is sampled in parallel via the Docker SDK.
                 Works on Linux (cgroupsv1 + cgroupsv2) and Windows Docker Desktop.

  Process mode : no containers provided.
                 Monitors a single process (PID) via psutil.
                 Defaults to the calling process (os.getpid()).
                 Override at any time with set_pid() / set_pipeann_pid().

Per-sample shape (answers "right now at time t"):
  cpu_pct           - CPU utilisation % (point-in-time snapshot)
  mem_mb            - RSS / container memory in MB (point-in-time snapshot)
  mem_pct           - memory % of limit (Docker only, point-in-time)
  read_mbps         - disk read throughput MB/s  (delta since previous sample)
  write_mbps        - disk write throughput MB/s (delta since previous sample)
  read_total_mb     - cumulative MB read   since monitor.start()
  write_total_mb    - cumulative MB written since monitor.start()
  net_rx_mbps       - container network RX throughput MB/s (Docker mode, delta)
  net_tx_mbps       - container network TX throughput MB/s (Docker mode, delta)
  net_rx_total_mb   - cumulative MB received since monitor.start() (Docker mode)
  net_tx_total_mb   - cumulative MB sent     since monitor.start() (Docker mode)
  pgfault_delta     - minor page fault count delta per sample interval
  pgmajfault_delta  - major page fault delta (OS cache eviction proxy; Docker exec)
  mem_cache_mb      - OS page cache memory in MB (Docker mode, point-in-time)

Opt-in via config: { "global": { "monitor_system": true } }
"""

import os
import threading
import time
import docker

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SystemMonitor:
    # Global reference to the active monitor instance
    GLOBAL_MONITOR = None

    def __init__(
        self,
        poll_interval_s: float = 0.3,
        container_names: Optional[List[str]] = None,
        pid: Optional[int] = None,           # process mode only; defaults to os.getpid()
        data_dir: Optional[str] = None,      # process mode only; directory to measure disk footprint
        pgfault_sample_every: int = 5,       # exec page-fault read every N docker polls (limits overhead)
    ):
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil is required: pip install psutil")

        self._interval             = poll_interval_s
        self._stop                 = threading.Event()
        self._thread               = None
        self._container_names      = container_names or []
        self._data_dir             = data_dir
        self._pgfault_sample_every = pgfault_sample_every
        self._pgfault_counter      = 0

        # ── Mode ──────────────────────────────────────────────────────────────
        self._mode = "docker" if self._container_names else "process"

        # ── Timestamps ────────────────────────────────────────────────────────
        # _sample_times[i] = seconds since monitor.start() when sample i landed.
        # Recorded after collection so rate denominators use real elapsed time.
        self._sample_times: List[float] = []
        self._start_time:   float       = 0.0

        # ── Docker series ─────────────────────────────────────────────────────
        # All containers are sampled in parallel; each gets its own sub-dict.
        #
        #   cpu            - CPU % at sample time
        #   mem_mb         - memory usage in MB at sample time
        #   mem_pct        - memory % of container limit at sample time
        #   read_mbps      - disk read throughput MB/s  (delta / elapsed)
        #   write_mbps     - disk write throughput MB/s (delta / elapsed)
        #   read_total_mb  - cumulative MB read  since monitor.start()
        #   write_total_mb - cumulative MB written since monitor.start()
        self.docker_stats: Dict[str, Dict[str, Any]] = {}
        for name in self._container_names:
            self.docker_stats[name] = {
                "cpu":              [],
                "mem_mb":           [],
                "mem_pct":          [],
                "read_mbps":        [],
                "write_mbps":       [],
                "read_total_mb":    [],
                "write_total_mb":   [],
                "disk_usage_mb":    [],
                "baseline_disk_mb": 0.0,
                # Network I/O (cumulative + instantaneous rate)
                "net_rx_mbps":      [],
                "net_tx_mbps":      [],
                "net_rx_total_mb":  [],
                "net_tx_total_mb":  [],
                # Page faults sampled via docker exec (delta per sampling event)
                "pgfault_delta":    [],
                "pgmajfault_delta": [],
                "mem_cache_mb":     [],
            }

        # Cumulative blkio baseline per container.
        # Set from the first valid sample so totals are relative to
        # monitor.start(), not to container start.
        self._blkio_baseline: Dict[str, Optional[Dict[str, float]]] = {
            name: None for name in self._container_names
        }

        # Previous blkio snapshot + monotonic timestamp per container.
        # Used to compute instantaneous MB/s between consecutive samples.
        # Value shape: {"read": float, "write": float, "time": float} | None
        self._blkio_prev: Dict[str, Optional[Dict[str, float]]] = {
            name: None for name in self._container_names
        }

        # Network I/O: cumulative baseline + previous snapshot per container.
        self._net_baseline: Dict[str, Optional[Dict[str, float]]] = {
            name: None for name in self._container_names
        }
        self._net_prev: Dict[str, Optional[Dict[str, float]]] = {
            name: None for name in self._container_names
        }

        # Page fault cumulative snapshot per container (delta computed on each exec).
        # Value shape: {"pgfault": int, "pgmajfault": int} | None
        self._pgfault_prev: Dict[str, Optional[Dict[str, int]]] = {
            name: None for name in self._container_names
        }

        # ── Process (PID) series ───────────────────────────────────────────────
        #   proc_cpu           - CPU % at sample time
        #   proc_mem_mb        - RSS in MB at sample time
        #   proc_read_mbps     - disk read throughput MB/s  (delta / elapsed)
        #   proc_write_mbps    - disk write throughput MB/s (delta / elapsed)
        #   proc_read_total_mb  - cumulative MB read  since set_pid() / start()
        #   proc_write_total_mb - cumulative MB written since set_pid() / start()
        self._proc:     Optional[Any] = None
        self._target_pid: Optional[int] = pid     # resolved in start() if still None

        self.proc_cpu:            List[float] = []
        self.proc_mem_mb:         List[float] = []
        self.proc_read_mbps:      List[float] = []
        self.proc_write_mbps:     List[float] = []
        self.proc_read_total_mb:  List[float] = []
        self.proc_write_total_mb: List[float] = []
        self.proc_disk_usage_mb:  List[float] = []
        # Page fault deltas (process mode)
        # Windows: num_page_faults is combined minor+major (no split available)
        # Linux:   minflt from /proc/<pid>/stat field 10, majflt from field 12
        self.proc_pgfault_delta:  List[int]   = []   # minor faults per sample (Windows: combined)
        self.proc_majflt_delta:   List[int]   = []   # major faults per sample (Linux only)

        self._proc_disk_baseline_mb: float = 0.0

        # io_counters snapshot at set_pid() / start() — for cumulative total.
        self._proc_io_baseline: Optional[Any] = None
        # Previous io_counters snapshot + monotonic time — for instantaneous rate.
        self._prev_proc_io:   Optional[Any] = None
        self._prev_proc_time: float         = 0.0

        # Previous page fault cumulative snapshot for delta computation.
        # Tuple: (pgfault_total, pgmajfault_total) | None
        self._proc_pgfault_prev: Optional[tuple] = None
        
        # State for snapshot() deltas
        self._snapshot_prev: Dict[str, Any] = {}

        # ── Docker client ─────────────────────────────────────────────────────
        self._docker_client = None
        if self._container_names:
            try:
                self._docker_client = docker.from_env()
            except Exception as e:
                print(f"[SystemMonitor] Docker unavailable: {e}")
                self._container_names = []
                self._mode = "process"

    # =========================================================================
    # Static helpers
    # =========================================================================

    @staticmethod
    def _sample_container(client, container_name: str) -> Optional[Dict[str, float]]:
        """Return a point-in-time snapshot from the Docker stats API.

        Returns cumulative raw blkio bytes; the caller computes both the
        instantaneous rate and the cumulative total relative to its baselines.

        Compatibility:
          Linux cgroupsv1         - blkio_stats.io_service_bytes_recursive populated.
          Linux cgroupsv2 22.04+  - blkio_stats empty; falls back to reading io.stat
                                    from the cgroup filesystem directly.
          Windows Docker Desktop  - blkio_stats populated via Docker daemon.
        """
        try:
            container = client.containers.get(container_name)
            s = container.stats(stream=False)

            # ── CPU ──────────────────────────────────────────────────────────
            cpu_delta    = (s["cpu_stats"]["cpu_usage"]["total_usage"]
                            - s["precpu_stats"]["cpu_usage"]["total_usage"])
            system_delta = (s["cpu_stats"]["system_cpu_usage"]
                            - s["precpu_stats"]["system_cpu_usage"])
            num_cpus     = s["cpu_stats"].get("online_cpus", 1)
            cpu_pct      = (
                (cpu_delta / system_delta) * num_cpus * 100.0
                if system_delta else 0.0
            )

            # ── Memory ───────────────────────────────────────────────────────
            mem_usage_mb = s["memory_stats"]["usage"] / (1024 ** 2)
            mem_limit_mb = s["memory_stats"]["limit"] / (1024 ** 2)

            # ── Block I/O — cgroupsv1 / Windows ──────────────────────────────
            bio = s.get("blkio_stats", {}).get("io_service_bytes_recursive") or []
            blkio_read_mb  = next((x["value"] for x in bio if x["op"] == "Read"),  0) / (1024 ** 2)
            blkio_write_mb = next((x["value"] for x in bio if x["op"] == "Write"), 0) / (1024 ** 2)

            # ── Block I/O — cgroupsv2 fallback (Linux only) ───────────────────
            # Safe to attempt on Windows; FileNotFoundError is caught silently.
            if blkio_read_mb == 0.0 and blkio_write_mb == 0.0:
                cid = container.id
                io_stat_paths = [
                    f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/io.stat",
                    f"/sys/fs/cgroup/docker/{cid}/io.stat",
                ]
                for path in io_stat_paths:
                    try:
                        total_read = total_write = 0
                        with open(path) as fh:
                            for line in fh:
                                parts = line.strip().split()
                                if len(parts) < 2:
                                    continue
                                kv = dict(p.split("=") for p in parts[1:] if "=" in p)
                                total_read  += int(kv.get("rbytes", 0))
                                total_write += int(kv.get("wbytes", 0))
                        blkio_read_mb  = total_read  / (1024 ** 2)
                        blkio_write_mb = total_write / (1024 ** 2)
                        break
                    except (FileNotFoundError, PermissionError):
                        continue

            # ── Network I/O ───────────────────────────────────────────────────
            # s["networks"] is a dict of interface → stats (eth0, eth1, ...).
            # Cumulative bytes since container start; caller converts to rate + total.
            networks  = s.get("networks", {})
            net_rx_mb = sum(iface.get("rx_bytes", 0) for iface in networks.values()) / (1024 ** 2)
            net_tx_mb = sum(iface.get("tx_bytes", 0) for iface in networks.values()) / (1024 ** 2)

            return {
                "cpu_pct":        round(cpu_pct, 2),
                "mem_usage_mb":   round(mem_usage_mb, 2),
                "mem_pct":        round(mem_usage_mb / mem_limit_mb * 100, 2) if mem_limit_mb else 0.0,
                # Raw cumulative bytes — rounded by caller after delta math
                "blkio_read_mb":  blkio_read_mb,
                "blkio_write_mb": blkio_write_mb,
                # Raw cumulative network bytes (rate + total computed by caller)
                "net_rx_mb":      net_rx_mb,
                "net_tx_mb":      net_tx_mb,
            }
        except Exception as e:
            print(f"[SystemMonitor] Failed to sample {container_name}: {e}")
            return None

    # =========================================================================
    # PID registration
    # =========================================================================

    def set_pid(self, pid: int) -> None:
        """Register the process PID to monitor in process mode.

        Call right after the target process / library starts.
        For in-process libraries (pybind11, ctypes) pass os.getpid().
        Resets both the cumulative baseline and the rate delta state.
        """
        self._target_pid = pid
        self._proc = psutil.Process(pid)
        self._proc.cpu_percent(interval=None)   # prime internal cpu_percent state
        try:
            io = self._proc.io_counters()
            self._proc_io_baseline = io     # cumulative total is relative to this
            self._prev_proc_io     = io     # rate delta starts here
            self._prev_proc_time   = time.monotonic()
        except (AttributeError, psutil.AccessDenied):
            self._proc_io_baseline = None
            self._prev_proc_io     = None
            self._prev_proc_time   = 0.0
        self._proc_pgfault_prev = None    # reset page fault delta baseline

    def set_process_pid(self, pid: int) -> None:
        """Alias for set_pid() — kept for backward compatibility."""
        self.set_pid(pid)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> "SystemMonitor":
        """Clear all series and start the background polling thread."""
        self._stop.clear()

        # ── Measure Docker disk baselines ─────────────────────────────────────
        if self._mode == "docker":
            for name in self._container_names:
                try:
                    container = self._docker_client.containers.get(name)
                    baseline = self._get_docker_size_mb(container)
                    self.docker_stats[name]["baseline_disk_mb"] = baseline
                except Exception as e:
                    print(f"[SystemMonitor] Warning: failed to get disk baseline for {name}: {e}")

        # ── Clear all series ──────────────────────────────────────────────────
        self._sample_times.clear()
        self._pgfault_counter = 0
        for lst in (
            self.proc_cpu, self.proc_mem_mb,
            self.proc_read_mbps,     self.proc_write_mbps,
            self.proc_read_total_mb, self.proc_write_total_mb,
            self.proc_disk_usage_mb,
            self.proc_pgfault_delta, self.proc_majflt_delta,
        ):
            lst.clear()

        for name in self.docker_stats:
            for key, series in self.docker_stats[name].items():
                if isinstance(series, list):
                    series.clear()

        # ── Reset baselines and delta state ───────────────────────────────────
        # Docker: re-captured from first valid sample after start().
        self._blkio_baseline = {name: None for name in self._container_names}
        self._blkio_prev     = {name: None for name in self._container_names}
        self._net_baseline   = {name: None for name in self._container_names}
        self._net_prev       = {name: None for name in self._container_names}
        self._pgfault_prev   = {name: None for name in self._container_names}

        # Process: reset so cumulative and rate both start from start() time.
        self._proc_pgfault_prev = None
        if self._proc is not None:
            try:
                io = self._proc.io_counters()
                self._proc_io_baseline = io
                self._prev_proc_io     = io
                self._prev_proc_time   = time.monotonic()
            except (AttributeError, psutil.AccessDenied):
                self._proc_io_baseline = None
                self._prev_proc_io     = None
                self._prev_proc_time   = 0.0
        
        self._snapshot_prev.clear()

        # ── Default PID in process mode ───────────────────────────────────────
        if self._mode == "process" and self._proc is None:
            self.set_pid(self._target_pid or os.getpid())

        # ── Measure process disk baseline ─────────────────────────────────────
        if self._mode == "process" and self._data_dir and os.path.exists(self._data_dir):
            total_size = 0
            for dirpath, _, filenames in os.walk(self._data_dir):
                for f in filenames:
                    try:
                        total_size += os.path.getsize(os.path.join(dirpath, f))
                    except FileNotFoundError:
                        pass
            self._proc_disk_baseline_mb = total_size / (1024 * 1024)

        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        """Stop polling and return aggregated stats dict."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

        import numpy as np

        def _agg(lst: List[float], key: str) -> Dict[str, float]:
            if not lst:
                return {}
            a = np.array(lst)
            return {
                f"{key}_mean": float(a.mean()),
                f"{key}_peak":  float(a.max()),
            }

        stats: Dict[str, Any] = {
            "samples":         len(self._sample_times),
            "poll_interval_s": self._interval,
            "mode":            self._mode,
        }

        if self._mode == "process":
            stats.update(_agg(self.proc_cpu,        "proc_cpu_pct"))
            stats.update(_agg(self.proc_mem_mb,     "proc_mem_mb"))
            stats.update(_agg(self.proc_read_mbps,  "proc_read_mbps"))
            stats.update(_agg(self.proc_write_mbps, "proc_write_mbps"))
            if self.proc_disk_usage_mb:
                stats.update(_agg(self.proc_disk_usage_mb, "proc_disk_usage_mb"))
            # Cumulative total = last recorded value
            if self.proc_read_total_mb:
                stats["proc_read_total_mb"]  = self.proc_read_total_mb[-1]
            if self.proc_write_total_mb:
                stats["proc_write_total_mb"] = self.proc_write_total_mb[-1]
            # Page fault totals over the entire monitored window
            if self.proc_pgfault_delta:
                stats["proc_pgfault_total"] = int(sum(self.proc_pgfault_delta))
            if self.proc_majflt_delta:
                stats["proc_majflt_total"]  = int(sum(self.proc_majflt_delta))
                stats["proc_majflt_peak"]   = int(max(self.proc_majflt_delta))

        else:   # docker
            for name, st in self.docker_stats.items():
                safe = name.replace("-", "_")
                stats.update(_agg(st["cpu"],        f"docker_{safe}_cpu_pct"))
                stats.update(_agg(st["mem_mb"],     f"docker_{safe}_mem_mb"))
                stats.update(_agg(st["mem_pct"],    f"docker_{safe}_mem_pct"))
                stats.update(_agg(st["read_mbps"],  f"docker_{safe}_read_mbps"))
                stats.update(_agg(st["write_mbps"], f"docker_{safe}_write_mbps"))

                # Network I/O
                stats.update(_agg(st["net_rx_mbps"], f"docker_{safe}_net_rx_mbps"))
                stats.update(_agg(st["net_tx_mbps"], f"docker_{safe}_net_tx_mbps"))
                if st["net_rx_total_mb"]:
                    stats[f"docker_{safe}_net_rx_total_mb"] = st["net_rx_total_mb"][-1]
                if st["net_tx_total_mb"]:
                    stats[f"docker_{safe}_net_tx_total_mb"] = st["net_tx_total_mb"][-1]

                # Page faults (sum of deltas = total faults during the monitored window)
                if st["pgmajfault_delta"]:
                    stats[f"docker_{safe}_pgmajfault_total"] = int(sum(st["pgmajfault_delta"]))
                    stats[f"docker_{safe}_pgmajfault_peak"]  = int(max(st["pgmajfault_delta"]))
                if st["pgfault_delta"]:
                    stats[f"docker_{safe}_pgfault_total"]    = int(sum(st["pgfault_delta"]))
                
                if st["mem_cache_mb"]:
                    stats.update(_agg(st["mem_cache_mb"], f"docker_{safe}_mem_cache_mb"))

                if st["disk_usage_mb"]:
                    stats.update(_agg(st["disk_usage_mb"], f"docker_{safe}_disk_usage_mb"))

                if st["read_total_mb"]:
                    stats[f"docker_{safe}_read_total_mb"]  = st["read_total_mb"][-1]
                if st["write_total_mb"]:
                    stats[f"docker_{safe}_write_total_mb"] = st["write_total_mb"][-1]

            # True deployment peak combined over all containers
            if self.docker_stats:
                lengths = [len(st["disk_usage_mb"]) for st in self.docker_stats.values() if st["disk_usage_mb"]]
                if lengths:
                    min_len = min(lengths)
                    if min_len > 0:
                        combined_disk = np.zeros(min_len)
                        for st in self.docker_stats.values():
                            if st["disk_usage_mb"]:
                                combined_disk += np.array(st["disk_usage_mb"][:min_len])
                        stats["docker_combined_disk_usage_mb_peak"] = float(combined_disk.max())
                        stats["docker_combined_disk_usage_mb_mean"] = float(combined_disk.mean())

        return stats

    # =========================================================================
    # Poll loop
    # =========================================================================

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ts = time.perf_counter() - self._start_time

                if self._mode == "docker":
                    self._sample_docker(ts)
                else:
                    self._sample_process(ts)

            except Exception as e:
                print(f"[SystemMonitor] poll error: {e}")

            self._stop.wait(timeout=self._interval)

    def _sample_docker(self, ts: float) -> None:
        """Sample all containers in parallel and append to their series.

        All containers fire concurrently so total wait = slowest single
        container, not the sum.  This matters because container.stats()
        blocks ~1-2 s on Windows Docker Desktop.

        A single monotonic timestamp is captured once before the executor
        fires so every container in this batch shares the same time reference
        for its rate denominator.
        """
        if not self._docker_client or not self._container_names:
            return

        now = time.monotonic()  # shared time reference for this batch

        with ThreadPoolExecutor(max_workers=len(self._container_names)) as ex:
            futures = {
                ex.submit(self._sample_container, self._docker_client, name): name
                for name in self._container_names
            }
            try:
                for f in as_completed(futures, timeout=8):
                    name   = futures[f]
                    try:
                        sample = f.result()
                    except Exception:
                        sample = None
                    if sample is None:
                        continue
                    st = self.docker_stats[name]
                    raw_read  = sample["blkio_read_mb"]
                    raw_write = sample["blkio_write_mb"]

                    # ── Cumulative baseline (relative to monitor.start()) ─────────
                    if self._blkio_baseline[name] is None:
                        self._blkio_baseline[name] = {"read": raw_read, "write": raw_write}

                    baseline    = self._blkio_baseline[name]
                    read_total  = raw_read  - baseline["read"]
                    write_total = raw_write - baseline["write"]

                    # ── Instantaneous rate (delta / actual elapsed time) ──────────
                    prev = self._blkio_prev[name]
                    if prev is not None:
                        elapsed = now - prev["time"]
                        if elapsed > 0:
                            read_mbps  = (raw_read  - prev["read"])  / elapsed
                            write_mbps = (raw_write - prev["write"]) / elapsed
                        else:
                            read_mbps = write_mbps = 0.0
                    else:
                        # First sample — no previous point to diff against.
                        read_mbps = write_mbps = 0.0

                    self._blkio_prev[name] = {"read": raw_read, "write": raw_write, "time": now}

                    # ── Network I/O delta (same pattern as blkio) ─────────────────
                    raw_net_rx = sample["net_rx_mb"]
                    raw_net_tx = sample["net_tx_mb"]

                    if self._net_baseline[name] is None:
                        self._net_baseline[name] = {"rx": raw_net_rx, "tx": raw_net_tx}

                    net_base     = self._net_baseline[name]
                    net_rx_total = raw_net_rx - net_base["rx"]
                    net_tx_total = raw_net_tx - net_base["tx"]

                    net_prev = self._net_prev[name]
                    if net_prev is not None:
                        elapsed = now - net_prev["time"]
                        if elapsed > 0:
                            net_rx_mbps = (raw_net_rx - net_prev["rx"]) / elapsed
                            net_tx_mbps = (raw_net_tx - net_prev["tx"]) / elapsed
                        else:
                            net_rx_mbps = net_tx_mbps = 0.0
                    else:
                        net_rx_mbps = net_tx_mbps = 0.0

                    self._net_prev[name] = {"rx": raw_net_rx, "tx": raw_net_tx, "time": now}

                    # ── Append to series ──────────────────────────────────────────
                    st["cpu"].append(sample["cpu_pct"])
                    st["mem_mb"].append(sample["mem_usage_mb"])
                    st["mem_pct"].append(sample["mem_pct"])
                    st["read_mbps"].append(round(read_mbps, 2))
                    st["write_mbps"].append(round(write_mbps, 2))
                    st["read_total_mb"].append(round(read_total, 4))
                    st["write_total_mb"].append(round(write_total, 4))
                    st["net_rx_mbps"].append(round(net_rx_mbps, 4))
                    st["net_tx_mbps"].append(round(net_tx_mbps, 4))
                    st["net_rx_total_mb"].append(round(net_rx_total, 6))
                    st["net_tx_total_mb"].append(round(net_tx_total, 6))

                    # Track Disk Footprint
                    try:
                        container = self._docker_client.containers.get(name)
                        curr_size_mb = self._get_docker_size_mb(container)
                        
                        # If the database dropped the collection, the size will drop below the initial baseline.
                        # We should reset the baseline to this new low watermark so we track the fresh ingestion.
                        if curr_size_mb < st.get("baseline_disk_mb", 0.0):
                            st["baseline_disk_mb"] = curr_size_mb
                            
                        net_size_mb = curr_size_mb - st.get("baseline_disk_mb", 0.0)
                        st["disk_usage_mb"].append(max(0.0, round(net_size_mb, 4)))
                    except Exception:
                        pass
            except TimeoutError:
                # One or more container stats calls timed out; skip this poll cycle
                return

        # ── Page faults via docker exec (throttled every N polls) ───────────
        self._pgfault_counter += 1
        if self._pgfault_counter % self._pgfault_sample_every == 0:
            self._sample_page_faults_docker()

        # Timestamp appended after all containers in this batch are done.
        self._sample_times.append(ts)

    def _sample_page_faults_docker(self) -> None:
        """
        Read pgfault / pgmajfault from inside each container via exec_run.

        Tries cgroup v2 (/sys/fs/cgroup/memory.stat) first, falls back to
        cgroup v1 (/sys/fs/cgroup/memory/memory.stat).  Called every
        _pgfault_sample_every polls to cap the exec_run overhead
        (~5-20 ms per container on Docker Desktop).

        On Windows Docker Desktop the memory_stats.stats blob returned by
        container.stats() is empty (confirmed), so exec_run is the only
        reliable path to cgroup page-fault counters.
        """
        if not self._docker_client:
            return
        for name in self._container_names:
            try:
                container = self._docker_client.containers.get(name)
                pgfault = pgmajfault = cache_bytes = 0

                # cgroup v2 (WSL2 / modern Linux)
                exit_code, out = container.exec_run(
                    "cat /sys/fs/cgroup/memory.stat", stdout=True
                )
                if exit_code == 0:
                    for line in out.decode(errors="ignore").splitlines():
                        parts = line.split()
                        if len(parts) == 2:
                            if parts[0] == "pgfault":
                                pgfault    = int(parts[1])
                            elif parts[0] == "pgmajfault":
                                pgmajfault = int(parts[1])
                            elif parts[0] in ("file", "cache", "total_cache"):
                                # cgroup v2 uses 'file' for page cache, v1 uses 'cache'
                                cache_bytes = int(parts[1])

                # cgroup v1 fallback (older Linux)
                if pgfault == 0:
                    exit_code, out = container.exec_run(
                        "awk '/^pgfault|^pgmajfault|^cache|^total_cache/{print}' "
                        "/sys/fs/cgroup/memory/memory.stat",
                        stdout=True,
                    )
                    if exit_code == 0:
                        for line in out.decode(errors="ignore").splitlines():
                            parts = line.split()
                            if len(parts) == 2:
                                if parts[0] == "pgfault":
                                    pgfault    = int(parts[1])
                                elif parts[0] == "pgmajfault":
                                    pgmajfault = int(parts[1])
                                elif parts[0] in ("cache", "total_cache"):
                                    cache_bytes = int(parts[1])

                st   = self.docker_stats[name]
                prev = self._pgfault_prev.get(name)
                if prev is not None:
                    st["pgfault_delta"].append(max(0, pgfault    - prev["pgfault"]))
                    st["pgmajfault_delta"].append(max(0, pgmajfault - prev["pgmajfault"]))
                else:
                    # First reading — emit 0 delta; establishes the baseline.
                    st["pgfault_delta"].append(0)
                    st["pgmajfault_delta"].append(0)
                
                st["mem_cache_mb"].append(round(cache_bytes / (1024 ** 2), 2))
                self._pgfault_prev[name] = {"pgfault": pgfault, "pgmajfault": pgmajfault}

            except Exception:
                pass   # Never disrupt the main sampling loop

    def _sample_process(self, ts: float) -> None:
        """Sample the monitored PID and append to the proc_* series."""
        if self._proc is None:
            return

        now = time.monotonic()

        try:
            with self._proc.oneshot():      # single /proc read for all fields
                cpu = self._proc.cpu_percent(interval=None)
                mem = self._proc.memory_info().rss / (1024 ** 2)
                try:
                    io = self._proc.io_counters()
                except (AttributeError, psutil.AccessDenied):
                    # io_counters() requires elevation on some Windows configs.
                    io = None

            self.proc_cpu.append(round(cpu, 2))
            self.proc_mem_mb.append(round(mem, 2))

            if io is not None:
                # Lazily capture baseline if set_pid() was called before start().
                if self._proc_io_baseline is None:
                    self._proc_io_baseline = io
                    self._prev_proc_io     = io
                    self._prev_proc_time   = now

                # Cumulative total since baseline
                read_total  = (io.read_bytes  - self._proc_io_baseline.read_bytes)  / (1024 ** 2)
                write_total = (io.write_bytes - self._proc_io_baseline.write_bytes) / (1024 ** 2)

                # Instantaneous rate
                if self._prev_proc_io is not None:
                    elapsed = now - self._prev_proc_time
                    if elapsed > 0:
                        read_mbps  = (io.read_bytes  - self._prev_proc_io.read_bytes)  / elapsed / (1024 ** 2)
                        write_mbps = (io.write_bytes - self._prev_proc_io.write_bytes) / elapsed / (1024 ** 2)
                    else:
                        read_mbps = write_mbps = 0.0
                else:
                    read_mbps = write_mbps = 0.0

                self._prev_proc_io   = io
                self._prev_proc_time = now

                self.proc_read_mbps.append(round(read_mbps, 2))
                self.proc_write_mbps.append(round(write_mbps, 2))
                self.proc_read_total_mb.append(round(read_total, 4))
                self.proc_write_total_mb.append(round(write_total, 4))

            # ── Page faults (process mode) ────────────────────────────────────
            pgfault = pgmajfault = 0
            try:
                mem_info = self._proc.memory_info()
                if hasattr(mem_info, "num_page_faults"):
                    # Windows: combined minor+major total (no split available via psutil)
                    pgfault = mem_info.num_page_faults
                else:
                    # Linux: read minflt (field 9) and majflt (field 11) from /proc/<pid>/stat
                    with open(f"/proc/{self._proc.pid}/stat") as fh:
                        fields = fh.read().split()
                    pgfault    = int(fields[9])    # minflt
                    pgmajfault = int(fields[11])   # majflt
            except Exception:
                pass

            prev_pf = self._proc_pgfault_prev
            if prev_pf is not None:
                self.proc_pgfault_delta.append(max(0, pgfault    - prev_pf[0]))
                self.proc_majflt_delta.append( max(0, pgmajfault - prev_pf[1]))
            else:
                self.proc_pgfault_delta.append(0)
                self.proc_majflt_delta.append(0)
            self._proc_pgfault_prev = (pgfault, pgmajfault)

            # Sample true disk footprint via os.walk if data_dir is provided
            if self._data_dir:
                if os.path.exists(self._data_dir):
                    total_size = 0
                    for dirpath, _, filenames in os.walk(self._data_dir):
                        for f in filenames:
                            try:
                                total_size += os.path.getsize(os.path.join(dirpath, f))
                            except FileNotFoundError:
                                pass
                    size_mb = total_size / (1024 * 1024)
                    
                    # If the database dropped the collection, the size will drop below the initial baseline.
                    # We should reset the baseline to this new low watermark so we track the fresh ingestion.
                    if size_mb < self._proc_disk_baseline_mb:
                        self._proc_disk_baseline_mb = size_mb
                        
                    net_size_mb = size_mb - self._proc_disk_baseline_mb
                    self.proc_disk_usage_mb.append(max(0.0, round(net_size_mb, 4)))
                else:
                    self.proc_disk_usage_mb.append(0.0)

            self._sample_times.append(ts)

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"[SystemMonitor] Process sample failed: {e}")

    # =========================================================================
    # Snapshot (On-Demand Instantaneous Read)
    # =========================================================================

    def snapshot(self) -> Dict[str, Any]:
        """
        Return the delta of cumulative metrics (NetIO, Page Faults) since the *last* 
        time snapshot() was called, and the instantaneous value for point-in-time metrics 
        (like mem_cache_mb).
        
        Reads the most recent values collected by the background polling thread.
        This is a purely passive, non-blocking read.
        """
        raw_state = {}
        
        if self._mode == "docker":
            for name in self._container_names:
                net_prev = self._net_prev.get(name) or {"rx": 0.0, "tx": 0.0}
                pg_prev = self._pgfault_prev.get(name) or {"pgfault": 0, "pgmajfault": 0}
                st = self.docker_stats.get(name, {})
                cache = st.get("mem_cache_mb", [0.0])
                cache_val = cache[-1] if cache else 0.0
                
                raw_state[name] = {
                    "net_rx_mb": net_prev.get("rx", 0.0),
                    "net_tx_mb": net_prev.get("tx", 0.0),
                    "pgfault": pg_prev.get("pgfault", 0),
                    "pgmajfault": pg_prev.get("pgmajfault", 0),
                    "mem_cache_mb": cache_val,
                }
        elif self._mode == "process":
            # For process mode, we fetch from the last io_counters recorded in _poll_loop
            read_mb = 0.0
            write_mb = 0.0
            if getattr(self, "_prev_proc_io", None) is not None:
                read_mb = round(self._prev_proc_io.read_bytes / (1024 ** 2), 4)
                write_mb = round(self._prev_proc_io.write_bytes / (1024 ** 2), 4)
                
            raw_state["process"] = {
                "read_mb": read_mb,
                "write_mb": write_mb,
                "pgfault": 0,  # Psutil pgfault is tracked as deltas in process mode, absolute not strictly retained here
                "pgmajfault": 0,
            }
                
        # Calculate deltas
        result = {}
        for name, current in raw_state.items():
            result[name] = {}
            prev = self._snapshot_prev.get(name)
            for k, v in current.items():
                if k == "mem_cache_mb":
                    # Point in time (not cumulative), return as-is
                    result[name][k] = v
                else:
                    # Cumulative counters, return delta
                    if prev is not None:
                        # For floats, round the delta. For ints, leave as int.
                        delta = max(0, v - prev.get(k, 0))
                        result[name][k] = round(delta, 4) if isinstance(delta, float) else delta
                    else:
                        result[name][k] = 0.0 if isinstance(v, float) else 0

        self._snapshot_prev = raw_state
        return result

    # =========================================================================
    # Timeline property — passed to generate_plots()
    # =========================================================================

    @property
    def timeline(self) -> Dict[str, Any]:
        """
        Live reference to all raw time-series data.

        sample_times_s[i] is the wall-clock offset (seconds since monitor.start())
        at which sample i was recorded.  Intervals are NOT guaranteed to be
        exactly poll_interval_s — actual gap = collection_time + poll_interval_s.

        Docker mode
        -----------
        docker_stats[container]["cpu"][i]            - CPU % at sample i
        docker_stats[container]["mem_mb"][i]         - RAM MB at sample i
        docker_stats[container]["read_mbps"][i]      - read throughput MB/s at sample i
        docker_stats[container]["write_mbps"][i]     - write throughput MB/s at sample i
        docker_stats[container]["read_total_mb"][i]  - cumulative MB read since start()
        docker_stats[container]["write_total_mb"][i] - cumulative MB written since start()

        Note: different containers may have slightly different sample counts if
        individual stat calls fail; they carry their own implicit index.

        Process mode
        ------------
        proc_cpu[i]            - CPU % at sample i
        proc_mem_mb[i]         - RSS MB at sample i
        proc_read_mbps[i]      - read throughput MB/s at sample i
        proc_write_mbps[i]     - write throughput MB/s at sample i
        proc_read_total_mb[i]  - cumulative MB read since set_pid() / start()
        proc_write_total_mb[i] - cumulative MB written since set_pid() / start()
        """
        return {
            "mode":            self._mode,
            "sample_times_s":  self._sample_times,
            "poll_interval_s": self._interval,
            # Docker mode — all per-container series are in docker_stats[name]:
            #   cpu, mem_mb, mem_pct, read_mbps, write_mbps, read_total_mb,
            #   write_total_mb, disk_usage_mb, net_rx_mbps, net_tx_mbps,
            #   net_rx_total_mb, net_tx_total_mb, pgfault_delta, pgmajfault_delta,
            #   mem_cache_mb
            "docker_stats":    self.docker_stats,
            # Process mode
            "proc_cpu":            self.proc_cpu,
            "proc_mem_mb":         self.proc_mem_mb,
            "proc_read_mbps":      self.proc_read_mbps,
            "proc_write_mbps":     self.proc_write_mbps,
            "proc_read_total_mb":  self.proc_read_total_mb,
            "proc_write_total_mb": self.proc_write_total_mb,
            "proc_disk_usage_mb":  self.proc_disk_usage_mb,
            "proc_pgfault_delta":  self.proc_pgfault_delta,   # minor faults (Windows: combined)
            "proc_majflt_delta":   self.proc_majflt_delta,    # major faults (Linux only)
        }

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _get_docker_size_mb(self, container) -> float:
        """
        Calculates the disk footprint of a Docker container.
        If it has bind/volume mounts, sum their source directories on the host.
        Fallback to SizeRw via the Docker API if there are no mounts.
        """
        container.reload()
        mounts = container.attrs.get("Mounts", [])
        
        if mounts:
            total_bytes = 0
            mounts_accessed = False
            for mount in mounts:
                source = mount.get("Source")
                if source and os.path.exists(source):
                    mounts_accessed = True
                    for dirpath, _, filenames in os.walk(source):
                        for f in filenames:
                            try:
                                total_bytes += os.path.getsize(os.path.join(dirpath, f))
                            except FileNotFoundError:
                                pass
            
            if mounts_accessed:
                return total_bytes / (1024 * 1024)
            
            # Windows / Docker Desktop Fallback: Host cannot access WSL volume sources
            # Use docker exec `du -sm` to measure the volume from inside the container
            total_mb = 0.0
            for mount in mounts:
                dest = mount.get("Destination")
                if dest:
                    try:
                        exit_code, output = container.exec_run(f"du -sm {dest}")
                        if exit_code == 0:
                            # Output format: "1234\t/var/lib/milvus"
                            size_mb = float(output.decode("utf-8").strip().split()[0])
                            total_mb += size_mb
                    except Exception:
                        pass
            
            if total_mb > 0:
                return total_mb
            
            # Final fallback if du fails
            info = self._docker_client.api.inspect_container(container.id)
            size_rw = info.get("SizeRw") or 0
            return size_rw / (1024 * 1024)
        else:
            # Fallback to SizeRw
            info = self._docker_client.api.inspect_container(container.id)
            size_rw = info.get("SizeRw") or 0
            return size_rw / (1024 * 1024)
