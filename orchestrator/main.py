"""
Orchestrator - Central Controller

The main entry point that coordinates all system activities:
1. Read configs
2. Initialize DB connection
3. Initialize workload environment
4. Schedule and run workloads
5. Manage ingestion and query execution
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import TypeAdapter
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Orchestrator:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.adapter = None
        self.workload = None

    def run(self) -> Dict[str, Any]:
        self._load_config()
        self._init_db()
        self._init_workload()
        query_results,results, monitor = self._execute_workload()
        return query_results,results, monitor

    # =========================================================================
    # CONFIG
    # =========================================================================

    def _load_config(self) -> None:
        from parser.parser import load_config

        print(f"[Orchestrator] Loading config: {self.config_path}")
        self.config = load_config(self.config_path)

        try:
            from parser.schema import GlobalConfig, WorkloadConfig, IndexConfig
        except Exception:
            raise

        try:
            self.global_config_model = GlobalConfig(**self.config["global"])

            workload_data = self.config.get("workload", {})
            adapter = TypeAdapter(WorkloadConfig)
            self.workload_config_model = adapter.validate_python(workload_data)

        except Exception as e:
            raise RuntimeError(f"Configuration validation error: {e}") from e

        print(f"[Orchestrator] Dataset:     {self.global_config_model.dataset}")
        print(f"[Orchestrator] Concurrency: {self.global_config_model.concurrency}")
        print(f"[Orchestrator] Ingest Batch size:  {self.global_config_model.ingest_batch_size}")

    # =========================================================================
    # DATABASE
    # =========================================================================

    def _init_db(self) -> None:
        from database_adapter import get_adapter

        db_config = self.global_config_model.database
        print(f"[Orchestrator] Connecting to {db_config.adapter}://{db_config.host}:{db_config.port}")

        extra_kwargs = {}
        if db_config.adapter in ("pinecone", "pinecone_serverless") and hasattr(db_config, "pinecone_config") and db_config.pinecone_config:
            extra_kwargs["pinecone_config"] = db_config.pinecone_config.model_dump()

        self.adapter = get_adapter(
            adapter_type=db_config.adapter,
            host=db_config.host,
            port=db_config.port,
            collection=db_config.collection,
            **extra_kwargs,
        )
        self.adapter.connect()
        print("[Orchestrator] Database connected")

    # =========================================================================
    # WORKLOAD
    # =========================================================================

    def _init_workload(self) -> None:
        workload_type = self.workload_config_model.type
        print(f"[Orchestrator] Initializing workload: {workload_type}")

        self.workload_module = self._get_workload_module(workload_type)

        seed = self.global_config_model.seed
        self._set_seed(seed)
        print(f"[Orchestrator] Seed set: {seed}")

    def _get_workload_module(self, workload_type: str):
        from parser.schema import normalize_to_underscore
        workload_type = normalize_to_underscore(workload_type)

        import importlib
        module_path = f"workloads.{workload_type}.main"

        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            raise RuntimeError(f"Workload not found: {workload_type}") from e

    def _set_seed(self, seed: int) -> None:
        import random
        random.seed(seed)

        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def _execute_workload(self) -> Dict[str, Any]:
        """Run the workload, optionally monitored."""
        monitor_enabled = self.config.get("global", {}).get("monitor_system", False)
        container_names = self.config.get("global", {}).get("containers", [])
        metrics_cfg     = self.config.get("global", {}).get("system_metrics", {})
        monitor         = None

        if monitor_enabled:
            from orchestrator.operations.system_monitor import SystemMonitor
            monitor = SystemMonitor(container_names=container_names)
            monitor.start()
            print("[Orchestrator] System monitor started.")

        print("[Orchestrator] Starting workload execution...")
        query_results,results = self.workload_module.run_workload(
            config=self.config,
            adapter=self.adapter
        )

        if monitor is not None:
            sys_stats = monitor.stop()
            filtered_sys: Dict[str, Any] = {}

            for key in metrics_cfg.get("system", []):
                if key in sys_stats:
                    filtered_sys[key] = sys_stats[key]

            for name in container_names:
                safe = name.replace("-", "_")
                for metric in metrics_cfg.get("docker", []):
                    key = f"docker_{safe}_{metric}"
                    if key in sys_stats:
                        filtered_sys[key] = sys_stats[key]

           
            results["system_metrics"] = filtered_sys

        print("[Orchestrator] Workload completed")
        return query_results,results, monitor

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def _cleanup(self) -> None:
        print("[Orchestrator] Cleaning up...")
        if self.adapter:
            self.adapter.disconnect()
            print("[Orchestrator] Database disconnected")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(config_path: Optional[str] = None) -> Dict[str, Any]:
    import json
    from datetime import datetime
    import numpy as np

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: python -m orchestrator.main <config.json>")
            sys.exit(1)
        config_path = sys.argv[1]

    orchestrator = Orchestrator(config_path)

    try:
        query_results, results, monitor = orchestrator.run()

        # =========================================================================
        # CREATE RUN OUTPUT DIRECTORY
        # =========================================================================

        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)

        config_name = Path(config_path).stem
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_id = f"{config_name}_{timestamp}"

        output_dir = results_dir / run_id
        output_dir.mkdir(exist_ok=True)

        # =========================================================================
        # SAVE GLOBAL RESULTS
        # =========================================================================

        results_file = output_dir / "results.json"

        global_output = {
            "config": config_name,
            "timestamp": timestamp,

            "results": {
                k: v for k, v in results.items()
                if not k.startswith("_")
            },

            "passes": [
                {
                    "path": f"pass_{p['pass']}/results.json",
                    **p
                }
                for p in query_results["passes"]
            ]
        }

        with open(results_file, "w") as f:
            json.dump(global_output, f, indent=2, cls=_NumpyEncoder)

        print(f"\n[Orchestrator] Global results saved to: {results_file}")

        # =========================================================================
        # GENERATE GLOBAL PLOTS
        # =========================================================================

        from orchestrator.operations.plotting import generate_plots

        generate_plots(
            stats=results,
            latencies=[],
            output_dir=output_dir,
            monitor_timeline=monitor.timeline if monitor else None,
        )

        # =========================================================================
        # SAVE PER-PASS RESULTS + PLOTS
        # =========================================================================

        for pass_result in query_results.get("passes", []):

            pass_num = pass_result["pass"]

            pass_dir = output_dir / f"pass_{pass_num}"
            pass_dir.mkdir(exist_ok=True)

            pass_results_file = pass_dir / "results.json"

            pass_output = {
                "config": config_name,
                "timestamp": timestamp,

                "results": {
                    k: v for k, v in pass_result.items()
                    if not k.startswith("_")
                }
            }

            with open(pass_results_file, "w") as f:
                json.dump(pass_output, f, indent=2, cls=_NumpyEncoder)

            print(
                f"[Orchestrator] Pass {pass_num} results saved to: "
                f"{pass_results_file}"
            )

            generate_plots(
                stats=pass_result,
                latencies=pass_result.get("_raw_latencies", []),
                output_dir=pass_dir,
                monitor_timeline= None,
            )
   
    finally:
        orchestrator._cleanup()

    return results


if __name__ == "__main__":
    main()
