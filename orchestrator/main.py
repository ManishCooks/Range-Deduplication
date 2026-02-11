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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Orchestrator:
    def __init__(self, config_path: str):
        """
        Initialize orchestrator with config file.
        
        Args:
            config_path: Path to JSON config file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.adapter = None
        self.workload = None
        
    def run(self) -> Dict[str, Any]:
        """
        Execute full workflow.
        
        Returns:
            Dict of metrics/results
        """
        # 1. Load config
        self._load_config()
        
        # 2. Initialize database connection
        self._init_db()
        
        # 3. Initialize workload environment
        self._init_workload()
        
        # 4. Run workload (ingestion + queries)
        results = self._execute_workload()
        
        return results
    
    # =========================================================================
    # CONFIG
    # =========================================================================
    
    def _load_config(self) -> None:
        """Load and parse configuration."""
        from parser.parser import load_config
        
        print(f"[Orchestrator] Loading config: {self.config_path}")
        # Load raw config (dict)
        self.config = load_config(self.config_path)

        # Validate and coerce using pydantic models
        try:
            from parser.schema import GlobalConfig, WorkloadConfig, IndexConfig
        except Exception:
            # Defensive import error (shouldn't happen in normal runs)
            raise

        try:
            self.global_config_model = GlobalConfig(**self.config["global"])
            
            workload_data = self.config.get("workload", {})
            adapter = TypeAdapter(WorkloadConfig)
            self.workload_config_model = adapter.validate_python(workload_data)
            
        except Exception as e:
            raise RuntimeError(f"Configuration validation error: {e}") from e

        # Backwards-compatible prints using validated models
        print(f"[Orchestrator] Dataset: {self.global_config_model.dataset}")
        print(f"[Orchestrator] Concurrency: {self.global_config_model.concurrency}")
        print(f"[Orchestrator] Batch size: {self.global_config_model.batch_size}")
    
    # =========================================================================
    # DATABASE
    # =========================================================================
    
    def _init_db(self) -> None:
        """Initialize database connection."""
        from database_adapter import get_adapter
        
        # Use validated DatabaseConfig from GlobalConfig model
        db_config = self.global_config_model.database
        print(f"[Orchestrator] Connecting to {db_config.adapter}://{db_config.host}:{db_config.port}")

        self.adapter = get_adapter(
            adapter_type=db_config.adapter,
            host=db_config.host,
            port=db_config.port,
            collection=db_config.collection
        )
        self.adapter.connect()
        print("[Orchestrator] Database connected")
    
    # =========================================================================
    # WORKLOAD
    # =========================================================================
    
    def _init_workload(self) -> None:
        """Initialize workload environment."""
        workload_type = self.workload_config_model.type
        print(f"[Orchestrator] Initializing workload: {workload_type}")
        
        # Import workload module based on type
        self.workload_module = self._get_workload_module(workload_type)
        
        # Set random seed for reproducibility
        seed = self.global_config_model.seed
        self._set_seed(seed)
        print(f"[Orchestrator] Seed set: {seed}")
    
    def _get_workload_module(self, workload_type: str):
        """Dynamically import workload module."""
        # Normalize workload type
        from parser.schema import normalize_to_underscore
        workload_type = normalize_to_underscore(workload_type)
        
        # Import the workload module
        import importlib
        module_path = f"workloads.{workload_type}.main"
        
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            raise RuntimeError(f"Workload not found: {workload_type}") from e
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    def _execute_workload(self) -> Dict[str, Any]:
        """Run the workload."""
        print("[Orchestrator] Starting workload execution...")
        
        # Pass full config to workload
        # Workload handles: load data, normalize, ingest, query, collect metrics
        results = self.workload_module.run_workload(
            config=self.config,
            adapter=self.adapter
        )
        
        print("[Orchestrator] Workload completed")
        return results
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        print("[Orchestrator] Cleaning up...")
        
        if self.adapter:
            self.adapter.disconnect()
            print("[Orchestrator] Database disconnected")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point.
    
    Args:
        config_path: Path to config file. If None, uses command line arg.
        
    Returns:
        Dict of metrics/results
    """
    import json
    from datetime import datetime
    
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: python -m orchestrator.main <config.json>")
            sys.exit(1)
        config_path = sys.argv[1]
    
    orchestrator = Orchestrator(config_path)
    
    try:
        results = orchestrator.run()
        
        # Save results to JSON file
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        
        config_name = Path(config_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{config_name}_{timestamp}.json"
        
        # Include config metadata in output
        output = {
            "config": config_name,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[Orchestrator] Results saved to: {results_file}")
        
    finally:
        orchestrator._cleanup()
    
    return results


if __name__ == "__main__":
    main()
