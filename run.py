"""
Generator - Main Entry Point

"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.main import main as run_orchestrator


def main():
    """Run workload from config file."""
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.json>")
        print("\nExample:")
        print("  python run.py configs/glove1M_hnsw.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Resolve relative paths
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_path
    
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    print(f"Running workload from: {config_file}")
    results = run_orchestrator(str(config_file))
    
    return results


if __name__ == "__main__":
    main()