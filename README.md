# Range Deduplication Workload

A testing and orchestration framework for evaluating range deduplication in vector databases (e.g., Qdrant, Milvus).

## Overview
This repository contains a framework for benchmarking and testing data deduplication in vector databases via range search and Bloom filter strategies. It orchestrates vector workloads, queries, system monitoring, and latency plotting.

## Project Structure
- `data_loader/`: Utilities for loading datasets (e.g., HDF5).
- `database_adapter/`: Adapters for connecting to vector databases (`qdrant_adapter.py`, `milvus_adapter.py`).
- `deduplication_workload/`: Defines the core deduplication logic and tests.
- `orchestrator/`: Central controller to coordinate database initialization, run the workload, capture metrics, and manage the system.
- `parser/`: Parses and validates configuration schemas using Pydantic.
- `reference_config/`: Example and reference JSON configurations for various workloads.
- `results/`: Directory where benchmark outputs, system metrics, and generated plots are stored.
- `test/`: Contains unit and integration tests for the application logic and adapters.
- `run.py`: Main entry point for the application.

## Prerequisites
- Python 3.8+
- Docker (optional, for system monitoring capabilities)

## Installation
Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage
Run the application by providing a configuration JSON file:

```bash
python run.py <path/to/config.json>
```

Example:
```bash
python run.py reference_config/deduplication.json
```
