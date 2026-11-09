#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"
export POLARS_MAX_THREADS="4"
export MPLCONFIGDIR="/tmp/matplotlib"

PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "Virtual environment not found at .venv. Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "========================================================================"
echo "BFT65B: FDIC Bank Failure Early Warning Big-Data Pipeline"
echo "National Economics University - Big Data Analytics in Finance"
echo "========================================================================"
echo "Using Python: $($PYTHON --version)"
echo "Start Time: $(date)"

echo ""
echo "[Step 0/8] Part 1: Verifying Raw FDIC Data Completeness..."
$PYTHON scripts/00_download.py --verify-only

echo ""
echo "[Step 1/8] Part 1: Storage Layer Ingestion & Format Benchmarking..."
$PYTHON scripts/01_storage.py

echo ""
echo "[Step 2/8] Part 2: Automated Data Quality Audit..."
$PYTHON scripts/02_quality.py

echo ""
echo "[Step 3/8] Part 2 (cont): Clean Silver Panel Build & Exit Classification..."
$PYTHON scripts/03_panel.py

echo ""
echo "[Step 4/8] Part 4: Gold Leak-Free Feature Pipeline & Target Generation..."
$PYTHON scripts/04_features.py

echo ""
echo "[Step 5/8] Part 3: Scalable Processing Benchmarks (Polars vs Pandas)..."
$PYTHON scripts/05_benchmark.py

echo ""
echo "[Step 6/8] Part 3 & 5: PySpark Pipeline, Execution Plan & MLlib Demo..."
$PYTHON scripts/06_spark.py

echo ""
echo "[Step 7/8] Part 5: Model Validation, Cost-Sensitive Threshold & Test..."
$PYTHON scripts/07_model.py

echo ""
echo "[Step 8/8] Generating Executed Notebooks & Final Report (MD + DOCX)..."
$PYTHON scripts/08_build_notebooks_and_report.py

echo ""
echo "=== Running Integration & Unit Tests ==="
$PYTHON -m pytest -v tests/

echo ""
echo "Pipeline execution finished successfully at $(date)!"
