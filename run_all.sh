#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"
export POLARS_MAX_THREADS="4"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "Virtual environment not found at .venv. Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

mkdir -p outputs/tables
TIMING_FILE="outputs/tables/run_all_timing.txt"
echo "BFT65B Pipeline Execution Timing Log ($(date -Iseconds))" > "$TIMING_FILE"
echo "--------------------------------------------------------" >> "$TIMING_FILE"

T_TOTAL_START=$(date +%s)

run_step() {
    local label="$1"
    shift
    echo ""
    echo "$label"
    local t0=$(date +%s)
    "$@"
    local t1=$(date +%s)
    local dt=$((t1 - t0))
    echo "${label}: ${dt}s" | tee -a "$TIMING_FILE"
}

echo "========================================================================"
echo "BFT65B: FDIC Bank Failure Early Warning Big-Data Pipeline"
echo "National Economics University - Big Data Analytics in Finance"
echo "========================================================================"
echo "Using Python: $("$PYTHON" --version)"
if command -v free >/dev/null 2>&1; then
    echo "Available Memory: $(free -h | awk '/^Mem:/ {print $7}') free of $(free -h | awk '/^Mem:/ {print $2}')"
fi
echo "Start Time: $(date)"

run_step "[Step 0/9] Part 1: Verifying Raw FDIC Data Completeness" "$PYTHON" scripts/00_download.py --verify-only
run_step "[Step 1/9] Part 1: Storage Layer Ingestion & Format Benchmarking" "$PYTHON" scripts/01_storage.py
run_step "[Step 2/9] Part 2: Automated Data Quality Audit" "$PYTHON" scripts/02_quality.py
run_step "[Step 3/9] Part 2 (cont): Clean Silver Panel Build & Exit Classification" "$PYTHON" scripts/03_panel.py
run_step "[Step 4/9] Part 4: Gold Leak-Free Feature Pipeline & Target Generation" "$PYTHON" scripts/04_features.py
run_step "[Step 5/9] Part 3: Scalable Processing Benchmarks (Polars vs Pandas)" "$PYTHON" scripts/05_benchmark.py
run_step "[Step 6/9] Part 3 & 5: PySpark Pipeline, Execution Plan & MLlib Demo" "$PYTHON" scripts/06_spark.py
run_step "[Step 7/9] Part 5: Model Validation, Cost-Sensitive Threshold & Test" "$PYTHON" scripts/07_model.py
run_step "[Step 8/9] Executing Notebooks via nbclient & Building Report (MD + DOCX + PDF)" "$PYTHON" scripts/08_build_notebooks_and_report.py
run_step "[Step 9/9] Verifying Report & README Number Consistency" "$PYTHON" scripts/09_check_report_numbers.py
run_step "[Tests] Running Automated Pytest Suite" "$PYTHON" -m pytest -v tests/

T_TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((T_TOTAL_END - T_TOTAL_START))
echo "--------------------------------------------------------" >> "$TIMING_FILE"
echo "Total Wall-Clock Time: ${TOTAL_ELAPSED}s" | tee -a "$TIMING_FILE"
echo ""
echo "Pipeline execution finished successfully at $(date)!"
