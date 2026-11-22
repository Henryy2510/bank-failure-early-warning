# Native Windows PowerShell runner for the BFT65B FDIC Bank Failure Early Warning Pipeline (no WSL required).
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at .venv. Please run: py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

$env:PYTHONPATH = $ScriptDir
$env:POLARS_MAX_THREADS = "4"
$env:PYTHONUTF8 = "1"
$env:PYSPARK_PYTHON = $Python
$env:PYSPARK_DRIVER_PYTHON = $Python
if (-not $env:MPLCONFIGDIR) {
    $env:MPLCONFIGDIR = Join-Path $env:TEMP "matplotlib"
}

$OutTables = Join-Path $ScriptDir "outputs\tables"
New-Item -ItemType Directory -Force -Path $OutTables | Out-Null
$TimingFile = Join-Path $OutTables "run_all_timing.txt"
$StartIso = Get-Date -Format "o"
@(
    "BFT65B Pipeline Execution Timing Log ($StartIso)"
    "--------------------------------------------------------"
) | Set-Content -Path $TimingFile -Encoding UTF8

$TotalStart = Get-Date

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$ArgsList
    )
    Write-Host ""
    Write-Host $Label
    $t0 = Get-Date
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Label"
    }
    $dt = [int][Math]::Round(((Get-Date) - $t0).TotalSeconds)
    $line = "${Label}: ${dt}s"
    Write-Host $line
    Add-Content -Path $TimingFile -Value $line -Encoding UTF8
}

Write-Host "========================================================================"
Write-Host "BFT65B: FDIC Bank Failure Early Warning Big-Data Pipeline (Windows)"
Write-Host "National Economics University - Big Data Analytics in Finance"
Write-Host "========================================================================"
$PyVer = & $Python --version
Write-Host "Using Python: $PyVer"
try {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGb = [Math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $totGb = [Math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    Write-Host "Available Memory: ${freeGb} GB free of ${totGb} GB"
} catch {
    # Ignore if WMI/CIM is unavailable
}
if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    Write-Warning "Java (OpenJDK 17 or 21) was not found on PATH. Step 6 (PySpark) requires Java + JAVA_HOME + HADOOP_HOME (winutils.exe)."
}
Write-Host "Start Time: $(Get-Date)"

Invoke-Step "[Step 0/9] Part 1: Verifying Raw FDIC Data Completeness" @("scripts\00_download.py", "--verify-only")
Invoke-Step "[Step 1/9] Part 1: Storage Layer Ingestion & Format Benchmarking" @("scripts\01_storage.py")
Invoke-Step "[Step 2/9] Part 2: Automated Data Quality Audit" @("scripts\02_quality.py")
Invoke-Step "[Step 3/9] Part 2 (cont): Clean Silver Panel Build & Exit Classification" @("scripts\03_panel.py")
Invoke-Step "[Step 4/9] Part 4: Gold Leak-Free Feature Pipeline & Target Generation" @("scripts\04_features.py")
Invoke-Step "[Step 5/9] Part 3: Scalable Processing Benchmarks (Polars vs Pandas)" @("scripts\05_benchmark.py")
Invoke-Step "[Step 6/9] Part 3 & 5: PySpark Pipeline, Execution Plan & MLlib Demo" @("scripts\06_spark.py")
Invoke-Step "[Step 7/9] Part 5: Model Validation, Cost-Sensitive Threshold & Test" @("scripts\07_model.py")
Invoke-Step "[Step 8/9] Executing Notebooks via nbclient & Building Report (MD + DOCX + PDF)" @("scripts\08_build_notebooks_and_report.py")
Invoke-Step "[Step 9/9] Verifying Report & README Number Consistency" @("scripts\09_check_report_numbers.py")
Invoke-Step "[Tests] Running Automated Pytest Suite" @("-m", "pytest", "-v", "tests")

$TotalElapsed = [int][Math]::Round(((Get-Date) - $TotalStart).TotalSeconds)
Add-Content -Path $TimingFile -Value "--------------------------------------------------------" -Encoding UTF8
$SummaryLine = "Total Wall-Clock Time: ${TotalElapsed}s"
Write-Host $SummaryLine
Add-Content -Path $TimingFile -Value $SummaryLine -Encoding UTF8
Write-Host ""
Write-Host "Pipeline execution finished successfully at $(Get-Date)!"
