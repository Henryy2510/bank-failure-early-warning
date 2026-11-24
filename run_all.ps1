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

function Test-HadoopBins {
    param([Parameter(Mandatory = $true)][string]$HomeDir)
    if ([string]::IsNullOrWhiteSpace($HomeDir)) { return $false }
    $winutils = Join-Path $HomeDir "bin\winutils.exe"
    $hadoopDll = Join-Path $HomeDir "bin\hadoop.dll"
    return (Test-Path $winutils) -and (Test-Path $hadoopDll) -and
           ((Get-Item $winutils).Length -gt 10000) -and ((Get-Item $hadoopDll).Length -gt 10000)
}

function Ensure-WindowsHadoop {
    $candidates = @()
    if ($env:HADOOP_HOME) { $candidates += $env:HADOOP_HOME }
    $candidates += (Join-Path $ScriptDir ".hadoop")
    $candidates += "C:\hadoop"

    $resolved = $null
    foreach ($cand in $candidates) {
        if (Test-HadoopBins $cand) {
            $resolved = $cand
            break
        }
    }

    if (-not $resolved) {
        $resolved = Join-Path $ScriptDir ".hadoop"
        $binDir = Join-Path $resolved "bin"
        New-Item -ItemType Directory -Force -Path $binDir | Out-Null
        Write-Host "Downloading Windows Hadoop binaries (winutils.exe, hadoop.dll) to $binDir ..."
        $mirrors = @(
            "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin",
            "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin"
        )
        $downloaded = $false
        foreach ($baseUrl in $mirrors) {
            try {
                foreach ($name in @("winutils.exe", "hadoop.dll")) {
                    $dest = Join-Path $binDir $name
                    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 10000)) { continue }
                    $tmp = "$dest.tmp"
                    Invoke-WebRequest -Uri "$baseUrl/$name" -OutFile $tmp -UseBasicParsing -Headers @{ "User-Agent" = "BFT65B-Pipeline" } -TimeoutSec 30
                    $bytes = [System.IO.File]::ReadAllBytes($tmp)
                    if ($bytes.Length -le 10000 -or $bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) {
                        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
                        throw "Invalid PE binary from $baseUrl/$name"
                    }
                    Move-Item -Path $tmp -Destination $dest -Force
                }
                $downloaded = $true
                break
            } catch {
                Write-Warning "Mirror failed ($baseUrl): $_"
            }
        }
        if (-not $downloaded) {
            throw "Failed to download winutils.exe / hadoop.dll. Place them in $binDir or set HADOOP_HOME manually."
        }
    }

    $env:HADOOP_HOME = $resolved
    $binPath = Join-Path $resolved "bin"
    $pathParts = ($env:PATH -split ";") | Where-Object { $_ }
    if (-not ($pathParts | Where-Object { $_.Trim().ToLowerInvariant() -eq $binPath.ToLowerInvariant() })) {
        $env:PATH = "$binPath;$env:PATH"
    }
    Write-Host "Using HADOOP_HOME: $env:HADOOP_HOME"
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
    Write-Warning "Java (OpenJDK 17 or 21) was not found on PATH. Step 6 (PySpark) requires OpenJDK 17 or 21 with JAVA_HOME."
}
Ensure-WindowsHadoop
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
