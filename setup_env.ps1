# ── setup_env.ps1 ─────────────────────────────────────────────────────────────
# Shared environment bootstrap - dot-source this from every launch script:
#   . (Join-Path $root 'setup_env.ps1')
#
# Sets:
#   1. All temp/cache dirs -> D:\... (keeps C drive free)
#   2. CPU multithreading  -> bounded to $env:CPU_CORES (default 10)
#   3. GPU policy          -> 'single_100' (default) or 'dual_80' for cluster DDP
# ─────────────────────────────────────────────────────────────────────────────

if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }

$cacheDir = Join-Path $root 'data\cache'

# Force Python to use UTF-8 instead of cp1252 for console output/logging
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# ── 1. Temp & cache -> D drive ─────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'temp')         | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'pip')          | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'torch')        | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'huggingface')  | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'cuda')         | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'numba')        | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'matplotlib')   | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'darts')        | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root   'darts_logs')     | Out-Null

$env:TMP  = Join-Path $cacheDir 'temp'
$env:TEMP = Join-Path $cacheDir 'temp'

$env:PIP_CACHE_DIR       = Join-Path $cacheDir 'pip'
$env:TORCH_HOME          = Join-Path $cacheDir 'torch'
$env:HF_HOME             = Join-Path $cacheDir 'huggingface'
$env:TRANSFORMERS_CACHE  = Join-Path $cacheDir 'huggingface'
$env:JOBLIB_TEMP_FOLDER  = Join-Path $cacheDir 'temp'
$env:CUDA_CACHE_PATH     = Join-Path $cacheDir 'cuda'
$env:CUDA_CACHE_DISABLE  = '0'
$env:NUMBA_CACHE_DIR     = Join-Path $cacheDir 'numba'
$env:MPLCONFIGDIR        = Join-Path $cacheDir 'matplotlib'
$env:DARTS_LOG_DIR       = Join-Path $root 'darts_logs'

# ── 2. CPU policy - bounded thread count (user request: 10 cores w/ MT) ──────
# Override with `$env:CPU_CORES = '8'` before calling restart_all.ps1 if needed.
if (-not $env:CPU_CORES) { $env:CPU_CORES = '10' }
$cpuCount = [int]$env:CPU_CORES
$totalLogical = (Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum
if ($cpuCount -gt $totalLogical) { $cpuCount = $totalLogical }
if ($cpuCount -lt 1) { $cpuCount = 1 }

$env:OMP_NUM_THREADS         = "$cpuCount"
$env:MKL_NUM_THREADS         = "$cpuCount"
$env:OPENBLAS_NUM_THREADS    = "$cpuCount"
$env:NUMEXPR_NUM_THREADS     = "$cpuCount"
$env:VECLIB_MAXIMUM_THREADS  = "$cpuCount"
$env:LOKY_MAX_CPU_COUNT      = "$cpuCount"
# PyTorch CPU intra-op + inter-op parallelism
$env:TORCH_NUM_THREADS       = "$cpuCount"
$env:TORCH_NUM_INTEROP_THREADS = "2"

Write-Host "  [ENV] CPU = $cpuCount of $totalLogical logical cores (multithreaded)" -ForegroundColor Cyan

# ── 3. GPU policy ─────────────────────────────────────────────────────────────
#   single_100 (default) - bind to GPU 0, allow up to 100% memory.
#   dual_80              - both GPUs visible, cap each at 80% memory (DDP).
if (-not $env:GPU_POLICY) { $env:GPU_POLICY = 'single_100' }

if ($env:GPU_POLICY -eq 'dual_80') {
    $env:CUDA_VISIBLE_DEVICES = '0,1'
    $env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:512,garbage_collection_threshold:0.80'
    Write-Host "  [ENV] GPU policy = dual_80 (both GPUs, 80% mem cap)" -ForegroundColor Cyan
} else {
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:1024,garbage_collection_threshold:0.95'
    Write-Host "  [ENV] GPU policy = single_100 (GPU 0, full memory)" -ForegroundColor Cyan
}

# Ampere TF32 + cuDNN benchmark
$env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = '1'
$env:NVIDIA_TF32_OVERRIDE             = '1'
$env:CUDNN_BENCHMARK                  = '1'

Write-Host "  [ENV] CUDA cache -> $($env:CUDA_CACHE_PATH)" -ForegroundColor Cyan
Write-Host "  [ENV] TF32 + cuDNN benchmark enabled" -ForegroundColor Cyan
Write-Host "  [ENV] All caches -> $cacheDir" -ForegroundColor Cyan
