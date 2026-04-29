# ── setup_env.ps1 ─────────────────────────────────────────────────────────────
# Shared environment bootstrap — dot-source this from every launch script:
#   . (Join-Path $root 'setup_env.ps1')
#
# Sets:
#   1. All temp/cache dirs → D:\... (keeps C drive free)
#   2. CPU multithreading  → all logical cores (20 on this machine)
#   3. GPU optimization    → CUDA cache on D:, TF32 + cuDNN benchmark flags
# ─────────────────────────────────────────────────────────────────────────────

if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }

$cacheDir = Join-Path $root 'data\cache'

# Force Python to use UTF-8 instead of cp1252 for console output/logging
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# ── 1. Temp & cache → D drive ─────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'temp')         | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'pip')          | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'torch')        | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'huggingface')  | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'cuda')         | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'numba')        | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'matplotlib')   | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'darts')        | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root   'darts_logs')     | Out-Null

# Windows TEMP / TMP (used by joblib, sklearn parallelism, pip)
$env:TMP  = Join-Path $cacheDir 'temp'
$env:TEMP = Join-Path $cacheDir 'temp'

# Python / ML tool caches
$env:PIP_CACHE_DIR       = Join-Path $cacheDir 'pip'
$env:TORCH_HOME          = Join-Path $cacheDir 'torch'
$env:HF_HOME             = Join-Path $cacheDir 'huggingface'
$env:TRANSFORMERS_CACHE  = Join-Path $cacheDir 'huggingface'
$env:JOBLIB_TEMP_FOLDER  = Join-Path $cacheDir 'temp'

# CUDA kernel cache (prevents C:\Users\...\AppData\Local\NVIDIA writes)
$env:CUDA_CACHE_PATH     = Join-Path $cacheDir 'cuda'
$env:CUDA_CACHE_DISABLE  = '0'          # keep caching, just on D:

# Numba / llvmlite JIT cache
$env:NUMBA_CACHE_DIR     = Join-Path $cacheDir 'numba'

# Matplotlib (font cache)
$env:MPLCONFIGDIR        = Join-Path $cacheDir 'matplotlib'

# Darts checkpoint dir (overridden in code too, but belt + suspenders)
$env:DARTS_LOG_DIR       = Join-Path $root 'darts_logs'

# ── 2. CPU multithreading ──────────────────────────────────────────────────────
$cpuCount = (Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum
if (-not $cpuCount -or $cpuCount -lt 1) { $cpuCount = 20 }

$env:OMP_NUM_THREADS     = "$cpuCount"   # OpenMP (sklearn, numpy, scipy internals)
$env:MKL_NUM_THREADS     = "$cpuCount"   # Intel MKL (numpy on Windows often uses MKL)
$env:OPENBLAS_NUM_THREADS = "$cpuCount"  # OpenBLAS fallback
$env:NUMEXPR_NUM_THREADS  = "$cpuCount"  # numexpr (pandas speed-ups)
$env:VECLIB_MAXIMUM_THREADS = "$cpuCount" # macOS vecLib (harmless on Windows)

# Tell sklearn/joblib the loky worker count
$env:LOKY_MAX_CPU_COUNT  = "$cpuCount"

Write-Host "  [ENV] $cpuCount logical CPUs → OMP/MKL/LOKY all set" -ForegroundColor Cyan

# ── 3. GPU / CUDA optimizations ───────────────────────────────────────────────
# These env vars tell PyTorch / cuDNN to use full RTX 3080 Ti capability:
#   TF32 on Ampere: ~3x faster matmul with negligible accuracy loss
#   cuDNN benchmark: picks fastest conv algorithm for the given input size
$env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = '1'
$env:NVIDIA_TF32_OVERRIDE             = '1'   # driver-level TF32 override
$env:CUDNN_BENCHMARK                  = '1'

Write-Host "  [ENV] CUDA cache → $($env:CUDA_CACHE_PATH)" -ForegroundColor Cyan
Write-Host "  [ENV] TF32 + cuDNN benchmark enabled for RTX 3080 Ti" -ForegroundColor Cyan
Write-Host "  [ENV] All caches → $cacheDir" -ForegroundColor Cyan
