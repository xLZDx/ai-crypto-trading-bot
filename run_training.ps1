Set-Location 'd:\test 2\AI trading assistance'
$env:PYTHONPATH = 'd:\test 2\AI trading assistance'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:OPENBLAS_MAIN_FREE = '1'
.\venv\Scripts\Activate.ps1
python src\engine\train_all_models.py 2>&1 | Tee-Object -FilePath 'logs\training_run.log'
Write-Host '=== TRAINING COMPLETE ===' -ForegroundColor Green
