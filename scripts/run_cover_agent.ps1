# Cover-Agent test generation for critical ML pipeline files
# Usage: .\scripts\run_cover_agent.ps1 [-Target <filename>]
# Requires: pip install cover-agent
# Run from project root: D:\test 2\AI trading assistance

param(
    [string]$Target = "all"
)

$ProjectRoot = "D:\test 2\AI trading assistance"
$VenvPython  = "$ProjectRoot\venv\Scripts\python.exe"

function Run-CoverAgent {
    param([string]$SourceFile, [string]$TestFile, [string]$FunctionFilter)

    $cmd = @(
        "cover-agent",
        "--source-file-path", $SourceFile,
        "--test-file-path",   $TestFile,
        "--code-coverage-report-path", "coverage.xml",
        "--test-command", "pytest $TestFile --cov=src --cov-report=xml -x -q",
        "--test-command-dir", $ProjectRoot,
        "--coverage-type", "cobertura",
        "--max-iterations", "3"
    )
    if ($FunctionFilter) {
        $cmd += "--included-functions", $FunctionFilter
    }
    Write-Host "`n>>> Cover-Agent: $SourceFile" -ForegroundColor Cyan
    & $VenvPython -m $cmd
}

Set-Location $ProjectRoot

switch ($Target) {
    "triple_barrier" {
        Run-CoverAgent `
            "src/analysis/triple_barrier.py" `
            "tests/test_triple_barrier.py" `
            "triple_barrier_labels_vectorized"
    }
    "purged_kfold" {
        Run-CoverAgent `
            "src/utils/purged_kfold.py" `
            "tests/test_purged_kfold.py" `
            "split"
    }
    "train_meta" {
        Run-CoverAgent `
            "src/engine/train_meta_labeler.py" `
            "tests/test_train_meta_labeler.py" `
            "train_meta_labeler"
    }
    "meta_labeler" {
        Run-CoverAgent `
            "src/analysis/meta_labeler.py" `
            "tests/test_meta_labeler.py" `
            "filter,batch_filter"
    }
    default {
        # Run all critical ML files
        Run-CoverAgent "src/analysis/triple_barrier.py"     "tests/test_triple_barrier.py"    "triple_barrier_labels_vectorized"
        Run-CoverAgent "src/utils/purged_kfold.py"          "tests/test_purged_kfold.py"       "split"
        Run-CoverAgent "src/engine/train_meta_labeler.py"   "tests/test_train_meta_labeler.py" "train_meta_labeler"
        Run-CoverAgent "src/analysis/meta_labeler.py"       "tests/test_meta_labeler.py"       "filter,batch_filter"
    }
}

Write-Host "`nCover-Agent run complete. Review generated tests before committing." -ForegroundColor Green
