$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = "python"

# Set this to $true for a fast smoke run before the full experiment.
$Quick = $false
$QuickArgs = @()
if ($Quick) {
    $QuickArgs = @("--train-limit", "2048", "--valid-limit", "512", "--test-limit", "512", "--epochs", "1", "--eval-iters", "20", "--log-iters", "20")
}

function Run-Step {
    param(
        [string]$Title,
        [string[]]$ArgsList
    )
    Write-Host ""
    Write-Host "========== $Title =========="
    & $Python @ArgsList
}

Run-Step "Train MLP baseline" @(
    "run_experiment.py",
    "--model", "MLP",
    "--experiment", "baseline",
    "--run-name", "mlp_baseline",
    "--epochs", "5",
    "--batch-size", "64",
    "--lr", "0.06",
    "--log-iters", "200",
    "--eval-iters", "200"
) + $QuickArgs

Run-Step "Train CNN baseline" @(
    "run_experiment.py",
    "--model", "CNN",
    "--experiment", "baseline",
    "--run-name", "cnn_baseline",
    "--epochs", "5",
    "--batch-size", "64",
    "--lr", "0.05",
    "--log-iters", "200",
    "--eval-iters", "200"
) + $QuickArgs

Run-Step "Part C Direction 2 - MLP with dropout" @(
    "run_experiment.py",
    "--model", "MLP",
    "--experiment", "dropout",
    "--run-name", "mlp_dropout",
    "--epochs", "5",
    "--batch-size", "64",
    "--lr", "0.06",
    "--dropout-rate", "0.5",
    "--log-iters", "200",
    "--eval-iters", "200"
) + $QuickArgs

Run-Step "Part C Direction 2 - CNN with dropout" @(
    "run_experiment.py",
    "--model", "CNN",
    "--experiment", "dropout",
    "--run-name", "cnn_dropout",
    "--epochs", "5",
    "--batch-size", "64",
    "--lr", "0.05",
    "--dropout-rate", "0.3",
    "--log-iters", "200",
    "--eval-iters", "200"
) + $QuickArgs

Run-Step "Part C Direction 5 - MLP error analysis and visualization" @(
    "error_analysis.py",
    "--model", "MLP",
    "--model-path", ".\best_models\mlp_baseline\best_model.pickle",
    "--output-dir", ".\figs\mlp_baseline"
)

Run-Step "Part C Direction 5 - CNN error analysis and visualization" @(
    "error_analysis.py",
    "--model", "CNN",
    "--model-path", ".\best_models\cnn_baseline\best_model.pickle",
    "--output-dir", ".\figs\cnn_baseline"
)

Run-Step "Compare all experiment results" @(
    "compare_results.py",
    "--runs", "mlp_baseline", "cnn_baseline", "mlp_dropout", "cnn_dropout",
    "--output", ".\experiment_summary.md"
)

Write-Host ""
Write-Host "Done. Key outputs:"
Write-Host "- best_models\<run_name>\best_model.pickle"
Write-Host "- best_models\<run_name>\learning_curve.png"
Write-Host "- best_models\<run_name>\metrics.json"
Write-Host "- figs\mlp_baseline\*.png"
Write-Host "- figs\cnn_baseline\*.png"
Write-Host "- experiment_summary.md"
