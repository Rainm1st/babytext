param(
  [string]$PythonExe = "D:/conda_envs/chatgpt/python.exe",
  [string]$ProjectRoot = "",
  [int]$NumWorkers = 4,
  [int]$LoggingSteps = 20,
  [int]$EvalMaxSamples = -1
)

if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$expDir = Join-Path $ProjectRoot "gpt2-masked-objective"
$runTrain = Join-Path $expDir "run_train.ps1"
$preflight = Join-Path $expDir "prepare_experiment.py"
$evalScript = Join-Path $expDir "evaluate_and_fill_tables.py"
if (-not $env:GPT2_MASKED_RUN_BASE) {
  $env:GPT2_MASKED_RUN_BASE = "/data0/language/babylm_runs/gpt2_masked_objective"
}
$outDir = $env:GPT2_MASKED_RUN_BASE
$resultDir = Join-Path $expDir "results"

$env:TRANSFORMERS_NO_TORCHVISION = "1"
$env:TRANSFORMERS_NO_VISUAL_BACKENDS = "1"

Write-Host "== Preflight (mask-only) =="
& $PythonExe $preflight --no-baseline-compare
if ($LASTEXITCODE -ne 0) { throw "Preflight failed." }

Write-Host "== Train mask5 =="
& $runTrain -Model mask5 -PythonExe $PythonExe -ProjectRoot $ProjectRoot -NumWorkers $NumWorkers -LoggingSteps $LoggingSteps
if ($LASTEXITCODE -ne 0) { throw "mask5 training failed." }

Write-Host "== Train mask10 =="
& $runTrain -Model mask10 -PythonExe $PythonExe -ProjectRoot $ProjectRoot -NumWorkers $NumWorkers -LoggingSteps $LoggingSteps
if ($LASTEXITCODE -ne 0) { throw "mask10 training failed." }

$mask5Summary = Join-Path $outDir "gpt2_mask5_run1/run_summary.json"
$mask10Summary = Join-Path $outDir "gpt2_mask10_run1/run_summary.json"
$compareJson = Join-Path $resultDir "mask_runs_summary_compare.json"

if (!(Test-Path $mask5Summary)) { throw "Missing $mask5Summary" }
if (!(Test-Path $mask10Summary)) { throw "Missing $mask10Summary" }

$m5 = Get-Content $mask5Summary | ConvertFrom-Json
$m10 = Get-Content $mask10Summary | ConvertFrom-Json

$comparison = [ordered]@{
  created_at = (Get-Date).ToString("s")
  baseline_reference = "official_gpt2_baseline"
  run_mask5 = $m5
  run_mask10 = $m10
  deltas = [ordered]@{
    masked_alpha = [double]$m10.masked_alpha - [double]$m5.masked_alpha
    words_seen = [double]$m10.words_seen - [double]$m5.words_seen
    global_step = [double]$m10.global_step - [double]$m5.global_step
  }
}

$comparison | ConvertTo-Json -Depth 8 | Set-Content -Path $compareJson -Encoding UTF8
Write-Host "Wrote comparison summary: $compareJson"

Write-Host "== Evaluate and fill tables =="
& $PythonExe $evalScript `
  --mask5-summary $mask5Summary `
  --mask10-summary $mask10Summary `
  --raw-csv (Join-Path $resultDir "raw_scores_template.csv") `
  --delta-csv (Join-Path $resultDir "delta_vs_baseline_template.csv") `
  --output-json (Join-Path $resultDir "eval_mask_runs.json") `
  --max-samples $EvalMaxSamples
if ($LASTEXITCODE -ne 0) { throw "Evaluation and table update failed." }

Write-Host "All done: training + evaluation + table fill completed."
