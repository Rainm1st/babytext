param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("mask5", "mask10", "mask20", "mask50")]
  [string]$Model,

  [string]$PythonExe = "D:/conda_envs/chatgpt/python.exe",
  [string]$ProjectRoot = "",  # 默认：本仓库根目录（experiments/）
  [int]$NumWorkers = 4,
  [int]$LoggingSteps = 20
)

if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$expDir = Join-Path $ProjectRoot "gpt2-masked-objective"
$trainScript = Join-Path $expDir "train_hybrid_gpt2.py"

$env:TRANSFORMERS_NO_TORCHVISION = "1"
$env:TRANSFORMERS_NO_VISUAL_BACKENDS = "1"
if (-not $env:GPT2_MASKED_RUN_BASE) {
  $env:GPT2_MASKED_RUN_BASE = "/data0/language/babylm_runs/gpt2_masked_objective"
}

switch ($Model) {
  "mask5" {
    $config = Join-Path $expDir "configs/run_mask5.json"
    $output = Join-Path $env:GPT2_MASKED_RUN_BASE "gpt2_mask5_run1"
  }
  "mask10" {
    $config = Join-Path $expDir "configs/run_mask10.json"
    $output = Join-Path $env:GPT2_MASKED_RUN_BASE "gpt2_mask10_run1"
  }
  "mask20" {
    $config = Join-Path $expDir "configs/run_mask20.json"
    $output = Join-Path $env:GPT2_MASKED_RUN_BASE "gpt2_mask20_run1"
  }
  "mask50" {
    $config = Join-Path $expDir "configs/run_mask50.json"
    $output = Join-Path $env:GPT2_MASKED_RUN_BASE "gpt2_mask50_run1"
  }
}

Write-Host "Python: $PythonExe"
Write-Host "Config: $config"
Write-Host "Output: $output"

& $PythonExe $trainScript `
  --config $config `
  --output-dir $output `
  --num-workers $NumWorkers `
  --logging-steps $LoggingSteps
