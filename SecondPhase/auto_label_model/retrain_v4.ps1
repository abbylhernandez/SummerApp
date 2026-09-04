param(
    [switch]$Quick,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
)

$packageDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = (Resolve-Path (Join-Path $packageDir "..\..\.venv\Scripts\python.exe")).Path
$builderPath = Join-Path $packageDir "dataset_builder.py"
$cachePath = Join-Path $packageDir "cache_video_features.py"
$trainPath = Join-Path $packageDir "train_v4_multimodal.py"

& $pythonPath $builderPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonPath $cachePath "--device" $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$arguments = @($trainPath, "--device", $Device)
if ($Quick) { $arguments += "--quick" }
& $pythonPath @arguments
exit $LASTEXITCODE
