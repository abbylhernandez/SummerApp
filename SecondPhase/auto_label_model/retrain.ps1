param(
    [switch]$Quick,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
)

$packageDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = (Resolve-Path (Join-Path $packageDir "..\..\.venv\Scripts\python.exe")).Path
$trainPath = Join-Path $packageDir "train.py"
$arguments = @($trainPath, "--rescan", "--device", $Device)
if ($Quick) {
    $arguments += "--quick"
}

& $pythonPath @arguments
exit $LASTEXITCODE
