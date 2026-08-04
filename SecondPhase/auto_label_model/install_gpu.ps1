param(
    [string]$TorchVersion = "2.13.0+cu130"
)

$ErrorActionPreference = "Stop"
$packageDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = (Resolve-Path (Join-Path $packageDir "..\..\.venv\Scripts\python.exe")).Path

& $pythonPath -m pip install "torch==$TorchVersion" --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $pythonPath -m pip install -r (Join-Path $packageDir "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $pythonPath -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}')"
exit $LASTEXITCODE
