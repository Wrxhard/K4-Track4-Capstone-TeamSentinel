$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { throw 'Run ./setup.ps1 first.' }
    & ./.venv/Scripts/python.exe -m pip install --ignore-installed --no-deps 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & ./.venv/Scripts/python.exe -m pip uninstall -y onnxruntime
    & ./.venv/Scripts/python.exe -m pip install onnxruntime-gpu
    & ./.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver'; x=torch.ones(32,32,device='cuda'); print('PyTorch CUDA:', torch.__version__, torch.cuda.get_device_name(0)); print((x@x).mean().item())"
    if ($LASTEXITCODE -ne 0) { throw 'GPU check failed.' }
    Write-Host 'GPU ready for both PyTorch and ONNX! Run ./start_camera.ps1 -Device cuda'
} finally { Pop-Location }
