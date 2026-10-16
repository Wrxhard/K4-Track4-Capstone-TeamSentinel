$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ required'"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12+ is required.' }
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
    }
    & ./.venv/Scripts/python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install demo runtime dependencies.' }
    & ./.venv/Scripts/python.exe -m pip install --no-deps 'git+https://github.com/Maelic/RelateAnything.git@197ac47d645f106e8beb79e49a3f62b91fcbe91f'
    if ($LASTEXITCODE -ne 0) { throw 'Could not install RelateAnything.' }
    & ./.venv/Scripts/python.exe -c "from relsgg import RelateAnything; import supervision; import cv2; print('Runtime imports OK')"
    if ($LASTEXITCODE -ne 0) { throw 'Runtime import check failed.' }
    Write-Host 'Ready. Run: ./.venv/Scripts/python.exe demo.py --source 0'
} finally {
    Pop-Location
}
