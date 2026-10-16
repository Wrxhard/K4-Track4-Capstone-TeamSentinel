param([int]$Camera = 0, [int]$Port = 8080, [double]$ExitSeconds = 5, [double]$UnattendedSeconds = 30)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run ./setup.ps1 first.' }
    $taskRequiredModels = @('models/yolo26n.pt', 'models/relsgg-vits16plus/model.pth', 'models/relsgg-vits16plus/text_student.pt', 'models/relsgg-vits16plus/tokenizer.json', 'models/relsgg-vits16plus/tokenizer_config.json', 'models/relsgg-vits16plus/calibration.json')
    $taskMissingModels = @($taskRequiredModels | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($taskMissingModels.Count -gt 0) {
        throw ('Missing local model files. Place downloads here before starting camera:' + [Environment]::NewLine + ($taskMissingModels -join [Environment]::NewLine) + [Environment]::NewLine + 'See models/README.md for download links.')
    }
    $taskPreviousHubOffline = $env:HF_HUB_OFFLINE
    $taskPreviousTransformersOffline = $env:TRANSFORMERS_OFFLINE
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    & $taskPython demo.py --source $Camera --detector models/yolo26n.pt --relation-model models/relsgg-vits16plus/model.pth --web --open-browser --no-preview --port $Port --exit-seconds $ExitSeconds --unattended-seconds $UnattendedSeconds
    if ($LASTEXITCODE -ne 0) { throw 'Camera demo stopped with an error. See the message above.' }
} finally {
    if (Test-Path variable:taskPreviousHubOffline) { $env:HF_HUB_OFFLINE = $taskPreviousHubOffline }
    if (Test-Path variable:taskPreviousTransformersOffline) { $env:TRANSFORMERS_OFFLINE = $taskPreviousTransformersOffline }
    Pop-Location
}
