param([int]$Camera = 0, [int]$Port = 8080, [double]$ExitSeconds = 5, [double]$UnattendedSeconds = 30, [string]$Device = 'auto', [int]$FPS = 30, [int]$RelationEvery = 10, [string]$Detector = 'models/yolo26l.pt')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run ./setup.ps1 first.' }
    if (-not (Test-Path -LiteralPath $Detector -PathType Leaf)) {
        if (Test-Path -LiteralPath 'models/yolo26n.pt' -PathType Leaf) {
            Write-Warning "Detector '$Detector' not found. Falling back to 'models/yolo26n.pt'."
            $Detector = 'models/yolo26n.pt'
        }
    }
    $taskRequiredModels = @($Detector, 'models/relsgg-vits16plus/model.pth', 'models/relsgg-vits16plus/text_student.pt', 'models/relsgg-vits16plus/tokenizer.json', 'models/relsgg-vits16plus/tokenizer_config.json', 'models/relsgg-vits16plus/calibration.json')
    $taskMissingModels = @($taskRequiredModels | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($taskMissingModels.Count -gt 0) {
        throw ('Missing local model files. Place downloads here before starting camera:' + [Environment]::NewLine + ($taskMissingModels -join [Environment]::NewLine) + [Environment]::NewLine + 'See models/README.md for download links.')
    }
    $taskPreviousHubOffline = $env:HF_HUB_OFFLINE
    $taskPreviousTransformersOffline = $env:TRANSFORMERS_OFFLINE
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    & $taskPython demo.py --source $Camera --detector $Detector --relation-model models/relsgg-vits16plus/model.pth --web --open-browser --no-preview --port $Port --exit-seconds $ExitSeconds --unattended-seconds $UnattendedSeconds --device $Device --camera-fps $FPS --relation-every $RelationEvery --latest-camera --no-record --max-boxes 16
    if ($LASTEXITCODE -ne 0) { throw 'Camera demo stopped with an error. See the message above.' }
} finally {
    if (Test-Path variable:taskPreviousHubOffline) { $env:HF_HUB_OFFLINE = $taskPreviousHubOffline }
    if (Test-Path variable:taskPreviousTransformersOffline) { $env:TRANSFORMERS_OFFLINE = $taskPreviousTransformersOffline }
    Pop-Location
}
