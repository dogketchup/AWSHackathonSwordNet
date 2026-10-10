# Download SwordNet weights from the team registry on the cluster and run them locally (Windows).
#
#   .\scripts\get_weights.ps1                       # list published models, download the best
#   .\scripts\get_weights.ps1 -Run 7                # a specific run
#   .\scripts\get_weights.ps1 -Serve                # ...then open the webcam with it
#   .\scripts\get_weights.ps1 -Serve -Source fight.mp4
#
# Works for any teammate with kubectl access and this repo; it does not need the FiftyOne DB.
# The registry lives on seat-145 (see swordnet/bridge/registry.py).
param(
    [string]$Run = "",
    [switch]$Serve,
    [string]$Source = "0"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) { throw "kubectl not found; see TUTORIAL.md" }
$py = if (Test-Path .venv\Scripts\python.exe) { ".venv\Scripts\python.exe" } else { "python" }
$env:PYTHONPATH = ""

& $py -m swordnet.bridge.cli models
if ($LASTEXITCODE -ne 0) { throw "could not read the registry" }
$dl = @("-m", "swordnet.bridge.cli", "download", "--out", "weights")
if ($Run) { $dl += @("--run", $Run) }
& $py @dl
if ($LASTEXITCODE -ne 0) { throw "download failed" }

if ($Serve) {
    & $py -c "import ultralytics" 2>$null
    if ($LASTEXITCODE -ne 0) { & $py -m pip install -q ultralytics opencv-python }
    & $py -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source $Source
}
