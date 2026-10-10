# Grab the best SwordNet weights from the team's trainer pods and run them locally (Windows).
#
#   .\scripts\get_weights.ps1                       # best run across all trainer seats -> weights\
#   .\scripts\get_weights.ps1 -Run 7                # a specific run
#   .\scripts\get_weights.ps1 -Serve                # ...then open the webcam with it
#   .\scripts\get_weights.ps1 -Serve -Source fight.mp4
#
# Works on any laptop with kubectl access to the cluster; it does not need the FiftyOne DB.
# Credentials come from .env (AWS_*), or whatever your shell already has.
param(
    [string]$Run = "",
    [switch]$Serve,
    [string]$Source = "0",
    [string[]]$Seats = @("seat-145", "seat-146", "seat-147", "seat-148", "seat-149")
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Data = "/workspace/swordnet-data"

if (Test-Path .env) {
    Get-Content .env | Where-Object { $_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.+)$' } | ForEach-Object {
        $null = $_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.+)$'
        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2].Trim().Trim('"').Trim("'")
    }
}
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) { throw "kubectl not found; see TUTORIAL.md" }

$probe = @'
import json, sys, pathlib
data, want = sys.argv[1], sys.argv[2]
for m in pathlib.Path(data, "runs").glob("run-*/metrics.json"):
    run = m.parent
    if want and run.name != f"run-{want}":
        continue
    if (run / "weights" / "best.pt").exists():
        print(json.loads(m.read_text()).get("best_mAP50_95") or 0, run)
'@

$best = $null
foreach ($seat in $Seats) {
    $lines = $probe | kubectl exec -i $seat -c app -- python3 - $Data $Run 2>$null
    foreach ($line in $lines) {
        if (-not $line) { continue }
        $score, $dir = $line -split ' ', 2
        Write-Host "  $seat $(Split-Path $dir -Leaf)  mAP50-95=$score"
        if (-not $best -or [double]$score -gt $best.Score) { $best = @{ Score = [double]$score; Seat = $seat; Dir = $dir } }
    }
}
if (-not $best) { throw "No finished runs with weights found." }

$name = Split-Path $best.Dir -Leaf
New-Item -ItemType Directory -Force weights | Out-Null
Write-Host "Downloading $name from $($best.Seat) (mAP50-95 $($best.Score))..."
# kubectl cp keeps the bytes intact; PowerShell's > redirection would re-encode a binary file.
kubectl cp -c app "$($best.Seat):$($best.Dir)/weights/best.pt" "weights/swordnet-$name.pt"
Copy-Item "weights/swordnet-$name.pt" weights/swordnet-best.pt -Force
Write-Host "Saved weights\swordnet-$name.pt and weights\swordnet-best.pt"

if ($Serve) {
    $py = if (Test-Path .venv\Scripts\python.exe) { ".venv\Scripts\python.exe" } else { "python" }
    & $py -c "import ultralytics" 2>$null
    if ($LASTEXITCODE -ne 0) { & $py -m pip install -q ultralytics opencv-python }
    & $py -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source $Source
}
