param(
  [string]$Executable = ".\build\bin\Release\cis5650_boids.exe"
)

$ErrorActionPreference = "Stop"
$capture_directory = Join-Path $PSScriptRoot "capture_frames"
if (Test-Path $capture_directory) {
  Remove-Item $capture_directory -Recurse -Force
}
New-Item -ItemType Directory -Path $capture_directory | Out-Null

$stdout = Join-Path $PSScriptRoot "capture_stdout.txt"
$stderr = Join-Path $PSScriptRoot "capture_stderr.txt"
$arguments = "--window-profile coherent 5000 128 2 120 90 `"$capture_directory`" 3"
$process = Start-Process -FilePath $Executable -ArgumentList $arguments `
  -WorkingDirectory (Get-Location) -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr -PassThru
$process.WaitForExit()

$frames = Get-ChildItem $capture_directory -Filter "frame_*.ppm"
if ($frames.Count -eq 0) {
  throw "Capture produced no frames: $(Get-Content $stderr -Raw)"
}
