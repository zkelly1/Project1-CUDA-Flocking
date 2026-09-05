param(
  [string]$Executable = ".\build\bin\Release\cis5650_boids.exe",
  [string]$Nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe",
  [string]$Ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\ncu.bat"
)

$ErrorActionPreference = "Stop"
$results = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $results | Out-Null

& $Nsys profile --trace=cuda,opengl,nvtx --sample=none --cpuctxsw=none `
  --force-overwrite=true --output="$results\nsight_systems_coherent" `
  $Executable --window-profile coherent 20000 128 2 15 45

& $Nsys stats --report cuda_gpu_kern_sum,cuda_api_sum `
  "$results\nsight_systems_coherent.nsys-rep" |
  Set-Content -Encoding utf8 "$results\nsight_systems_summary.txt"

foreach ($mode in @("scattered", "coherent")) {
  $kernel = "regex:.*kernUpdateVelNeighborSearch$($mode.Substring(0, 1).ToUpper())$($mode.Substring(1)).*"
  & $Ncu --set detailed --kernel-name-base function --kernel-name $kernel `
    --launch-skip 1 --launch-count 1 --export "$results\nsight_compute_$mode" `
    --force-overwrite $Executable --benchmark $mode 20000 128 2 1 1

  & $Ncu --import "$results\nsight_compute_$mode.ncu-rep" --page raw --csv |
    Set-Content -Encoding utf8 "$results\nsight_compute_$mode.csv"
}

$python = "py"
& $python (Join-Path $PSScriptRoot "render_nsight_reports.py")
