param(
  [string]$Executable = ".\build\bin\Release\cis5650_boids.exe",
  [int]$Trials = 3
)

$ErrorActionPreference = "Stop"
$output_directory = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $output_directory | Out-Null

function Run-SimulationOnlyProfile {
  param(
    [string]$Mode,
    [int]$Boids,
    [int]$BlockSize,
    [float]$CellWidthMultiplier,
    [int]$Trial,
    [string]$Experiment
  )

  $line = & $Executable --benchmark $Mode $Boids $BlockSize `
    $CellWidthMultiplier 20 100 | Select-Object -Last 1
  $parts = $line -split ","
  [pscustomobject]@{
    experiment = $Experiment
    visualization = "off"
    mode = $parts[0]
    boids = [int]$parts[1]
    block_size = [int]$parts[2]
    cell_width_multiplier = [float]$parts[3]
    trial = $Trial
    measured_steps = [int]$parts[4]
    ms_per_step = [float]$parts[5]
    fps = [float]$parts[6]
  }
}

function Run-WindowProfile {
  param(
    [string]$Mode,
    [int]$Boids,
    [int]$Trial
  )

  $stdout = Join-Path $output_directory "window_stdout.txt"
  $stderr = Join-Path $output_directory "window_stderr.txt"
  $arguments = "--window-profile $Mode $Boids 128 2 60 120"
  $process = Start-Process -FilePath $Executable -ArgumentList $arguments `
    -WorkingDirectory (Get-Location) -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -PassThru
  $process.WaitForExit()

  $line = Get-Content $stdout | Where-Object {
    $_ -match "^(naive|scattered|coherent),"
  } | Select-Object -Last 1
  if (-not $line) {
    throw "Window profile produced no measurement: $(Get-Content $stderr -Raw)"
  }
  $parts = $line -split ","
  [pscustomobject]@{
    experiment = "boid_count"
    visualization = "on"
    mode = $parts[0]
    boids = [int]$parts[1]
    block_size = 128
    cell_width_multiplier = 2.0
    trial = $Trial
    measured_steps = [int]$parts[2]
    ms_per_step = [float]$parts[3]
    fps = [float]$parts[4]
  }
}

$rows = [System.Collections.Generic.List[object]]::new()
$modes = @("naive", "scattered", "coherent")
$boid_counts = @(1000, 5000, 10000, 20000, 40000)

foreach ($mode in $modes) {
  foreach ($boids in $boid_counts) {
    for ($trial = 1; $trial -le $Trials; $trial++) {
      $rows.Add((Run-SimulationOnlyProfile $mode $boids 128 2 $trial "boid_count"))
    }
  }
}

foreach ($mode in $modes) {
  foreach ($boids in $boid_counts) {
    for ($trial = 1; $trial -le $Trials; $trial++) {
      $rows.Add((Run-WindowProfile $mode $boids $trial))
    }
  }
}

$block_sizes = @(32, 64, 128, 256, 512, 1024)
foreach ($mode in $modes) {
  foreach ($current_block_size in $block_sizes) {
    for ($trial = 1; $trial -le $Trials; $trial++) {
      $rows.Add((Run-SimulationOnlyProfile $mode 20000 $current_block_size 2 `
        $trial "block_size"))
    }
  }
}

foreach ($mode in @("scattered", "coherent")) {
  foreach ($multiplier in @(1.0, 2.0)) {
    for ($trial = 1; $trial -le $Trials; $trial++) {
      $rows.Add((Run-SimulationOnlyProfile $mode 20000 128 $multiplier `
        $trial "cell_width"))
    }
  }
}

$rows | Export-Csv (Join-Path $output_directory "raw_profile_results.csv") `
  -NoTypeInformation
Remove-Item (Join-Path $output_directory "window_stdout.txt") -ErrorAction Ignore
Remove-Item (Join-Path $output_directory "window_stderr.txt") -ErrorAction Ignore
