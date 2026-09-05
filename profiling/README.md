# Reproducing the profiling results

Run every command from the repository root in PowerShell.

## 1. Build the Release executable

```powershell
cmake --build build --config Release --parallel
```

Release mode matters because compiler optimization changes CUDA kernel performance substantially.

## 2. Run one simulation-only measurement

```powershell
.\build\bin\Release\cis5650_boids.exe --benchmark coherent 20000 128 2 20 100
```

The arguments after `--benchmark` are:

1. implementation: `naive`, `scattered`, or `coherent`
2. number of boids
3. threads per block
4. cell-width multiplier: `2` searches at most 8 cells and `1` searches at most 27
5. warmup steps
6. measured steps

The program initializes a fresh deterministic simulation, completes the warmup, synchronizes the GPU, and records the measured steps with CUDA events. A result such as

```text
coherent,20000,128,2,100,0.207065,4829.4
```

contains implementation, boids, block size, cell-width multiplier, measured steps, milliseconds per step, and FPS.

## 3. Run one measurement with visualization

```powershell
.\build\bin\Release\cis5650_boids.exe --window-profile coherent 20000 128 2 60 120
```

The first four parameters have the same meaning. The final two are warmup frames and measured frames. This path times the complete frame, including CUDA/OpenGL buffer mapping, simulation, VBO copies, drawing, and buffer swapping. `glfwSwapInterval(0)` disables application-level VSync.

## 4. Reproduce the complete data sweep

```powershell
powershell -ExecutionPolicy Bypass -File .\profiling\run_profiles.ps1 -Trials 3
```

The script runs each configuration sequentially so separate processes never compete for the GPU. It measures:

* 1,000, 5,000, 10,000, 20,000, and 40,000 boids for all three implementations, with and without visualization
* block sizes 32, 64, 128, 256, 512, and 1,024 for all three implementations at 20,000 boids
* cell-width multipliers 1 and 2 for both grid implementations at 20,000 boids

The raw output is `profiling/results/raw_profile_results.csv`. Keep the computer plugged in, close GPU-heavy applications, and avoid moving or resizing the profiling window during the run.

To change the experiment, edit the `$boid_counts` or `$block_sizes` arrays near the bottom of `run_profiles.ps1`. Change `-Trials 3` to collect more or fewer repetitions.

## 5. Capture the still and animation

```powershell
powershell -ExecutionPolicy Bypass -File .\profiling\capture_boids.ps1
```

This launches the coherent simulation with 5,000 boids and a fixed camera. It warms up for 120 frames, measures another 90 frames, and saves every third back-buffer image as a PPM file. Capturing is done in a separate run so framebuffer reads do not contaminate the performance measurements.

## 6. Generate the plots and final media

```powershell
py .\profiling\make_outputs.py
```

The script averages the three trials, computes standard deviations, draws the four PNG plots, converts the last captured frame to `images/boids.png`, and combines the frames into `images/boids.gif`. It also writes `profiling/results/profile_summary.csv`.

The plotting script requires `pandas` and `Pillow`. If those modules are installed in your normal Python environment, you can use `py .\profiling\make_outputs.py` instead.

## 7. Understanding the source changes

`kernel.cu` exposes setters for block size and cell-width multiplier. This allows one Release executable to test every setting without changing source code and recompiling between points.

`main.cpp` selects an implementation at runtime. `--benchmark` bypasses GLFW and measures only CUDA simulation steps. `--window-profile` runs a finite number of complete rendered frames and exits automatically. Its optional capture-directory and capture-frequency arguments call `glReadPixels` before buffer swapping to save the fixed-camera framebuffer.
