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

## 7. Record and open the NVIDIA Nsight reports

The automated route is:

```powershell
powershell -ExecutionPolicy Bypass -File .\profiling\run_nsight_profiles.ps1
```

This launches the application three times. Nsight Systems records one visualized coherent run. Nsight Compute then profiles one scattered neighbor-kernel launch and one coherent neighbor-kernel launch with the `detailed` section set. The script exports the reports, text/CSV data, and README images.

To open the interactive reports yourself:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\host-windows-x64\nsys-ui.exe" `
  .\profiling\results\nsight_systems_coherent.nsys-rep

& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\ncu-ui.bat" `
  .\profiling\results\nsight_compute_coherent.ncu-rep
```

If your installed version is different, replace the version number in the path. You can also open either application from the Start menu and use **File > Open** to select its report.

### Nsight Systems, step by step

1. Open NVIDIA Nsight Systems and create a new project or profiling session.
2. Select the local computer as the target.
3. Set the application to `build\bin\Release\cis5650_boids.exe` and the working directory to the repository root.
4. Use `--window-profile coherent 20000 128 2 15 45` as the command-line arguments.
5. Enable CUDA and OpenGL tracing. CPU sampling is optional and is disabled in the supplied script to keep this GPU timeline small.
6. Start the capture. The application exits automatically after the requested frames and Systems creates the report.
7. In the timeline, expand the process, CUDA HW, and CUDA API rows. Drag across a short interval to zoom into one frame. Click a kernel bar to see its duration and launch details. Open **CUDA GPU Kernel Summary** to see which kernels consume the most total GPU time.

Use Systems when the question is about the whole program: which kernels run, their order, time between launches, CUDA API overhead, or whether CPU and GPU work overlap.

### Nsight Compute, step by step

1. Open NVIDIA Nsight Compute and choose **Start Activity**, then **Profile**.
2. Set the application and working directory to the same values as above.
3. For coherent mode, use `--benchmark coherent 20000 128 2 1 1` as the arguments.
4. In the filter settings, select kernel-name base **function** and filter for `kernUpdateVelNeighborSearchCoherent`. Set launch skip to 1 and launch count to 1. This ignores the first matching warmup launch and collects one measured launch.
5. Select the `detailed` section set and choose an output report path.
6. Launch the activity. The tool replays the chosen kernel several times because the GPU cannot collect every hardware counter in one pass.
7. Start with **Summary** or **Details**. Read **GPU Speed of Light** for compute and memory utilization, **Memory Workload Analysis** for cache behavior, **Occupancy** for resident-warp limits, and **Warp State Statistics** for reasons warps could not issue instructions.
8. Repeat with `--benchmark scattered ...` and the scattered kernel filter. Compare matching metrics under the same boid count, block size, and cell width.

Use Compute when the question is inside one kernel: cache hit rates, memory stalls, branch behavior, occupancy, instruction mix, or whether the kernel is limited by compute or memory behavior. The `detailed` set replays the kernel, so use the ordinary CUDA-event benchmark for final application FPS.

### Nsight debugger from Project 0

The Visual Studio Nsight debugger is for correctness rather than performance. Build the Debug configuration, place a breakpoint in a CUDA kernel, then use **Nsight > Start CUDA Debugging**. When a thread reaches the breakpoint, inspect its `blockIdx`, `threadIdx`, local variables, and device arrays. Use this when a kernel computes the wrong result or accesses bad memory. Return to Release mode for every timing and profiler capture.

If Compute reports that GPU performance counters are unavailable, enable access in the NVIDIA Control Panel under **Desktop > Enable Developer Settings > Developer > Manage GPU Performance Counters**, or run the profiler with an account that has permission.

## 8. Understanding the source changes

`kernel.cu` exposes setters for block size and cell-width multiplier. This allows one Release executable to test every setting without changing source code and recompiling between points.

`main.cpp` selects an implementation at runtime. `--benchmark` bypasses GLFW and measures only CUDA simulation steps. `--window-profile` runs a finite number of complete rendered frames and exits automatically. Its optional capture-directory and capture-frequency arguments call `glReadPixels` before buffer swapping to save the fixed-camera framebuffer.
