# CUDA Flocking

![Animated boids](images/boids.gif)

![Boids screenshot](images/boids.png)

University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 1 - Flocking

* Student: Zachary Kelly
* Tested on: Windows 10 Pro, AMD Ryzen 7 3700X, NVIDIA GeForce RTX 3070 8 GB
* CUDA 13.3, NVIDIA driver 616.56, Release build

## Implementations

This project contains three CUDA implementations of Reynolds-style flocking:

* **Naive:** every boid checks every other boid.
* **Scattered uniform grid:** boids are sorted by grid-cell index, but position and velocity reads use the sorted array as an index into the original particle arrays.
* **Coherent uniform grid:** positions and velocities are gathered into grid-sorted arrays before the neighbor search, making reads within a cell contiguous.

## Profiling methodology

I ran all tests in Release mode on the machine listed above. Each plotted point is the mean of three trials, and each error bar is one standard deviation.

For simulation-only results, I disabled the window and used CUDA events around 100 complete simulation steps after 20 warmup steps. Each step includes the work required by that implementation: velocity and position updates for naive; grid construction, Thrust sorting, neighbor search, and updates for the grid methods. FPS is `1000 / mean milliseconds per step`.

For visualization results, I used a 1280 by 720 window with `glfwSwapInterval(0)`. I discarded 60 warmup frames and timed 120 complete frames. This measurement includes CUDA/OpenGL buffer mapping, simulation, VBO copies, drawing, and buffer swapping. The camera stayed fixed.

Unless a graph says otherwise, the block size was 128 threads and the grid cell width was twice the maximum flocking-rule distance. The raw trials and averaged values are in [`profiling/results`](profiling/results), and the commands used to reproduce them are in [`profiling/run_profiles.ps1`](profiling/run_profiles.ps1).

## Boid count

![Simulation-only boid count profile](images/profile_boid_count_off.png)

The naive method loses performance rapidly as the number of boids increases. It checks every possible pair, so the amount of comparison work grows approximately with the square of the boid count. GPU occupancy hides some of this growth at the smallest counts, but by 40,000 boids the measured rate falls to 20.0 simulation steps per second.

The grid methods first pay the cost of computing grid indices, sorting, and building cell ranges. They then inspect only cells that can contain relevant neighbors. Their simulation-only rates remain much flatter: at 40,000 boids, scattered reaches 2,617 FPS and coherent reaches 4,073 FPS. The spatial culling more than repays its preprocessing cost once the flock is large.

![Visualized boid count profile](images/profile_boid_count_on.png)

With visualization enabled, the scattered and coherent implementations cluster around 460-520 FPS. Their simulation steps are faster than the CUDA/OpenGL mapping, VBO copy, draw, and presentation work, so graphics overhead becomes the main frame cost. The naive method still declines with boid count because its simulation eventually dominates that fixed graphics cost.

## Coherent versus scattered grid

The coherent layout improved simulation-only performance. In the boid-count experiment, at 20,000 boids it averaged 4,829 FPS compared with 4,149 FPS for scattered, an improvement of about 16%. At 40,000 boids it averaged 4,073 FPS compared with 2,617 FPS, an improvement of about 56%.

This was the expected direction. Threads processing nearby boids repeatedly read positions and velocities from the same cell ranges. Gathering those values into grid order turns indirect, scattered reads into neighboring reads. The gain is modest at small counts because sorting and gathering add fixed work, and it is mostly hidden in the visualized results because rendering costs more than either grid simulation.

## Block size and block count

![Block-size profile](images/profile_block_size.png)

Changing the block size also changes the block count according to `ceil(numberOfBoids / blockSize)`. More, smaller blocks give the scheduler flexibility to distribute work, while very large blocks reduce the number of blocks that can reside on a streaming multiprocessor when registers, warps, or other resources become limiting.

For naive, 32 and 64 threads were best at about 78.8 FPS, while 1,024 threads fell to 49.8 FPS. Scattered was also strongest at 32-64 threads and gradually declined at larger sizes. Coherent stayed within roughly 4,600-4,830 FPS across the tested sizes and peaked at 512 threads. There is no universal best block size here because the kernels differ in control flow and memory-access behavior; 128 remains a reasonable common setting, although it was not the measured optimum for every implementation on this GPU.

## Cell width: 27 cells versus 8 cells

![Cell-width profile](images/profile_cell_width.png)

Using cells twice the maximum interaction distance, which requires at most eight candidate cells, was 2.76 times faster for scattered and 2.92 times faster for coherent than using cells equal to the interaction distance and checking as many as 27 cells.

The explanation is more specific than "27 is greater than 8." Smaller cells contain fewer particles, which can reduce the number of distance tests. At uniform density, 27 cells of width `r` cover less total volume than eight cells of width `2r`, so the 27-cell version can theoretically examine fewer particle candidates. In this test, however, the extra nested-loop iterations, cell-index calculations, and start/end lookups were expensive, and many visited cells were empty. Those costs outweighed the reduction in candidates per cell.

## Reproducing the results

Build and run the full profiling sweep from PowerShell at the repository root:

```powershell
cmake --build build --config Release --parallel
powershell -ExecutionPolicy Bypass -File .\profiling\run_profiles.ps1 -Trials 3
powershell -ExecutionPolicy Bypass -File .\profiling\capture_boids.ps1
py .\profiling\make_outputs.py
```

One simulation-only measurement can be reproduced directly:

```powershell
.\build\bin\Release\cis5650_boids.exe --benchmark coherent 20000 128 2 20 100
```

The arguments are implementation, boid count, block size, cell-width multiplier, warmup steps, and measured steps. The program prints the mean milliseconds per step and FPS.

One visualized measurement can be reproduced with:

```powershell
.\build\bin\Release\cis5650_boids.exe --window-profile coherent 20000 128 2 60 120
```

The visual mode measures the complete rendered frame. VSync is disabled in the application for profiling.
