import csv
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "profiling" / "results"
IMAGES = ROOT / "images"
FONT_DIRECTORY = Path("C:/Windows/Fonts")

BACKGROUND = "#171A1F"
PANEL = "#23272E"
GRID = "#3A404A"
TEXT = "#F4F6F8"
MUTED = "#AAB2BF"
GREEN = "#76B900"
CYAN = "#43BCCD"
ORANGE = "#F4A261"
PURPLE = "#9B7EDE"


def font(size, bold=False):
    filename = "seguisb.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(FONT_DIRECTORY / filename), size)


TITLE = font(38, True)
HEADING = font(25, True)
BODY = font(21)
SMALL = font(17)


def base(title, subtitle):
    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 8), fill=GREEN)
    draw.text((62, 38), title, fill=TEXT, font=TITLE)
    draw.text((64, 92), subtitle, fill=MUTED, font=BODY)
    return image, draw


def load_ncu(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    headings, units, values = rows[:3]
    return dict(zip(headings, values)), dict(zip(headings, units))


def metric(data, key, digits=2):
    return f"{float(data[key].replace(',', '')):.{digits}f}"


def render_ncu(mode):
    data, _ = load_ncu(RESULTS / f"nsight_compute_{mode}.csv")
    label = mode.capitalize()
    image, draw = base(
        f"NVIDIA Nsight Compute — {label} neighbor kernel",
        "RTX 3070 · 20,000 boids · 128 threads/block · cell width = 2r · one profiled launch",
    )

    cards = [
        ("Kernel duration", metric(data, "gpu__time_duration.sum"), "µs", GREEN),
        ("Compute throughput", metric(data, "sm__throughput.avg.pct_of_peak_sustained_elapsed"), "%", CYAN),
        ("Memory throughput", metric(data, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"), "%", ORANGE),
        ("L1/TEX hit rate", metric(data, "l1tex__t_sector_hit_rate.pct"), "%", PURPLE),
    ]
    for index, (name, value, unit, color) in enumerate(cards):
        left = 62 + index * 380
        draw.rounded_rectangle((left, 145, left + 350, 290), 12, fill=PANEL, outline=GRID, width=2)
        draw.text((left + 24, 167), name, fill=MUTED, font=SMALL)
        draw.text((left + 24, 207), value, fill=color, font=font(39, True))
        draw.text((left + 235, 222), unit, fill=MUTED, font=BODY)

    draw.rounded_rectangle((62, 325, 980, 806), 12, fill=PANEL, outline=GRID, width=2)
    draw.text((90, 353), "Selected hardware metrics", fill=TEXT, font=HEADING)
    rows = [
        ("DRAM throughput", metric(data, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed") + "%"),
        ("L2 hit rate", metric(data, "lts__t_sector_hit_rate.pct") + "%"),
        ("Achieved occupancy", metric(data, "sm__warps_active.avg.pct_of_peak_sustained_active") + "%"),
        ("Registers per thread", data["launch__registers_per_thread"]),
        ("Uniform branch targets", metric(data, "smsp__sass_average_branch_targets_threads_uniform.pct") + "%"),
        ("Long-scoreboard stall samples", data["smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"]),
        ("Excess global L2 sectors", metric(data, "derived__memory_l2_theoretical_sectors_global_excessive") + " MB"),
    ]
    y = 415
    for name, value in rows:
        draw.line((90, y + 43, 950, y + 43), fill=GRID, width=1)
        draw.text((95, y), name, fill=MUTED, font=BODY)
        draw.text((925, y), value, fill=TEXT, font=BODY, anchor="ra")
        y += 55

    draw.rounded_rectangle((1010, 325, 1538, 806), 12, fill=PANEL, outline=GRID, width=2)
    draw.text((1040, 353), "How to read this", fill=TEXT, font=HEADING)
    notes = (
        [
            "The coherent kernel completes",
            "this launch in less than half",
            "the scattered kernel's time.",
            "",
            "Its much higher L1 hit rate",
            "shows the benefit of storing",
            "nearby boids next to each other.",
            "",
            "Fewer long-scoreboard stalls",
            "mean warps wait less often for",
            "dependent memory loads.",
        ]
        if mode == "coherent"
        else [
            "Indirect particle-array reads",
            "produce a much lower L1 hit rate",
            "and a longer kernel duration.",
            "",
            "The high memory-throughput value",
            "does not mean useful data access",
            "is efficient; extra transactions",
            "can keep memory hardware busy.",
            "",
            "Long-scoreboard stalls show warps",
            "waiting on memory dependencies.",
        ]
    )
    y = 420
    for line in notes:
        draw.text((1042, y), line, fill=TEXT if line else MUTED, font=BODY)
        y += 34

    output = IMAGES / f"nsight-compute-{mode}.png"
    image.save(output)
    return data


def render_nsys():
    database = sqlite3.connect(RESULTS / "nsight_systems_coherent.sqlite")
    kernels = list(database.execute("""
        SELECT k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.shortName
        ORDER BY k.start
    """))
    target_index = next(
        i for i, row in enumerate(kernels[20:], start=20)
        if row[2] == "kernUpdateVelNeighborSearchCoherent"
    )
    frame_start = target_index
    while frame_start > 0 and kernels[frame_start][2] != "kernComputeIndices":
        frame_start -= 1
    frame_end = target_index
    while frame_end + 1 < len(kernels) and kernels[frame_end][2] != "kernCopyVelocitiesToVBO":
        frame_end += 1
    frame = kernels[frame_start:frame_end + 1]
    start = frame[0][0]
    end = frame[-1][1]

    image, draw = base(
        "NVIDIA Nsight Systems — one coherent simulation frame",
        "CUDA GPU timeline · RTX 3070 · 20,000 boids · captured from the visualized application",
    )
    left, right, top, bottom = 315, 1530, 190, 725
    draw.rounded_rectangle((45, 145, 1555, 815), 12, fill=PANEL, outline=GRID, width=2)
    total_us = (end - start) / 1000
    for tick in range(6):
        x = left + (right - left) * tick / 5
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.text((x, 160), f"{total_us * tick / 5:.0f} µs", fill=MUTED, font=SMALL, anchor="ma")

    colors = {
        "kernComputeIndices": CYAN,
        "DeviceRadixSortHistogramKernel": PURPLE,
        "DeviceRadixSortExclusiveSumKernel": PURPLE,
        "DeviceRadixSortOnesweepKernel": PURPLE,
        "kernResetIntBuffer": ORANGE,
        "kernIdentifyCellStartEnd": ORANGE,
        "kernGatherCoherent": GREEN,
        "kernUpdateVelNeighborSearchCoherent": "#E76F51",
        "kernUpdatePos": CYAN,
        "kernCopyPositionsToVBO": GREEN,
        "kernCopyVelocitiesToVBO": GREEN,
    }
    display_names = {
        "DeviceRadixSortHistogramKernel": "CUB radix histogram",
        "DeviceRadixSortExclusiveSumKernel": "CUB radix prefix sum",
        "DeviceRadixSortOnesweepKernel": "CUB radix sort pass",
        "kernUpdateVelNeighborSearchCoherent": "neighbor search (coherent)",
    }
    y = top + 8
    for kernel_start, kernel_end, name in frame:
        x1 = left + (kernel_start - start) / (end - start) * (right - left)
        x2 = left + (kernel_end - start) / (end - start) * (right - left)
        duration = (kernel_end - kernel_start) / 1000
        label = display_names.get(name, name.replace("kern", ""))
        draw.text((70, y + 9), label, fill=TEXT, font=SMALL)
        draw.rounded_rectangle((x1, y + 5, max(x1 + 3, x2), y + 28), 4, fill=colors.get(name, CYAN))
        if x2 - x1 > 62:
            draw.text(((x1 + x2) / 2, y + 16), f"{duration:.1f} µs", fill=BACKGROUND, font=font(14, True), anchor="mm")
        y += 34

    draw.text((70, 758), "Each bar is one CUDA kernel; blank space is CPU/API work between launches.", fill=MUTED, font=SMALL)
    draw.text((70, 784), f"Displayed frame span: {total_us:.1f} µs. Full capture contains 60 coherent neighbor-search launches.", fill=MUTED, font=SMALL)
    output = IMAGES / "nsight-systems-timeline.png"
    image.save(output)


def write_comparison(coherent, scattered):
    rows = [
        ["metric", "scattered", "coherent"],
        ["kernel_duration_us", scattered["gpu__time_duration.sum"], coherent["gpu__time_duration.sum"]],
        ["l1_tex_hit_rate_pct", scattered["l1tex__t_sector_hit_rate.pct"], coherent["l1tex__t_sector_hit_rate.pct"]],
        ["l2_hit_rate_pct", scattered["lts__t_sector_hit_rate.pct"], coherent["lts__t_sector_hit_rate.pct"]],
        ["long_scoreboard_stall_samples", scattered["smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"], coherent["smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"]],
    ]
    with (RESULTS / "nsight_compute_comparison.csv").open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows(rows)


def main():
    IMAGES.mkdir(exist_ok=True)
    coherent = render_ncu("coherent")
    scattered = render_ncu("scattered")
    render_nsys()
    write_comparison(coherent, scattered)


if __name__ == "__main__":
    main()
