from math import ceil, floor, log10
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "profiling" / "results"
IMAGES = ROOT / "images"
RAW_DATA = RESULTS / "raw_profile_results.csv"
FONT_DIRECTORY = Path("C:/Windows/Fonts")

COLORS = {"naive": "#D14B5A", "scattered": "#2878B5", "coherent": "#2A9D5B"}
LABELS = {"naive": "Naive", "scattered": "Scattered grid", "coherent": "Coherent grid"}
BACKGROUND, INK, MUTED, GRID = "#F7F9FC", "#172033", "#667085", "#D8DEE9"


def font(size, bold=False):
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(FONT_DIRECTORY / name), size)


TITLE_FONT, SUBTITLE_FONT = font(42, True), font(23)
AXIS_FONT, TICK_FONT = font(22), font(19)
LEGEND_FONT, SMALL_FONT = font(21, True), font(17)


def grouped(data, columns):
    return data.groupby(columns, as_index=False)["fps"].agg(mean="mean", std="std").sort_values(columns)


def base_canvas(title, subtitle):
    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), title, fill=INK, font=TITLE_FONT)
    draw.text((92, 102), subtitle, fill=MUTED, font=SUBTITLE_FONT)
    return image, draw


def text_center(draw, xy, value, selected_font, fill=INK):
    box = draw.textbbox((0, 0), value, font=selected_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, fill=fill, font=selected_font)


def nice_linear_ticks(maximum, count=5):
    raw_step = maximum / count
    magnitude = 10 ** floor(log10(raw_step))
    normalized = raw_step / magnitude
    step = magnitude if normalized <= 1 else 2 * magnitude if normalized <= 2 else 5 * magnitude if normalized <= 5 else 10 * magnitude
    top = ceil(maximum / step) * step
    return [step * index for index in range(int(top / step) + 1)], top


def draw_line_chart(data, path, title, subtitle, logarithmic=False):
    image, draw = base_canvas(title, subtitle)
    left, top, right, bottom = 155, 175, 1530, 765
    modes = ["naive", "scattered", "coherent"]
    all_x = sorted(data["boids"].unique())

    if logarithmic:
        minimum_y = 10 ** floor(log10(max(1, data["mean"].min())))
        maximum_y = 10 ** ceil(log10(data["mean"].max()))
        ticks = [10 ** exponent for exponent in range(int(log10(minimum_y)), int(log10(maximum_y)) + 1)]
        def map_y(value):
            fraction = (log10(value) - log10(minimum_y)) / (log10(maximum_y) - log10(minimum_y))
            return bottom - fraction * (bottom - top)
    else:
        ticks, maximum_y = nice_linear_ticks(data["mean"].max() * 1.08)
        minimum_y = 0.0
        def map_y(value):
            return bottom - value / maximum_y * (bottom - top)

    def map_x(value):
        return left + (value - all_x[0]) / (all_x[-1] - all_x[0]) * (right - left)

    for tick in ticks:
        y = map_y(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 15, y), f"{int(tick):,}", fill=MUTED, font=TICK_FONT, anchor="rm")
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for value in all_x:
        x = map_x(value)
        draw.line((x, bottom, x, bottom + 8), fill=INK, width=2)
        text_center(draw, (x, bottom + 34), f"{int(value):,}", TICK_FONT, MUTED)

    for mode in modes:
        rows = data[data["mode"] == mode]
        points = [(map_x(row.boids), map_y(row.mean)) for row in rows.itertuples()]
        draw.line(points, fill=COLORS[mode], width=5, joint="curve")
        for row, (x, y) in zip(rows.itertuples(), points):
            high = map_y(row.mean + row.std)
            low = map_y(max(row.mean - row.std, minimum_y if logarithmic else 0.0))
            draw.line((x, high, x, low), fill=COLORS[mode], width=2)
            draw.line((x - 7, high, x + 7, high), fill=COLORS[mode], width=2)
            draw.line((x - 7, low, x + 7, low), fill=COLORS[mode], width=2)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[mode])

    text_center(draw, ((left + right) / 2, 840), "Number of boids", AXIS_FONT)
    rotated = Image.new("RGBA", (330, 45), (0, 0, 0, 0))
    ImageDraw.Draw(rotated).text((0, 0), "Frames per second", fill=INK, font=AXIS_FONT)
    rotated = rotated.rotate(90, expand=True)
    image.paste(rotated, (35, 350), rotated)
    for index, mode in enumerate(modes):
        x = 1000 + index * 190
        draw.line((x, 145, x + 35, 145), fill=COLORS[mode], width=5)
        draw.text((x + 45, 145), LABELS[mode], fill=INK, font=SMALL_FONT, anchor="lm")
    image.save(path, optimize=True)


def draw_block_chart(data, path):
    image, draw = base_canvas("Block size versus performance", "Simulation only, 20,000 boids, mean of three trials; bars show one standard deviation")
    modes = ["naive", "scattered", "coherent"]
    panel_width, panel_gap, first_left = 430, 55, 125
    for panel_index, mode in enumerate(modes):
        rows = data[data["mode"] == mode]
        left = first_left + panel_index * (panel_width + panel_gap)
        right, top, bottom = left + panel_width, 235, 735
        ticks, maximum_y = nice_linear_ticks(rows["mean"].max() * 1.12, 4)
        def map_y(value):
            return bottom - value / maximum_y * (bottom - top)
        x_values = list(rows["block_size"])
        def map_x(value):
            return left + x_values.index(value) / (len(x_values) - 1) * (right - left)
        draw.text((left, 180), LABELS[mode], fill=COLORS[mode], font=LEGEND_FONT)
        for tick in ticks:
            y = map_y(tick)
            draw.line((left, y, right, y), fill=GRID, width=2)
            draw.text((left - 10, y), f"{int(tick):,}", fill=MUTED, font=SMALL_FONT, anchor="rm")
        draw.line((left, top, left, bottom), fill=INK, width=3)
        draw.line((left, bottom, right, bottom), fill=INK, width=3)
        points = [(map_x(row.block_size), map_y(row.mean)) for row in rows.itertuples()]
        draw.line(points, fill=COLORS[mode], width=5, joint="curve")
        for row, (x, y) in zip(rows.itertuples(), points):
            draw.line((x, map_y(row.mean + row.std), x, map_y(max(0, row.mean - row.std))), fill=COLORS[mode], width=2)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COLORS[mode])
            text_center(draw, (x, bottom + 28), str(row.block_size), SMALL_FONT, MUTED)
        text_center(draw, ((left + right) / 2, 805), "Threads per block", SMALL_FONT)
    image.save(path, optimize=True)


def draw_cell_width_chart(data, path):
    image, draw = base_canvas("Cell width and neighbor-cell cost", "Simulation only, 20,000 boids, 128 threads per block, mean of three trials")
    left, top, right, bottom = 180, 190, 1500, 745
    ticks, maximum_y = nice_linear_ticks(data["mean"].max() * 1.15)
    def map_y(value):
        return bottom - value / maximum_y * (bottom - top)
    for tick in ticks:
        y = map_y(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 15, y), f"{int(tick):,}", fill=MUTED, font=TICK_FONT, anchor="rm")
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    fills = {1.0: "#ED9B40", 2.0: "#4776E6"}
    for center, mode in zip([560, 1120], ["scattered", "coherent"]):
        rows = data[data["mode"] == mode]
        for offset, row in zip([-90, 90], rows.itertuples()):
            x, y = center + offset, map_y(row.mean)
            draw.rectangle((x - 70, y, x + 70, bottom), fill=fills[row.cell_width_multiplier])
            draw.line((x, map_y(row.mean + row.std), x, map_y(row.mean - row.std)), fill=INK, width=3)
            text_center(draw, (x, y - 25), f"{row.mean:,.0f}", SMALL_FONT)
        text_center(draw, (center, bottom + 40), LABELS[mode], AXIS_FONT)
    for index, (label, color) in enumerate([("27 cells (1x width)", fills[1.0]), ("8 cells (2x width)", fills[2.0])]):
        x = 900 + index * 320
        draw.rectangle((x, 145, x + 28, 170), fill=color)
        draw.text((x + 40, 157), label, fill=INK, font=SMALL_FONT, anchor="lm")
    image.save(path, optimize=True)


def create_boid_media():
    frame_directory = ROOT / "profiling" / "capture_frames"
    frames = sorted(frame_directory.glob("frame_*.ppm"), key=lambda path: int(path.stem.split("_")[-1]))
    if not frames:
        raise RuntimeError("No capture frames were found.")
    converted = [Image.open(path).convert("RGB") for path in frames]
    converted[-1].save(IMAGES / "boids.png", optimize=True)
    converted[0].save(IMAGES / "boids.gif", save_all=True, append_images=converted[1:], duration=80, loop=0, optimize=True)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RAW_DATA)
    boid_data = grouped(data[data["experiment"] == "boid_count"], ["visualization", "mode", "boids"])
    draw_line_chart(boid_data[boid_data["visualization"] == "off"], IMAGES / "profile_boid_count_off.png", "Boid count versus performance", "Simulation only, Release build, mean of three trials; logarithmic FPS axis", True)
    draw_line_chart(boid_data[boid_data["visualization"] == "on"], IMAGES / "profile_boid_count_on.png", "Boid count versus performance", "Visualization enabled, Release build, mean of three trials")
    block_data = grouped(data[data["experiment"] == "block_size"], ["mode", "block_size"])
    draw_block_chart(block_data, IMAGES / "profile_block_size.png")
    cell_data = grouped(data[data["experiment"] == "cell_width"], ["mode", "cell_width_multiplier"])
    draw_cell_width_chart(cell_data, IMAGES / "profile_cell_width.png")
    create_boid_media()
    summary = grouped(data, ["experiment", "visualization", "mode", "boids", "block_size", "cell_width_multiplier"])
    summary.to_csv(RESULTS / "profile_summary.csv", index=False)


if __name__ == "__main__":
    main()
