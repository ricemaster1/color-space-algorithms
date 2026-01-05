from __future__ import annotations

from PIL import Image
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import closest_color


@dataclass
class ColorBox:
    indices: List[int]
    r_min: int
    r_max: int
    g_min: int
    g_max: int
    b_min: int
    b_max: int

    def range(self) -> tuple[int, int, int]:
        return (self.r_max - self.r_min, self.g_max - self.g_min, self.b_max - self.b_min)


def _compute_bounds(pixels: List[tuple[int, int, int]], indices: List[int]) -> ColorBox:
    r_vals = [pixels[idx][0] for idx in indices]
    g_vals = [pixels[idx][1] for idx in indices]
    b_vals = [pixels[idx][2] for idx in indices]
    return ColorBox(
        indices=indices,
        r_min=min(r_vals),
        r_max=max(r_vals),
        g_min=min(g_vals),
        g_max=max(g_vals),
        b_min=min(b_vals),
        b_max=max(b_vals),
    )


def _split_box(pixels: List[tuple[int, int, int]], box: ColorBox) -> list[ColorBox]:
    if len(box.indices) <= 1:
        return [box]
    ranges = box.range()
    axis = ranges.index(max(ranges))
    if ranges[axis] == 0:
        return [box]
    sorted_indices = sorted(box.indices, key=lambda idx: pixels[idx][axis])
    mid = len(sorted_indices) // 2
    if mid == 0 or mid == len(sorted_indices):
        return [box]
    left_indices = sorted_indices[:mid]
    right_indices = sorted_indices[mid:]
    return [_compute_bounds(pixels, left_indices), _compute_bounds(pixels, right_indices)]


def _average_color(pixels: List[tuple[int, int, int]], indices: List[int]) -> tuple[int, int, int]:
    total_r = total_g = total_b = 0
    for idx in indices:
        r, g, b = pixels[idx]
        total_r += r
        total_g += g
        total_b += b
    count = max(1, len(indices))
    return (
        int(round(total_r / count)),
        int(round(total_g / count)),
        int(round(total_b / count)),
    )


def apply_median_cut(img: Image.Image, max_colors: int = 16) -> list[list[str]]:
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    if not pixels:
        return []
    max_colors = max(1, min(max_colors, len(pixels)))
    initial_indices = list(range(len(pixels)))
    boxes: list[ColorBox] = [_compute_bounds(pixels, initial_indices)]

    while len(boxes) < max_colors:
        boxes.sort(key=lambda box: max(box.range()), reverse=True)
        box = boxes.pop(0)
        splits = _split_box(pixels, box)
        if len(splits) == 1:
            boxes.append(box)
            if all(len(candidate.indices) <= 1 or max(candidate.range()) == 0 for candidate in boxes):
                break
            continue
        boxes.extend(splits)

    palette = [_average_color(pixels, box.indices) for box in boxes]
    palette_names = [closest_color(color) for color in palette]

    assignments = ['black'] * len(pixels)
    for palette_idx, box in enumerate(boxes):
        name = palette_names[palette_idx]
        for idx in box.indices:
            assignments[idx] = name

    grid: list[list[str]] = []
    iter_idx = 0
    for _ in range(height):
        row: list[str] = []
        for _ in range(width):
            row.append(assignments[iter_idx])
            iter_idx += 1
        grid.append(row)
    return grid


def generate_assembly(color_grid, output_path):
    height = len(color_grid)
    width = len(color_grid[0]) if height else 0
    lines = [
        '; === Fullscreen Sprite ===',
        '    MOV R0, #2',
        '    STR R0, .Resolution',
        '    MOV R1, #.PixelScreen',
        '    MOV R6, #512 ; row stride (128 * 4)'
    ]
    for y in range(height):
        for x in range(width):
            offset = ((y * width) + x) * 4
            lines.append(f'    MOV R5, #{offset}')
            lines.append('    ADD R4, R1, R5')
            color_name = color_grid[y][x]
            lines.append(f'    MOV R0, #.{color_name}')
            lines.append(f'    STR R0, [R4]   ; Pixel ({x},{y})')
    lines.append('    HALT')
    with open(output_path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f"Assembly sprite file written to {output_path}")


def process_image(image_path, output_path, colors: int):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_median_cut(img, max_colors=colors)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Median Cut renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', help='Output assembly file path (default: converted.s)')
    parser.add_argument('-c', '--colors', type=int, default=16, help='Number of median cut palette clusters (default: 16)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)
    
    output_path = args.output
    if output_path:
        os.path.expanduser(output_path)
        if os.path.isdir(output_path):
            os.path.join(output_path, 'converted.s')
        else:
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                print(f'Output path does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = 'converted.s'

    process_image(args.image, args.output, args.colors)
