from __future__ import annotations

from PIL import Image
import argparse
import colorsys
import math
import os
import sys
from typing import Callable, Iterable, Tuple

from armlite import ARMLITE_RGB, closest_color


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*(channel / 255.0 for channel in rgb))


def _rgb_to_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*(channel / 255.0 for channel in rgb))
    return (h, s, l)


def _weighted_distance(weight_h: float, weight_s: float, weight_v: float):
    def metric(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dh = min(abs(a[0] - b[0]), 1.0 - abs(a[0] - b[0]))
        ds = abs(a[1] - b[1])
        dv = abs(a[2] - b[2])
        return math.sqrt((weight_h * dh) ** 2 + (weight_s * ds) ** 2 + (weight_v * dv) ** 2)

    return metric


def _palette_space(palette: dict[str, tuple[int, int, int]], transform: Callable[[tuple[int, int, int]], tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
    return {name: transform(rgb) for name, rgb in palette.items()}


def apply_rgb_to_hsv_hsl(
    img: Image.Image,
    space: str = 'hsv',
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[list[str]]:
    width, height = img.size
    transform = _rgb_to_hsv if space == 'hsv' else _rgb_to_hsl
    metric = _weighted_distance(*weights)

    palette_space = _palette_space(ARMLITE_RGB, transform)
    grid: list[list[str]] = []

    for y in range(height):
        row: list[str] = []
        for x in range(width):
            rgb = img.getpixel((x, y))
            converted = transform(rgb)
            best_name = 'black'
            best_distance = float('inf')
            for name, target in palette_space.items():
                dist = metric(converted, target)
                if dist < best_distance:
                    best_distance = dist
                    best_name = name
            row.append(best_name)
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


def process_image(image_path, output_path, space: str, weights: tuple[float, float, float]):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_rgb_to_hsv_hsl(img, space=space, weights=weights)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RGB→HSV/HSL workflow renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    parser.add_argument('--space', choices=['hsv', 'hsl'], default='hsv', help='Color space to use for matching (default: hsv)')
    parser.add_argument('--weights', default='1,1,1', help='Comma-separated weights for hue,sat,value(lightness) matching (default: 1,1,1)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    try:
        weights = tuple(float(w.strip()) for w in args.weights.split(','))
    except ValueError:
        print('Invalid weights. Use comma-separated numbers, e.g. 1,1,0.5')
        sys.exit(1)
    if len(weights) != 3:
        print('Weights must contain exactly three values.')
        sys.exit(1)

    process_image(args.image, args.output, args.space, weights)
