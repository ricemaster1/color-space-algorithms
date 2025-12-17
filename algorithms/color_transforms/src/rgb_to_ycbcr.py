from __future__ import annotations

from PIL import Image
import argparse
import math
import os
import sys
from pathlib import Path

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB


def _rgb_to_ycbcr(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = rgb
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return (y, cb, cr)


def _weighted_ycbcr_distance(a: tuple[float, float, float], b: tuple[float, float, float], weights: tuple[float, float, float]) -> float:
    dy = (a[0] - b[0]) * weights[0]
    dcb = (a[1] - b[1]) * weights[1]
    dcr = (a[2] - b[2]) * weights[2]
    return math.sqrt(dy * dy + dcb * dcb + dcr * dcr)


def apply_rgb_to_ycbcr(img: Image.Image, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> list[list[str]]:
    width, height = img.size
    palette_ycbcr = {name: _rgb_to_ycbcr(rgb) for name, rgb in ARMLITE_RGB.items()}
    grid: list[list[str]] = []

    for y in range(height):
        row: list[str] = []
        for x in range(width):
            ycbcr_src = _rgb_to_ycbcr(img.getpixel((x, y)))
            best_name = 'black'
            best_distance = float('inf')
            for name, ycbcr in palette_ycbcr.items():
                dist = _weighted_ycbcr_distance(ycbcr_src, ycbcr, weights)
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


def process_image(image_path, output_path, weights: tuple[float, float, float]):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_rgb_to_ycbcr(img, weights=weights)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RGB→YCbCr workflow renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    parser.add_argument('--weights', default='1,1,1', help='Comma-separated weights for Y,Cb,Cr channels (default: 1,1,1)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    try:
        weights = tuple(float(w.strip()) for w in args.weights.split(','))
    except ValueError:
        print('Invalid weights. Use comma-separated numbers, e.g. 1,1,0.8')
        sys.exit(1)
    if len(weights) != 3:
        print('Weights must contain exactly three values.')
        sys.exit(1)

    process_image(args.image, args.output, weights)
