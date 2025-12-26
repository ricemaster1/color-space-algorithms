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


def _srgb_to_xyz(channel: float) -> float:
    channel /= 255.0
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r_lin = _srgb_to_xyz(rgb[0])
    g_lin = _srgb_to_xyz(rgb[1])
    b_lin = _srgb_to_xyz(rgb[2])

    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t):
        epsilon = 0.008856
        kappa = 903.3
        if t > epsilon:
            return t ** (1 / 3)
        return (kappa * t + 16) / 116

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return (L, a, b)


def _weighted_lab_distance(a: tuple[float, float, float], b: tuple[float, float, float], weights: tuple[float, float, float]) -> float:
    dL = (a[0] - b[0]) * weights[0]
    da = (a[1] - b[1]) * weights[1]
    db = (a[2] - b[2]) * weights[2]
    return math.sqrt(dL * dL + da * da + db * db)


def apply_rgb_to_lab(img: Image.Image, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> list[list[str]]:
    width, height = img.size
    palette_lab = {name: _rgb_to_lab(rgb) for name, rgb in ARMLITE_RGB.items()}
    grid: list[list[str]] = []

    for y in range(height):
        row: list[str] = []
        for x in range(width):
            lab_src = _rgb_to_lab(img.getpixel((x, y)))
            best_name = 'black'
            best_distance = float('inf')
            for name, lab in palette_lab.items():
                dist = _weighted_lab_distance(lab_src, lab, weights)
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
    grid = apply_rgb_to_lab(img, weights=weights)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RGB→Lab workflow renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', default='converted.s', help='Output assembly file path (default: converted.s)')
    parser.add_argument('--weights', default='1,1,1', help='Comma-separated weights for L,a,b channels (default: 1,1,1)')
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

    output_path = args.output
    if output_path:
        output_path = os.path.expanduser(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'converted.s')
        else:
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                print(f'Output directory does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = 'converted.s'
    
    process_image(args.image, output_path, weights)
