from __future__ import annotations

from PIL import Image
import argparse
import os
import sys
from pathlib import Path

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB
from distance_ciede2000 import rgb_to_lab, delta_e_ciede2000
from distance_cie94 import delta_e_cie94
from distance_cie76 import delta_e_cie76


def _delta(metric: str, lab_src, lab_dst, application: str) -> float:
    metric = metric.lower()
    if metric in {'ciede2000', '2000'}:
        return delta_e_ciede2000(lab_src, lab_dst)
    if metric in {'cie94', '94'}:
        return delta_e_cie94(lab_src, lab_dst, application='textiles' if application == 'textiles' else 'graphic_arts')
    if metric in {'cie76', '76'}:
        return delta_e_cie76(lab_src, lab_dst)
    if metric == 'hybrid':
        de2000 = delta_e_ciede2000(lab_src, lab_dst)
        de94 = delta_e_cie94(lab_src, lab_dst, application='textiles' if application == 'textiles' else 'graphic_arts')
        de76 = delta_e_cie76(lab_src, lab_dst)
        return 0.6 * de2000 + 0.3 * de94 + 0.1 * de76
    raise ValueError(f"Unknown metric '{metric}'")


def apply_distance_delta_e(img: Image.Image, metric: str = 'ciede2000', application: str = 'graphic_arts') -> list[list[str]]:
    width, height = img.size
    palette_lab = {name: rgb_to_lab(rgb) for name, rgb in ARMLITE_RGB.items()}
    grid: list[list[str]] = []
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            lab_src = rgb_to_lab(img.getpixel((x, y)))
            best_name = 'black'
            best_delta = float('inf')
            for name, lab_pal in palette_lab.items():
                delta = _delta(metric, lab_src, lab_pal, application)
                if delta < best_delta:
                    best_delta = delta
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


def process_image(image_path, output_path, metric: str, application: str):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_distance_delta_e(img, metric=metric, application=application)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Delta E renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    parser.add_argument(
        '-m', '--metric',
        default='ciede2000',
        choices=['ciede2000', 'cie94', 'cie76', '2000', '94', '76', 'hybrid'],
        help='Delta E metric to use (default: ciede2000)'
    )
    parser.add_argument(
        '--cie94-application',
        default='graphic_arts',
        choices=['graphic_arts', 'textiles'],
        help='Application weighting for CIE94 (used when metric is cie94 or hybrid)'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output, args.metric, args.cie94_application)
