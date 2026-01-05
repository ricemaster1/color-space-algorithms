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

from lib import ARMLITE_RGB, closest_color


STUCKI_KERNEL = (
    [
        (1, 0, 8),
        (2, 0, 4),
        (-2, 1, 2),
        (-1, 1, 4),
        (0, 1, 8),
        (1, 1, 4),
        (2, 1, 2),
        (-2, 2, 1),
        (-1, 2, 2),
        (0, 2, 4),
        (1, 2, 2),
        (2, 2, 1),
    ],
    42,
)


def apply_stucki(img: Image.Image) -> list[list[str]]:
    width, height = img.size
    kernel, divisor = STUCKI_KERNEL
    workspace = [
        [list(map(float, img.getpixel((x, y)))) for x in range(width)]
        for y in range(height)
    ]
    grid: list[list[str]] = [["black" for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            old_r, old_g, old_b = workspace[y][x]
            name = closest_color((int(round(old_r)), int(round(old_g)), int(round(old_b))))
            new_r, new_g, new_b = ARMLITE_RGB[name]
            grid[y][x] = name

            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b

            for dx, dy, weight in kernel:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    target = workspace[ny][nx]
                    factor = weight / divisor
                    target[0] = max(0.0, min(255.0, target[0] + err_r * factor))
                    target[1] = max(0.0, min(255.0, target[1] + err_g * factor))
                    target[2] = max(0.0, min(255.0, target[2] + err_b * factor))

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


def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_stucki(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stucki renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', default='converted.s', help='Output assembly file path (default: converted.s)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
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
    process_image(args.image, output_path)
