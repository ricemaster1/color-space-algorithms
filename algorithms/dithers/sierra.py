from __future__ import annotations

from PIL import Image
import argparse
import os
import sys
from typing import List

from armlite import ARMLITE_RGB, closest_color


KERNELS: dict[str, tuple[List[tuple[int, int, int]], int]] = {
    'sierra3': (
        [
            (1, 0, 5),
            (2, 0, 3),
            (-2, 1, 2),
            (-1, 1, 4),
            (0, 1, 5),
            (1, 1, 4),
            (2, 1, 2),
            (-1, 2, 2),
            (0, 2, 3),
            (1, 2, 2),
        ],
        32,
    ),
    'sierra2': (
        [
            (1, 0, 4),
            (2, 0, 3),
            (-1, 1, 1),
            (0, 1, 5),
            (1, 1, 3),
            (2, 1, 2),
        ],
        32,
    ),
    'sierra-lite': (
        [
            (1, 0, 2),
            (2, 0, 1),
            (-1, 1, 1),
            (0, 1, 1),
        ],
        4,
    ),
}


def apply_sierra(img: Image.Image, variant: str = 'sierra3') -> list[list[str]]:
    width, height = img.size
    variant_key = variant.lower()
    if variant_key not in KERNELS:
        raise ValueError(f"Unknown Sierra variant '{variant}'.")

    kernel, divisor = KERNELS[variant_key]
    workspace: List[List[List[float]]] = [
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


def process_image(image_path, output_path, variant: str):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_sierra(img, variant=variant)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sierra-family renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    parser.add_argument(
        '--variant',
        default='sierra3',
        choices=list(KERNELS.keys()),
        help='Choose Sierra variant (sierra3, sierra2, sierra-lite). Default: sierra3'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output, args.variant)
