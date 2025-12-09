from __future__ import annotations

from PIL import Image
import argparse
import os
import sys

from armlite import ARMLITE_RGB, closest_color


def apply_atkinson(img: Image.Image) -> list[list[str]]:
    width, height = img.size
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            old_r, old_g, old_b = pixels[x, y]
            name = closest_color((old_r, old_g, old_b))
            new_r, new_g, new_b = ARMLITE_RGB[name]
            pixels[x, y] = (new_r, new_g, new_b)

            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b

            for dx, dy in ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    pr, pg, pb = pixels[nx, ny]
                    pixels[nx, ny] = (
                        int(max(0, min(255, pr + err_r // 8))),
                        int(max(0, min(255, pg + err_g // 8))),
                        int(max(0, min(255, pb + err_b // 8))),
                    )

    grid: list[list[str]] = []
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            row.append(closest_color(pixels[x, y]))
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


def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_atkinson(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Atkinson renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
