from PIL import Image
import argparse
import os
import sys
import math
import random

from armlite import closest_color


def _train_som(pixels, grid_size=4, iterations=200, seed=42):
    random.seed(seed)
    if not pixels:
        return [(0, 0, 0) for _ in range(grid_size * grid_size)]

    nodes = [list(random.choice(pixels)) for _ in range(grid_size * grid_size)]
    positions = [(i % grid_size, i // grid_size) for i in range(len(nodes))]
    radius_start = grid_size / 2.0
    radius_end = 0.5
    lr_start = 0.5
    lr_end = 0.05

    for step in range(iterations):
        lr = lr_start + (lr_end - lr_start) * (step / max(1, iterations - 1))
        radius = radius_start + (radius_end - radius_start) * (step / max(1, iterations - 1))
        radius_sq = radius * radius
        px, py, pz = random.choice(pixels)

        # Find best matching unit
        best_idx = min(range(len(nodes)), key=lambda i: (nodes[i][0] - px) ** 2 + (nodes[i][1] - py) ** 2 + (nodes[i][2] - pz) ** 2)
        bx, by = positions[best_idx]

        for idx, node in enumerate(nodes):
            nx, ny = positions[idx]
            dist_sq = (nx - bx) ** 2 + (ny - by) ** 2
            if dist_sq > radius_sq:
                continue
            influence = math.exp(-dist_sq / max(radius_sq, 1e-8))
            node[0] += influence * lr * (px - node[0])
            node[1] += influence * lr * (py - node[1])
            node[2] += influence * lr * (pz - node[2])

    return [tuple(int(max(0, min(255, round(c)))) for c in node) for node in nodes]


def apply_som_quantizer(img):
    """Quantize image colours via a simple self-organizing map prior to palette mapping."""
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    som_nodes = _train_som(pixels)

    def nearest_node(rgb):
        return min(som_nodes, key=lambda n: (n[0] - rgb[0]) ** 2 + (n[1] - rgb[1]) ** 2 + (n[2] - rgb[2]) ** 2)

    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            node_rgb = nearest_node(img.getpixel((x, y)))
            row.append(closest_color(node_rgb))
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
    grid = apply_som_quantizer(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Self-organizing map renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
