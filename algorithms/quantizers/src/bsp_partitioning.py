from __future__ import annotations

from PIL import Image
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

from armlite import closest_color


def _channel_bounds(pixels: list[tuple[int, int, int]], indices: list[int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    r_min = g_min = b_min = 255
    r_max = g_max = b_max = 0
    for idx in indices:
        r, g, b = pixels[idx]
        if r < r_min:
            r_min = r
        if r > r_max:
            r_max = r
        if g < g_min:
            g_min = g
        if g > g_max:
            g_max = g
        if b < b_min:
            b_min = b
        if b > b_max:
            b_max = b
    return (r_min, r_max), (g_min, g_max), (b_min, b_max)


@dataclass
class BspNode:
    indices: list[int]
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    def range_size(self) -> int:
        (r_min, r_max), (g_min, g_max), (b_min, b_max) = self.bounds
        return max(r_max - r_min, g_max - g_min, b_max - b_min)


class BspPartitioner:
    def __init__(self, pixels: list[tuple[int, int, int]], target_leaves: int = 16) -> None:
        self.pixels = pixels
        self.target_leaves = max(1, target_leaves)

    def _split_axis(self, node: BspNode) -> int:
        ranges = [node.bounds[i][1] - node.bounds[i][0] for i in range(3)]
        axis = max(range(3), key=lambda i: ranges[i])
        return axis

    def partition(self) -> list[list[int]]:
        if not self.pixels:
            return [[]]

        all_indices = list(range(len(self.pixels)))
        initial_bounds = _channel_bounds(self.pixels, all_indices)
        root = BspNode(indices=all_indices, bounds=initial_bounds)
        nodes: List[BspNode] = [root]
        target = min(self.target_leaves, len(self.pixels))

        while len(nodes) < target:
            # Always split the region with the largest channel span to mimic BSP partitioning.
            nodes.sort(key=lambda n: n.range_size(), reverse=True)
            node = nodes.pop(0)
            if len(node.indices) <= 1:
                nodes.append(node)
                break
            axis = self._split_axis(node)
            node.indices.sort(key=lambda idx: self.pixels[idx][axis])
            mid = len(node.indices) // 2
            if mid == 0 or mid == len(node.indices):
                nodes.append(node)
                break
            left_indices = node.indices[:mid]
            right_indices = node.indices[mid:]
            left_bounds = _channel_bounds(self.pixels, left_indices)
            right_bounds = _channel_bounds(self.pixels, right_indices)
            nodes.extend([
                BspNode(indices=left_indices, bounds=left_bounds),
                BspNode(indices=right_indices, bounds=right_bounds),
            ])
            if len(nodes) >= target:
                break
        return [node.indices for node in nodes]


def apply_bsp_partitioning(img: Image.Image, clusters: int = 16) -> list[list[str]]:
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    partitioner = BspPartitioner(pixels, target_leaves=clusters)
    leaves = partitioner.partition()

    leaf_color: list[str] = ['black'] * len(leaves)
    for leaf_idx, indices in enumerate(leaves):
        if not indices:
            continue
        total_r = total_g = total_b = 0
        for idx in indices:
            r, g, b = pixels[idx]
            total_r += r
            total_g += g
            total_b += b
        avg = (
            int(round(total_r / len(indices))),
            int(round(total_g / len(indices))),
            int(round(total_b / len(indices)))
        )
        leaf_color[leaf_idx] = closest_color(avg)

    grid: list[list[str]] = []
    flat_assignments = ['black'] * len(pixels)
    for leaf_idx, indices in enumerate(leaves):
        color = leaf_color[leaf_idx]
        for idx in indices:
            flat_assignments[idx] = color

    idx = 0
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            row.append(flat_assignments[idx])
            idx += 1
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
    grid = apply_bsp_partitioning(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BSP palette renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
