from __future__ import annotations

from collections import Counter
from PIL import Image
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import closest_color


class OctreeNode:
    __slots__ = (
        'level',
        'is_leaf',
        'color_count',
        'red_sum',
        'green_sum',
        'blue_sum',
        'children',
        'next_reducible',
    )

    def __init__(self, level: int) -> None:
        self.level = level
        self.is_leaf = level == 7
        self.color_count = 0
        self.red_sum = 0
        self.green_sum = 0
        self.blue_sum = 0
        self.children: List[Optional['OctreeNode']] = [None] * 8
        self.next_reducible: Optional['OctreeNode'] = None

    def add_color(self, rgb: tuple[int, int, int], count: int) -> None:
        self.color_count += count
        self.red_sum += rgb[0] * count
        self.green_sum += rgb[1] * count
        self.blue_sum += rgb[2] * count

    def average_color(self) -> tuple[int, int, int]:
        if self.color_count == 0:
            return (0, 0, 0)
        return (
            int(round(self.red_sum / self.color_count)),
            int(round(self.green_sum / self.color_count)),
            int(round(self.blue_sum / self.color_count)),
        )


class OctreeQuantizer:
    def __init__(self, max_colors: int = 16) -> None:
        self.max_colors = max(1, max_colors)
        self.leaf_count = 0
        self.root = OctreeNode(level=0)
        self.root.is_leaf = False
        self.reducible: List[Optional[OctreeNode]] = [None] * 8

    def _child_index(self, rgb: tuple[int, int, int], level: int) -> int:
        shift = 7 - level
        r = (rgb[0] >> shift) & 1
        g = (rgb[1] >> shift) & 1
        b = (rgb[2] >> shift) & 1
        return (r << 2) | (g << 1) | b

    def add_color(self, rgb: tuple[int, int, int]) -> None:
        self._add_color(rgb, 1)

    def _add_color(self, rgb: tuple[int, int, int], count: int) -> None:
        node = self.root
        for level in range(8):
            node.add_color(rgb, count)
            if node.is_leaf:
                break
            index = self._child_index(rgb, level)
            child = node.children[index]
            if child is None:
                child = OctreeNode(level + 1)
                node.children[index] = child
                if not child.is_leaf:
                    child.next_reducible = self.reducible[child.level]
                    self.reducible[child.level] = child
            node.is_leaf = False
            node = child
        if node.is_leaf and node.color_count == count:
            self.leaf_count += 1
        while self.leaf_count > self.max_colors:
            self._reduce()

    def _reduce(self) -> None:
        level = len(self.reducible) - 1
        while level >= 0 and self.reducible[level] is None:
            level -= 1
        if level < 0:
            return
        node = self.reducible[level]
        if node is None:
            return
        self.reducible[level] = node.next_reducible
        red_sum = green_sum = blue_sum = 0
        count = 0
        for idx, child in enumerate(node.children):
            if child is None:
                continue
            red_sum += child.red_sum
            green_sum += child.green_sum
            blue_sum += child.blue_sum
            count += child.color_count
            if child.is_leaf:
                self.leaf_count -= 1
            node.children[idx] = None
        node.is_leaf = True
        node.red_sum += red_sum
        node.green_sum += green_sum
        node.blue_sum += blue_sum
        node.color_count += count
        node.next_reducible = None
        if node.is_leaf:
            self.leaf_count += 1

    def _collect_palette(self, node: Optional[OctreeNode], palette: List[tuple[int, int, int]]) -> None:
        if node is None:
            return
        if node.is_leaf or node.level == 7:
            palette.append(node.average_color())
            return
        for child in node.children:
            if child is not None:
                self._collect_palette(child, palette)

    def palette(self) -> List[tuple[int, int, int]]:
        palette: List[tuple[int, int, int]] = []
        self._collect_palette(self.root, palette)
        return palette[: self.max_colors]

    def map_pixels(self, pixels: List[tuple[int, int, int]]) -> List[str]:
        palette = self.palette()
        if not palette:
            return ['black'] * len(pixels)
        mapped_names = [closest_color(color) for color in palette]
        name_cache: dict[tuple[int, int, int], str] = {}
        for rgb in set(pixels):
            best_idx = 0
            best_dist = float('inf')
            for p_idx, color in enumerate(palette):
                dist = ((rgb[0] - color[0]) ** 2 + (rgb[1] - color[1]) ** 2 + (rgb[2] - color[2]) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = p_idx
            name_cache[rgb] = mapped_names[best_idx]
        return [name_cache[rgb] for rgb in pixels]


def apply_octree(img: Image.Image, max_colors: int = 16) -> list[list[str]]:
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    quantizer = OctreeQuantizer(max_colors=max_colors)
    counts = Counter(pixels)
    for rgb, freq in counts.items():
        quantizer._add_color(rgb, freq)
    mapped = quantizer.map_pixels(pixels)

    grid: List[List[str]] = []
    idx = 0
    for _ in range(height):
        row: List[str] = []
        for _ in range(width):
            row.append(mapped[idx])
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
    grid = apply_octree(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Octree renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
