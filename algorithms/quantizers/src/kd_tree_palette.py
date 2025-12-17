from __future__ import annotations

from PIL import Image
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB


def _dist_sq(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    dr = a[0] - b[0]
    dg = a[1] - b[1]
    db = a[2] - b[2]
    return dr * dr + dg * dg + db * db


@dataclass
class KDNode:
    point: tuple[int, int, int]
    index: int
    axis: int
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None


class KDTreePalette:
    def __init__(self) -> None:
        self.entries = list(ARMLITE_RGB.items())
        points = [(rgb, idx) for idx, (_, rgb) in enumerate(self.entries)]
        self.root = self._build(points, depth=0)

    def _build(self, points, depth: int) -> Optional[KDNode]:
        if not points:
            return None
        axis = depth % 3
        points.sort(key=lambda item: item[0][axis])
        median = len(points) // 2
        median_point, median_index = points[median]
        left = self._build(points[:median], depth + 1)
        right = self._build(points[median + 1 :], depth + 1)
        return KDNode(point=median_point, index=median_index, axis=axis, left=left, right=right)

    def nearest(self, target: tuple[int, int, int]) -> str:
        best_index = 0
        best_distance = float('inf')

        def _search(node: Optional[KDNode], depth: int = 0) -> None:
            nonlocal best_index, best_distance
            if node is None:
                return
            dist = _dist_sq(target, node.point)
            if dist < best_distance:
                best_distance = dist
                best_index = node.index

            axis = node.axis
            diff = target[axis] - node.point[axis]
            primary = node.left if diff <= 0 else node.right
            secondary = node.right if diff <= 0 else node.left

            _search(primary, depth + 1)
            if diff * diff < best_distance:
                _search(secondary, depth + 1)

        _search(self.root)
        return self.entries[best_index][0]


def apply_kd_tree_palette(img: Image.Image) -> list[list[str]]:
    width, height = img.size
    tree = KDTreePalette()
    grid: list[list[str]] = []
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            rgb = img.getpixel((x, y))
            row.append(tree.nearest(rgb))
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
    grid = apply_kd_tree_palette(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='kd-tree palette renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
