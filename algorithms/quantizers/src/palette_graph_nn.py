from __future__ import annotations

from PIL import Image
import argparse
import bisect
import heapq
import os
import sys
from pathlib import Path

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB


class PaletteGraph:
    """Graph accelerator for nearest-neighbor lookups in the ARMLite palette."""

    def __init__(self, k_neighbors: int = 8) -> None:
        self.palette = list(ARMLITE_RGB.items())
        self.colors = [rgb for _, rgb in self.palette]
        self.adj = self._build_knn_graph(k_neighbors)
        self.luma_sorted, self.luma_values = self._build_luma_index()

    @staticmethod
    def _distance_sq(rgb: tuple[int, int, int], other: tuple[int, int, int]) -> int:
        dr = rgb[0] - other[0]
        dg = rgb[1] - other[1]
        db = rgb[2] - other[2]
        return dr * dr + dg * dg + db * db

    @staticmethod
    def _luma(rgb: tuple[int, int, int]) -> float:
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    def _build_knn_graph(self, k_neighbors: int) -> list[list[int]]:
        size = len(self.colors)
        adjacency: list[set[int]] = [set() for _ in range(size)]
        for idx, base in enumerate(self.colors):
            distances = []
            for other_idx, other in enumerate(self.colors):
                if idx == other_idx:
                    continue
                dist = self._distance_sq(base, other)
                distances.append((dist, other_idx))
            distances.sort(key=lambda item: item[0])
            for _, neighbor in distances[:k_neighbors]:
                adjacency[idx].add(neighbor)
                adjacency[neighbor].add(idx)
        return [sorted(neighbors) for neighbors in adjacency]

    def _build_luma_index(self) -> tuple[list[int], list[float]]:
        entries = sorted(
            ((self._luma(rgb), idx) for idx, rgb in enumerate(self.colors)),
            key=lambda item: item[0]
        )
        indices = [idx for _, idx in entries]
        values = [value for value, _ in entries]
        return indices, values

    def _luma_start(self, rgb: tuple[int, int, int]) -> int:
        value = self._luma(rgb)
        position = bisect.bisect_left(self.luma_values, value)
        if position <= 0:
            return self.luma_sorted[0]
        if position >= len(self.luma_sorted):
            return self.luma_sorted[-1]
        low_idx = self.luma_sorted[position - 1]
        high_idx = self.luma_sorted[position]
        low_diff = abs(self.luma_values[position - 1] - value)
        high_diff = abs(self.luma_values[position] - value)
        return low_idx if low_diff <= high_diff else high_idx

    def nearest_index(self, rgb: tuple[int, int, int], seeds: list[int], budget: int = 48) -> int:
        best_idx = None
        best_dist = float('inf')
        visited: set[int] = set()
        heap: list[tuple[float, int]] = []

        unique_seeds = []
        seen_seed = set()
        for seed in seeds:
            if seed is None or seed in seen_seed:
                continue
            seen_seed.add(seed)
            unique_seeds.append(seed)
        luma_seed = self._luma_start(rgb)
        if luma_seed not in seen_seed:
            unique_seeds.append(luma_seed)

        for seed in unique_seeds:
            if seed in visited:
                continue
            visited.add(seed)
            dist = self._distance_sq(rgb, self.colors[seed])
            if dist < best_dist:
                best_dist = dist
                best_idx = seed
            heapq.heappush(heap, (dist, seed))

        expansions = 0
        while heap and expansions < budget:
            _, current = heapq.heappop(heap)
            expansions += 1
            for neighbor in self.adj[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dist = self._distance_sq(rgb, self.colors[neighbor])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = neighbor
                heapq.heappush(heap, (dist, neighbor))

        if best_idx is None:
            best_idx = 0
            best_dist = self._distance_sq(rgb, self.colors[0])

        if expansions >= budget:
            # Fallback to exact search when the frontier limit triggers.
            for idx, candidate in enumerate(self.colors):
                dist = self._distance_sq(rgb, candidate)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

        return best_idx

    def nearest_name(self, rgb: tuple[int, int, int], seeds: list[int]) -> tuple[str, int]:
        idx = self.nearest_index(rgb, seeds)
        return self.palette[idx][0], idx


def apply_palette_graph_nn(img: Image.Image) -> list[list[str]]:
    """Quantize using graph-assisted nearest-neighbor search in palette space."""
    width, height = img.size
    graph = PaletteGraph()
    index_grid: list[list[int]] = []
    name_grid: list[list[str]] = []

    for y in range(height):
        name_row: list[str] = []
        index_row: list[int] = []
        for x in range(width):
            rgb = img.getpixel((x, y))
            seeds: list[int] = []
            if x > 0:
                seeds.append(index_row[x - 1])
            if y > 0:
                seeds.append(index_grid[y - 1][x])
            color_name, idx = graph.nearest_name(rgb, seeds)
            name_row.append(color_name)
            index_row.append(idx)
        name_grid.append(name_row)
        index_grid.append(index_row)
    return name_grid


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
    grid = apply_palette_graph_nn(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Palette graph nearest-neighbor renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
