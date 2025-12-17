from __future__ import annotations

from PIL import Image
import argparse
import math
import os
import random
import sys
from pathlib import Path

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB, closest_color


class VoronoiPalette:
    def __init__(self, k: int = 16, seed: int = 42, iterations: int = 10) -> None:
        self.k = k
        self.random = random.Random(seed)
        self.iterations = iterations

    @staticmethod
    def _dist_sq(a: tuple[int, int, int], b: tuple[float, float, float]) -> float:
        dr = a[0] - b[0]
        dg = a[1] - b[1]
        db = a[2] - b[2]
        return dr * dr + dg * dg + db * db

    def _init_sites(self, pixels: list[tuple[int, int, int]]) -> list[tuple[float, float, float]]:
        sites: list[tuple[float, float, float]] = []
        # KMeans++ style seeding for spread-out initial sites.
        start = self.random.choice(pixels)
        sites.append(tuple(float(c) for c in start))
        distances = [0.0] * len(pixels)
        while len(sites) < self.k:
            total = 0.0
            for i, p in enumerate(pixels):
                d = min(self._dist_sq(p, site) for site in sites)
                distances[i] = d
                total += d
            if total == 0:
                sites.append(tuple(float(c) for c in pixels[self.random.randrange(len(pixels))]))
                continue
            target = self.random.random() * total
            cumulative = 0.0
            for idx, dist in enumerate(distances):
                cumulative += dist
                if cumulative >= target:
                    sites.append(tuple(float(c) for c in pixels[idx]))
                    break
        return sites

    def _assign(self, pixels: list[tuple[int, int, int]], sites: list[tuple[float, float, float]]) -> list[int]:
        assignments = []
        for p in pixels:
            best_idx = 0
            best_dist = math.inf
            for idx, site in enumerate(sites):
                dist = self._dist_sq(p, site)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            assignments.append(best_idx)
        return assignments

    def _update(self, pixels: list[tuple[int, int, int]], assignments: list[int]) -> list[tuple[float, float, float]]:
        accum = [(0.0, 0.0, 0.0, 0)] * self.k
        accum = [list(values) for values in accum]
        for pixel, idx in zip(pixels, assignments):
            group = accum[idx]
            group[0] += pixel[0]
            group[1] += pixel[1]
            group[2] += pixel[2]
            group[3] += 1
        sites: list[tuple[float, float, float]] = []
        for group in accum:
            count = group[3]
            if count == 0:
                # Re-seed empty clusters to keep them active.
                sites.append(tuple(float(c) for c in self.random.choice(pixels)))
            else:
                sites.append((group[0] / count, group[1] / count, group[2] / count))
        return sites

    def build(self, pixels: list[tuple[int, int, int]]) -> tuple[list[tuple[float, float, float]], list[int]]:
        if not pixels:
            default = [(0.0, 0.0, 0.0)]
            return default, [0]
        sites = self._init_sites(pixels)
        assignments = [0] * len(pixels)
        for _ in range(self.iterations):
            assignments = self._assign(pixels, sites)
            sites = self._update(pixels, assignments)
        return sites, assignments


def apply_voronoi_palette(img: Image.Image, clusters: int = 16) -> list[list[str]]:
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    max_clusters = min(clusters, len(ARMLITE_RGB))
    engine = VoronoiPalette(k=max_clusters)
    sites, assignments = engine.build(pixels)

    # Map each Voronoi site to the closest ARMLite color.
    site_to_color: list[str] = []
    for site in sites:
        rgb_site = (int(round(site[0])), int(round(site[1])), int(round(site[2])))
        site_to_color.append(closest_color(rgb_site))

    grid: list[list[str]] = []
    idx = 0
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            cluster_idx = assignments[idx]
            idx += 1
            row.append(site_to_color[cluster_idx])
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
    grid = apply_voronoi_palette(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Voronoi palette renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
