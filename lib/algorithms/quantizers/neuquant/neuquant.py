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

from lib import closest_color


class NeuQuantNetwork:
    def __init__(self, pixels: list[tuple[int, int, int]], palette_size: int = 128, samplefac: int = 10, seed: int = 7) -> None:
        self.random = random.Random(seed)
        self.pixels = pixels
        self.samplefac = max(1, samplefac)
        self.palette_size = max(1, min(palette_size, 256, len(pixels)))
        self.network = self._init_network()

    def _init_network(self) -> list[list[float]]:
        if len(self.pixels) >= self.palette_size:
            sample = self.random.sample(self.pixels, self.palette_size)
        else:
            sample = [self.random.choice(self.pixels) for _ in range(self.palette_size)]
        return [[float(r), float(g), float(b)] for r, g, b in sample]

    @staticmethod
    def _dist_sq(a: tuple[int, int, int], b: list[float]) -> float:
        dr = a[0] - b[0]
        dg = a[1] - b[1]
        db = a[2] - b[2]
        return dr * dr + dg * dg + db * db

    def _training_samples(self) -> list[tuple[int, int, int]]:
        stride = max(1, len(self.pixels) // (self.palette_size * self.samplefac))
        return self.pixels[::stride]

    def train(self, epochs: int = 6) -> None:
        samples = self._training_samples()
        if not samples:
            return
        total_steps = max(1, epochs * len(samples))
        start_radius = max(1.0, self.palette_size / 6.0)
        start_alpha = 0.5

        for step in range(total_steps):
            pixel = samples[step % len(samples)]
            frac = step / total_steps
            alpha = start_alpha * (1.0 - frac)
            radius = start_radius * (1.0 - frac)
            radius_sq = max(1e-5, radius * radius)

            # Find best matching unit.
            best_idx = 0
            best_dist = math.inf
            for idx, node in enumerate(self.network):
                dist = self._dist_sq(pixel, node)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            # Update BMU and its neighbors along the 1-D chain.
            span = int(max(1, radius))
            for idx in range(max(0, best_idx - span), min(self.palette_size, best_idx + span + 1)):
                node = self.network[idx]
                dist_idx = idx - best_idx
                influence = math.exp(-(dist_idx * dist_idx) / (2.0 * radius_sq))
                rate = alpha * influence
                node[0] += rate * (pixel[0] - node[0])
                node[1] += rate * (pixel[1] - node[1])
                node[2] += rate * (pixel[2] - node[2])

    def palette(self) -> list[tuple[int, int, int]]:
        return [
            (
                int(max(0, min(255, round(node[0])))),
                int(max(0, min(255, round(node[1])))),
                int(max(0, min(255, round(node[2]))))
            )
            for node in self.network
        ]


def apply_neuquant(img: Image.Image, netsize: int = 128, samplefac: int = 10) -> list[list[str]]:
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    trainer = NeuQuantNetwork(pixels, palette_size=netsize, samplefac=samplefac)
    trainer.train()
    palette = trainer.palette()
    if not palette:
        return [['black'] * width for _ in range(height)]
    mapped_names = [closest_color(color) for color in palette]

    def map_pixel(rgb: tuple[int, int, int]) -> str:
        best = palette[0]
        best_dist = (rgb[0] - best[0]) ** 2 + (rgb[1] - best[1]) ** 2 + (rgb[2] - best[2]) ** 2
        best_idx = 0
        for idx, color in enumerate(palette[1:], start=1):
            dist = (rgb[0] - color[0]) ** 2 + (rgb[1] - color[1]) ** 2 + (rgb[2] - color[2]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best = color
        return mapped_names[best_idx]

    grid: list[list[str]] = []
    idx = 0
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            row.append(map_pixel(pixels[idx]))
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
    grid = apply_neuquant(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NeuQuant renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', help='Output assembly file path (default: converted.s)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    output_path = args.output
    if output_path:
        os.path.expanduser(output_path)
        if os.path.isdir(output_path):
            os.path.join(output_path, 'converted.s')
        else:
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                print(f'Output path does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = 'converted.s'

    process_image(args.image, args.output)
