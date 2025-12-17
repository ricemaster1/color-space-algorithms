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

from lib import closest_color


_BINS = 33  # 32 quantization buckets + 1 guard so we can use 1-based indexing


def _idx(r: int, g: int, b: int) -> int:
    return (r * _BINS + g) * _BINS + b


class WuQuantizer:
    """Wu's fast color quantizer with cumulative-moment tables."""

    def __init__(self) -> None:
        total = _BINS ** 3
        self.weights = [0] * total
        self.moments_r = [0.0] * total
        self.moments_g = [0.0] * total
        self.moments_b = [0.0] * total
        self.moments = [0.0] * total

    def build_histogram(self, pixels: list[tuple[int, int, int]]) -> None:
        if not pixels:
            return
        max_index = _BINS - 1
        for r, g, b in pixels:
            ir = min(max_index, (r >> 3) + 1)
            ig = min(max_index, (g >> 3) + 1)
            ib = min(max_index, (b >> 3) + 1)
            index = _idx(ir, ig, ib)
            self.weights[index] += 1
            self.moments_r[index] += r
            self.moments_g[index] += g
            self.moments_b[index] += b
            self.moments[index] += r * r + g * g + b * b

    def compute_prefix_sums(self) -> None:
        for r in range(1, _BINS):
            area = [0] * _BINS
            area_r = [0.0] * _BINS
            area_g = [0.0] * _BINS
            area_b = [0.0] * _BINS
            area_m = [0.0] * _BINS
            for g in range(1, _BINS):
                line = 0
                line_r = 0.0
                line_g = 0.0
                line_b = 0.0
                line_m = 0.0
                for b in range(1, _BINS):
                    idx = _idx(r, g, b)
                    line += self.weights[idx]
                    line_r += self.moments_r[idx]
                    line_g += self.moments_g[idx]
                    line_b += self.moments_b[idx]
                    line_m += self.moments[idx]

                    area[b] += line
                    area_r[b] += line_r
                    area_g[b] += line_g
                    area_b[b] += line_b
                    area_m[b] += line_m

                    prev = _idx(r - 1, g, b)
                    self.weights[idx] = self.weights[prev] + area[b]
                    self.moments_r[idx] = self.moments_r[prev] + area_r[b]
                    self.moments_g[idx] = self.moments_g[prev] + area_g[b]
                    self.moments_b[idx] = self.moments_b[prev] + area_b[b]
                    self.moments[idx] = self.moments[prev] + area_m[b]

    def _volume(self, cube: tuple[int, int, int, int, int, int], data: list[float]) -> float:
        r0, r1, g0, g1, b0, b1 = cube
        return (
            data[_idx(r1, g1, b1)]
            - data[_idx(r1, g1, b0)]
            - data[_idx(r1, g0, b1)]
            - data[_idx(r0, g1, b1)]
            + data[_idx(r1, g0, b0)]
            + data[_idx(r0, g1, b0)]
            + data[_idx(r0, g0, b1)]
            - data[_idx(r0, g0, b0)]
        )

    def _variance(self, cube: tuple[int, int, int, int, int, int]) -> float:
        weight = self._volume(cube, self.weights)
        if weight <= 0:
            return 0.0
        r = self._volume(cube, self.moments_r)
        g = self._volume(cube, self.moments_g)
        b = self._volume(cube, self.moments_b)
        m = self._volume(cube, self.moments)
        return max(0.0, m - (r * r + g * g + b * b) / weight)

    def _maximize(self, cube: tuple[int, int, int, int, int, int], axis: int, totals: tuple[float, float, float, float]) -> tuple[int | None, float]:
        r0, r1, g0, g1, b0, b1 = cube
        total_r, total_g, total_b, total_w = totals

        lower = None
        upper = None
        if axis == 0:
            lower, upper = r0, r1
        elif axis == 1:
            lower, upper = g0, g1
        else:
            lower, upper = b0, b1

        best_value = 0.0
        best_cut: int | None = None
        for cut in range(lower + 1, upper):
            if axis == 0:
                cube_low = (r0, cut, g0, g1, b0, b1)
                cube_high = (cut, r1, g0, g1, b0, b1)
            elif axis == 1:
                cube_low = (r0, r1, g0, cut, b0, b1)
                cube_high = (r0, r1, cut, g1, b0, b1)
            else:
                cube_low = (r0, r1, g0, g1, b0, cut)
                cube_high = (r0, r1, g0, g1, cut, b1)

            weight_low = self._volume(cube_low, self.weights)
            weight_high = total_w - weight_low
            if weight_low <= 0 or weight_high <= 0:
                continue

            r_low = self._volume(cube_low, self.moments_r)
            g_low = self._volume(cube_low, self.moments_g)
            b_low = self._volume(cube_low, self.moments_b)

            r_high = total_r - r_low
            g_high = total_g - g_low
            b_high = total_b - b_low

            value = (
                (r_low * r_low + g_low * g_low + b_low * b_low) / weight_low
                + (r_high * r_high + g_high * g_high + b_high * b_high) / weight_high
            )
            if value > best_value:
                best_value = value
                best_cut = cut

        return best_cut, best_value

    def _split(self, cube: tuple[int, int, int, int, int, int]) -> tuple[tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int]] | None:
        total_r = self._volume(cube, self.moments_r)
        total_g = self._volume(cube, self.moments_g)
        total_b = self._volume(cube, self.moments_b)
        total_w = self._volume(cube, self.weights)

        if total_w <= 0:
            return None

        best_axis = None
        best_cut = None
        best_value = 0.0

        for axis in range(3):
            cut, value = self._maximize(cube, axis, (total_r, total_g, total_b, total_w))
            if cut is None:
                continue
            if best_axis is None or value > best_value:
                best_axis = axis
                best_cut = cut
                best_value = value

        if best_axis is None or best_cut is None:
            return None

        r0, r1, g0, g1, b0, b1 = cube
        if best_axis == 0:
            return (r0, best_cut, g0, g1, b0, b1), (best_cut, r1, g0, g1, b0, b1)
        if best_axis == 1:
            return (r0, r1, g0, best_cut, b0, b1), (r0, r1, best_cut, g1, b0, b1)
        return (r0, r1, g0, g1, b0, best_cut), (r0, r1, g0, g1, best_cut, b1)

    def _cube_color(self, cube: tuple[int, int, int, int, int, int]) -> tuple[int, int, int]:
        weight = self._volume(cube, self.weights)
        if weight <= 0:
            return (0, 0, 0)
        r = self._volume(cube, self.moments_r) / weight
        g = self._volume(cube, self.moments_g) / weight
        b = self._volume(cube, self.moments_b) / weight
        return (
            int(max(0, min(255, round(r)))),
            int(max(0, min(255, round(g)))),
            int(max(0, min(255, round(b)))),
        )

    def quantize(self, pixels: list[tuple[int, int, int]], max_colors: int = 16) -> list[tuple[int, int, int]]:
        if not pixels:
            return [(0, 0, 0)]

        self.build_histogram(pixels)
        self.compute_prefix_sums()

        cubes: list[tuple[int, int, int, int, int, int]] = [(0, _BINS - 1, 0, _BINS - 1, 0, _BINS - 1)]
        while len(cubes) < max_colors:
            cubes.sort(key=self._variance, reverse=True)
            cube = cubes.pop(0)
            result = self._split(cube)
            if result is None:
                cubes.append(cube)
                break
            cubes.extend(result)

        palette = [self._cube_color(cube) for cube in cubes if self._volume(cube, self.weights) > 0]
        return palette or [(0, 0, 0)]


def apply_wu_quantizer(img: Image.Image) -> list[list[str]]:
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    quantizer = WuQuantizer()
    palette = quantizer.quantize(pixels, max_colors=16)

    def nearest_palette(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        return min(palette, key=lambda p: (p[0] - rgb[0]) ** 2 + (p[1] - rgb[1]) ** 2 + (p[2] - rgb[2]) ** 2)

    grid: list[list[str]] = []
    for y in range(height):
        row: list[str] = []
        for x in range(width):
            mapped = nearest_palette(img.getpixel((x, y)))
            row.append(closest_color(mapped))
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
    grid = apply_wu_quantizer(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Wu's quantizer renderer for ARMLite sprites"
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
