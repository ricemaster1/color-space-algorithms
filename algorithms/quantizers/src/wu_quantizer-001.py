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


class WuQuantizer:
    """Minimal implementation of Wu's color quantization algorithm."""

    def __init__(self):
        self.size = 33
        total = self.size ** 3
        self.weights = [0] * total
        self.moments_r = [0.0] * total
        self.moments_g = [0.0] * total
        self.moments_b = [0.0] * total
        self.moments = [0.0] * total

    def _index(self, r, g, b):
        return (r * self.size + g) * self.size + b

    def _build_histogram(self, pixels):
        for r, g, b in pixels:
            ir = r >> 3
            ig = g >> 3
            ib = b >> 3
            idx = self._index(ir, ig, ib)
            self.weights[idx] += 1
            self.moments_r[idx] += r
            self.moments_g[idx] += g
            self.moments_b[idx] += b
            self.moments[idx] += r * r + g * g + b * b

    def _prefix_sums(self):
        for r in range(1, self.size):
            for g in range(1, self.size):
                for b in range(1, self.size):
                    idx = self._index(r, g, b)
                    w = self.weights
                    mr = self.moments_r
                    mg = self.moments_g
                    mb = self.moments_b
                    mm = self.moments
                    idx1 = self._index(r - 1, g, b)
                    idx2 = self._index(r, g - 1, b)
                    idx3 = self._index(r, g, b - 1)
                    idx4 = self._index(r - 1, g - 1, b)
                    idx5 = self._index(r - 1, g, b - 1)
                    idx6 = self._index(r, g - 1, b - 1)
                    idx7 = self._index(r - 1, g - 1, b - 1)
                    w[idx] += w[idx1] + w[idx2] + w[idx3] - w[idx4] - w[idx5] - w[idx6] + w[idx7]
                    mr[idx] += mr[idx1] + mr[idx2] + mr[idx3] - mr[idx4] - mr[idx5] - mr[idx6] + mr[idx7]
                    mg[idx] += mg[idx1] + mg[idx2] + mg[idx3] - mg[idx4] - mg[idx5] - mg[idx6] + mg[idx7]
                    mb[idx] += mb[idx1] + mb[idx2] + mb[idx3] - mb[idx4] - mb[idx5] - mb[idx6] + mb[idx7]
                    mm[idx] += mm[idx1] + mm[idx2] + mm[idx3] - mm[idx4] - mm[idx5] - mm[idx6] + mm[idx7]

    def _volume(self, cube, moment):
        r0, r1, g0, g1, b0, b1 = cube
        def idx(r, g, b):
            return self._index(r, g, b)

        return (
            moment[idx(r1, g1, b1)]
            - moment[idx(r1, g1, b0)]
            - moment[idx(r1, g0, b1)]
            - moment[idx(r0, g1, b1)]
            + moment[idx(r1, g0, b0)]
            + moment[idx(r0, g1, b0)]
            + moment[idx(r0, g0, b1)]
            - moment[idx(r0, g0, b0)]
        )

    def _variance(self, cube):
        vol = self._volume(cube, self.weights)
        if vol == 0:
            return 0
        r = self._volume(cube, self.moments_r)
        g = self._volume(cube, self.moments_g)
        b = self._volume(cube, self.moments_b)
        m = self._volume(cube, self.moments)
        return m - (r * r + g * g + b * b) / vol

    def _maximize(self, cube, direction, first, last, whole):
        whole_r, whole_g, whole_b, whole_w = whole
        best_var = -1
        cut = first
        for i in range(first, last):
            if direction == 0:
                left = (cube[0], i, cube[2], cube[3], cube[4], cube[5])
                right = (i, cube[1], cube[2], cube[3], cube[4], cube[5])
            elif direction == 1:
                left = (cube[0], cube[1], cube[2], i, cube[4], cube[5])
                right = (cube[0], cube[1], i, cube[3], cube[4], cube[5])
            else:
                left = (cube[0], cube[1], cube[2], cube[3], cube[4], i)
                right = (cube[0], cube[1], cube[2], cube[3], i, cube[5])

            weight_left = self._volume(left, self.weights)
            weight_right = whole_w - weight_left
            if weight_left == 0 or weight_right == 0:
                continue

            r_left = self._volume(left, self.moments_r)
            g_left = self._volume(left, self.moments_g)
            b_left = self._volume(left, self.moments_b)

            r_right = whole_r - r_left
            g_right = whole_g - g_left
            b_right = whole_b - b_left

            value = (
                (r_left * r_left + g_left * g_left + b_left * b_left) / weight_left
                + (r_right * r_right + g_right * g_right + b_right * b_right) / weight_right
            )
            if value > best_var:
                best_var = value
                cut = i

        return cut, best_var

    def _cut(self, cube):
        whole_r = self._volume(cube, self.moments_r)
        whole_g = self._volume(cube, self.moments_g)
        whole_b = self._volume(cube, self.moments_b)
        whole_w = self._volume(cube, self.weights)

        if whole_w == 0:
            return None

        best = None
        for direction in range(3):
            first = cube[direction * 2] + 1
            last = cube[direction * 2 + 1]
            if first >= last:
                continue
            cut = self._maximize(cube, direction, first, last, (whole_r, whole_g, whole_b, whole_w))
            if best is None or cut[1] > best[1]:
                best = (direction, cut[0], cut[1])

        if best is None or best[1] <= 0:
            return None

        direction, position, _ = best
        child = list(cube)
        parent = list(cube)
        if direction == 0:
            child[0] = position
            parent[1] = position
        elif direction == 1:
            child[2] = position
            parent[3] = position
        else:
            child[4] = position
            parent[5] = position

        return tuple(parent), tuple(child)

    def quantize(self, pixels, max_colors=16):
        if not pixels:
            return [(0, 0, 0)]

        self._build_histogram(pixels)
        self._prefix_sums()

        cubes = [(0, self.size - 1, 0, self.size - 1, 0, self.size - 1)]

        while len(cubes) < max_colors:
            cubes.sort(key=self._variance, reverse=True)
            cube = cubes.pop(0)
            split = self._cut(cube)
            if split is None:
                cubes.append(cube)
                break
            cubes.extend(split)

        palette = []
        for cube in cubes:
            weight = self._volume(cube, self.weights)
            if weight == 0:
                continue
            r = self._volume(cube, self.moments_r) / weight
            g = self._volume(cube, self.moments_g) / weight
            b = self._volume(cube, self.moments_b) / weight
            palette.append((int(r), int(g), int(b)))

        return palette or [(0, 0, 0)]


def apply_wu_quantizer(img):
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    wu = WuQuantizer()
    palette = wu.quantize(pixels, max_colors=16)

    def nearest_palette(rgb):
        return min(palette, key=lambda p: (p[0] - rgb[0]) ** 2 + (p[1] - rgb[1]) ** 2 + (p[2] - rgb[2]) ** 2)

    grid = []
    for y in range(height):
        row = []
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
