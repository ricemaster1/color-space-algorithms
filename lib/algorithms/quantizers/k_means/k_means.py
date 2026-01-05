from __future__ import annotations

from PIL import Image
import argparse
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


def _euclidean_sq(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr * dr + dg * dg + db * db


def _initialize_centroids(pixels, k, rng):
    if len(pixels) <= k:
        return pixels.copy()
    centroids = [rng.choice(pixels)]
    distances = [0.0] * len(pixels)
    for _ in range(1, k):
        total = 0.0
        for idx, pixel in enumerate(pixels):
            dist = min(_euclidean_sq(pixel, centroid) for centroid in centroids)
            distances[idx] = dist
            total += dist
        if total == 0:
            centroids.append(rng.choice(pixels))
            continue
        target = rng.random() * total
        cumulative = 0.0
        for idx, dist in enumerate(distances):
            cumulative += dist
            if cumulative >= target:
                centroids.append(pixels[idx])
                break
    return centroids


def _assign_pixels(pixels, centroids):
    assignments = []
    for pixel in pixels:
        best_idx = 0
        best_dist = _euclidean_sq(pixel, centroids[0])
        for idx, centroid in enumerate(centroids[1:], start=1):
            dist = _euclidean_sq(pixel, centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        assignments.append(best_idx)
    return assignments


def _update_centroids(pixels, assignments, k, rng):
    sums = [[0.0, 0.0, 0.0, 0] for _ in range(k)]
    for pixel, idx in zip(pixels, assignments):
        entry = sums[idx]
        entry[0] += pixel[0]
        entry[1] += pixel[1]
        entry[2] += pixel[2]
        entry[3] += 1
    centroids = []
    for total_r, total_g, total_b, count in sums:
        if count == 0:
            centroids.append(rng.choice(pixels))
        else:
            centroids.append(
                (
                    total_r / count,
                    total_g / count,
                    total_b / count,
                )
            )
    return centroids


def _converged(old_centroids, new_centroids, tolerance):
    for old, new in zip(old_centroids, new_centroids):
        if _euclidean_sq(old, new) > tolerance * tolerance:
            return False
    return True


def apply_k_means(img, clusters=16, max_iter=15, tolerance=1.5, seed=13):
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    k = min(clusters, len(pixels), len(ARMLITE_RGB))
    rng = random.Random(seed)
    centroids = _initialize_centroids(pixels, k, rng)

    for _ in range(max_iter):
        assignments = _assign_pixels(pixels, centroids)
        new_centroids = _update_centroids(pixels, assignments, k, rng)
        if _converged(centroids, new_centroids, tolerance):
            centroids = new_centroids
            break
        centroids = new_centroids
    else:
        assignments = _assign_pixels(pixels, centroids)

    palette_names = [closest_color((int(round(c[0])), int(round(c[1])), int(round(c[2])))) for c in centroids]

    grid = []
    idx = 0
    for y in range(height):
        row = []
        for x in range(width):
            cluster_idx = assignments[idx]
            idx += 1
            row.append(palette_names[cluster_idx])
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
    grid = apply_k_means(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-means renderer for ARMLite sprites'
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
            if parent and not os.path.exists(parent)
                print(f'Output path does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = 'converted.s'

    process_image(args.image, args.output)
