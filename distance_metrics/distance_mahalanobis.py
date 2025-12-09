from __future__ import annotations

from PIL import Image
import argparse
import os
import sys

from armlite import ARMLITE_RGB


def _mean_vector(pixels):
    n = len(pixels)
    if n == 0:
        return (0.0, 0.0, 0.0)
    total_r = sum(p[0] for p in pixels)
    total_g = sum(p[1] for p in pixels)
    total_b = sum(p[2] for p in pixels)
    return (total_r / n, total_g / n, total_b / n)


def _covariance_matrix(pixels, mean):
    n = len(pixels)
    if n <= 1:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    accum = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    for r, g, b in pixels:
        dr = r - mean[0]
        dg = g - mean[1]
        db = b - mean[2]
        accum[0][0] += dr * dr
        accum[0][1] += dr * dg
        accum[0][2] += dr * db
        accum[1][0] += dg * dr
        accum[1][1] += dg * dg
        accum[1][2] += dg * db
        accum[2][0] += db * dr
        accum[2][1] += db * dg
        accum[2][2] += db * db
    scale = 1.0 / (n - 1)
    for i in range(3):
        for j in range(3):
            accum[i][j] *= scale
    return accum


def _invert_3x3(m):
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if abs(det) < 1e-9:
        return None
    inv_det = 1.0 / det
    adj = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
            -(m[0][1] * m[2][2] - m[0][2] * m[2][1]),
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        ],
        [
            -(m[1][0] * m[2][2] - m[1][2] * m[2][0]),
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]),
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
            -(m[0][0] * m[2][1] - m[0][1] * m[2][0]),
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]),
        ],
    ]
    for i in range(3):
        for j in range(3):
            adj[i][j] *= inv_det
    return adj


def _mahalanobis_sq(delta, inv_cov):
    dr, dg, db = delta
    return (
        dr * (inv_cov[0][0] * dr + inv_cov[0][1] * dg + inv_cov[0][2] * db)
        + dg * (inv_cov[1][0] * dr + inv_cov[1][1] * dg + inv_cov[1][2] * db)
        + db * (inv_cov[2][0] * dr + inv_cov[2][1] * dg + inv_cov[2][2] * db)
    )


def apply_distance_mahalanobis(img: Image.Image, epsilon: float = 1e-3) -> list[list[str]]:
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
    if not pixels:
        return []
    mean = _mean_vector(pixels)
    cov = _covariance_matrix(pixels, mean)
    for i in range(3):
        cov[i][i] += epsilon
    inv_cov = _invert_3x3(cov)
    if inv_cov is None:
        inv_cov = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    palette = list(ARMLITE_RGB.items())
    mapped: list[str] = []
    for rgb in pixels:
        best_name = palette[0][0]
        pr, pg, pb = palette[0][1]
        best_dist = _mahalanobis_sq((rgb[0] - pr, rgb[1] - pg, rgb[2] - pb), inv_cov)
        for name, color in palette[1:]:
            delta = (rgb[0] - color[0], rgb[1] - color[1], rgb[2] - color[2])
            dist = _mahalanobis_sq(delta, inv_cov)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        mapped.append(best_name)

    grid: list[list[str]] = []
    idx = 0
    for _ in range(height):
        row: list[str] = []
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


def process_image(image_path, output_path, epsilon: float):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_distance_mahalanobis(img, epsilon=epsilon)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mahalanobis distance renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='Diagonal regularization term added to covariance (default: 1e-3)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output, args.epsilon)
