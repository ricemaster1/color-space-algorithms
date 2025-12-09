from PIL import Image
import argparse
import os
import sys
import math

from armlite import ARMLITE_RGB


def _srgb_to_xyz(channel: float) -> float:
    channel /= 255.0
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb):
    r, g, b = rgb
    r_lin = _srgb_to_xyz(r)
    g_lin = _srgb_to_xyz(g)
    b_lin = _srgb_to_xyz(b)

    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def f(t):
        epsilon = 0.008856
        kappa = 903.3
        if t > epsilon:
            return t ** (1 / 3)
        return (kappa * t + 16) / 116

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return (L, a, b_val)


def delta_e_cie94(lab1, lab2, application='graphic_arts'):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    delta_L = L1 - L2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    delta_C = C1 - C2

    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_sq = max(delta_a ** 2 + delta_b ** 2 - delta_C ** 2, 0)

    if application == 'textiles':
        k_L, k_C, k_H = 2, 1, 1
        K1, K2 = 0.048, 0.014
    else:
        k_L, k_C, k_H = 1, 1, 1
        K1, K2 = 0.045, 0.015

    S_L = 1
    S_C = 1 + K1 * C1
    S_H = 1 + K2 * C1

    delta_E = math.sqrt(
        (delta_L / (k_L * S_L)) ** 2
        + (delta_C / (k_C * S_C)) ** 2
        + (math.sqrt(delta_H_sq) / (k_H * S_H)) ** 2
    )
    return delta_E


def apply_distance_cie94(img):
    """Quantize using minimized CIE94 Delta E to the ARMLite palette."""
    width, height = img.size
    palette_lab = {name: rgb_to_lab(rgb) for name, rgb in ARMLITE_RGB.items()}
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            lab_src = rgb_to_lab(img.getpixel((x, y)))
            best = None
            best_delta = float('inf')
            for name, lab in palette_lab.items():
                delta = delta_e_cie94(lab_src, lab)
                if delta < best_delta:
                    best_delta = delta
                    best = name
            row.append(best or 'black')
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
    grid = apply_distance_cie94(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CIE94 distance renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
