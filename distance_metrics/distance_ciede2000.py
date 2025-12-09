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

    # sRGB D65 conversion matrix
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    xn, yn, zn = 0.95047, 1.00000, 1.08883  # D65 reference white

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


def delta_e_ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - math.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = math.hypot(a1_prime, b1)
    C2_prime = math.hypot(a2_prime, b2)

    def _atan2_deg(y_val, x_val):
        angle = math.degrees(math.atan2(y_val, x_val))
        return angle + 360 if angle < 0 else angle

    h1_prime = 0 if C1_prime == 0 else _atan2_deg(b1, a1_prime)
    h2_prime = 0 if C2_prime == 0 else _atan2_deg(b2, a2_prime)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        diff = h2_prime - h1_prime
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        delta_h_prime = diff

    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))

    avg_L_prime = (L1 + L2) / 2.0
    avg_C_prime = (C1_prime + C2_prime) / 2.0

    if C1_prime * C2_prime == 0:
        avg_h_prime = h1_prime + h2_prime
    else:
        diff = abs(h1_prime - h2_prime)
        if diff > 180:
            avg_h_prime = (h1_prime + h2_prime + 360) / 2.0 if h1_prime + h2_prime < 360 else (h1_prime + h2_prime - 360) / 2.0
        else:
            avg_h_prime = (h1_prime + h2_prime) / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(avg_h_prime - 30))
        + 0.24 * math.cos(math.radians(2 * avg_h_prime))
        + 0.32 * math.cos(math.radians(3 * avg_h_prime + 6))
        - 0.20 * math.cos(math.radians(4 * avg_h_prime - 63))
    )

    delta_theta = 30 * math.exp(-((avg_h_prime - 275) / 25) ** 2)
    R_C = 2 * math.sqrt((avg_C_prime ** 7) / (avg_C_prime ** 7 + 25 ** 7))
    S_L = 1 + (0.015 * (avg_L_prime - 50) ** 2) / math.sqrt(20 + (avg_L_prime - 50) ** 2)
    S_C = 1 + 0.045 * avg_C_prime
    S_H = 1 + 0.015 * avg_C_prime * T
    R_T = -math.sin(math.radians(2 * delta_theta)) * R_C

    delta_E = math.sqrt(
        (delta_L_prime / S_L) ** 2
        + (delta_C_prime / S_C) ** 2
        + (delta_H_prime / S_H) ** 2
        + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )
    return delta_E


def apply_distance_ciede2000(img):
    """Quantize using minimized CIEDE2000 Delta E to the ARMLite palette."""
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
                delta = delta_e_ciede2000(lab_src, lab)
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
    grid = apply_distance_ciede2000(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CIEDE2000 distance renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
