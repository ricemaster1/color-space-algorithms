from PIL import Image, ImageFilter
import os
import sys
import argparse
from pathlib import Path

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import closest_color


def apply_ordered_dither(img):
    """2x2 Bayer matrix ordered dithering scaled for 128x96 pixels"""
    bayer2x2 = [[0, 2],
                [3, 1]]
    width, height = img.size
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            threshold = bayer2x2[y % 2][x % 2]
            # Scale factor to subtle effect
            r = min(255, max(0, r + (threshold - 1) * 8))
            g = min(255, max(0, g + (threshold - 1) * 8))
            b = min(255, max(0, b + (threshold - 1) * 8))
            pixels[x, y] = (r, g, b)
    return img

def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    img = img.filter(ImageFilter.BoxBlur(1))  # subtle smoothing
    img = apply_ordered_dither(img)
    pixels = img.load()
    width, height = img.size

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
            addr_line = f'    MOV R5, #{offset}\n    ADD R4, R1, R5'
            color_name = closest_color(pixels[x, y])
            write_line = f'    MOV R0, #.{color_name}\n    STR R0, [R4]   ; Pixel ({x},{y})'
            lines.append(addr_line)
            lines.append(write_line)

    lines.append('    HALT')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Assembly sprite file written to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an image to ARMLite assembly sprite.')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("Image not found.")
        sys.exit(1)

    process_image(args.image, args.output)
