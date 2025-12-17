from PIL import Image
import sys
import os
import argparse
from pathlib import Path

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB, closest_color


def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    width, height = img.size
    pixels = img.load()  # pixel access object for in-place modification

    # Floyd-Steinberg dithering
    for y in range(height):
        for x in range(width):
            old_pixel = pixels[x, y]
            color_name = closest_color(old_pixel)
            new_pixel = ARMLITE_RGB[color_name]
            pixels[x, y] = new_pixel

            # Calculate error
            err = tuple(old_pixel[i] - new_pixel[i] for i in range(3))

            # Distribute the error
            def add_error(px, py, factor):
                if 0 <= px < width and 0 <= py < height:
                    r, g, b = pixels[px, py]
                    r = min(255, max(0, int(r + err[0] * factor)))
                    g = min(255, max(0, int(g + err[1] * factor)))
                    b = min(255, max(0, int(b + err[2] * factor)))
                    pixels[px, py] = (r, g, b)

            add_error(x + 1, y, 7/16)
            add_error(x - 1, y + 1, 3/16)
            add_error(x, y + 1, 5/16)
            add_error(x + 1, y + 1, 1/16)

    # Begin generating assembly
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
