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

from lib import ARMLITE_RGB


def top_n_colors(rgb, n=3):
    """Return the top-N closest ARMLite colors to the given RGB."""
    distances = []
    for name, color_rgb in ARMLITE_RGB.items():
        dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, color_rgb))
        distances.append((dist, name))
    distances.sort(key=lambda x: x[0])
    return [name for _, name in distances[:n]]

def select_best_color(x, y, pixels, width, height, n=3):
    """Pick the best color for a pixel based on neighbors and top-N candidates."""
    candidates = top_n_colors(pixels[x, y], n)
    if len(candidates) == 1:
        return candidates[0]

    # Compare against 4-connected neighbors
    neighbor_colors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbor_colors.append(pixels[nx, ny])

    best = candidates[0]
    min_total_dist = float('inf')
    for candidate in candidates:
        candidate_rgb = ARMLITE_RGB[candidate]
        total_dist = sum(sum((c1 - c2) ** 2 for c1, c2 in zip(candidate_rgb, n_rgb)) for n_rgb in neighbor_colors)
        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best = candidate
    return best

def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
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
            color_name = select_best_color(x, y, pixels, width, height, n=3)
            pixels[x, y] = ARMLITE_RGB[color_name]  # optional: update pixels
            write_line = f'    MOV R0, #.{color_name}\n    STR R0, [R4]   ; Pixel ({x},{y})'
            lines.append(addr_line)
            lines.append(write_line)

    lines.append('    HALT')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Assembly sprite file written to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARMLite context-aware color quantizer')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', help='Output assembly file path (default: converted.s)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("Image not found.")
        exit(1)
    
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
