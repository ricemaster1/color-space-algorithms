from PIL import Image
import argparse
import os
import sys

from armlite import closest_color, ARMLITE_RGB


def apply_jarvis_judice_ninke(img):
    """Apply Jarvis–Judice–Ninke error diffusion to map pixels onto the ARMLite palette."""

    def clamp(value: float) -> int:
        return max(0, min(255, int(round(value))))

    weights = [
        (1, 0, 7 / 48), (2, 0, 5 / 48),
        (-2, 1, 3 / 48), (-1, 1, 5 / 48), (0, 1, 7 / 48), (1, 1, 5 / 48), (2, 1, 3 / 48),
        (-2, 2, 1 / 48), (-1, 2, 3 / 48), (0, 2, 5 / 48), (1, 2, 3 / 48), (2, 2, 1 / 48),
    ]

    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            old_pixel = pixels[x, y]
            color_name = closest_color(old_pixel)
            new_pixel = ARMLITE_RGB[color_name]
            pixels[x, y] = new_pixel

            err = tuple(old_pixel[i] - new_pixel[i] for i in range(3))
            if err == (0, 0, 0):
                continue

            for dx, dy, weight in weights:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    r, g, b = pixels[nx, ny]
                    r = clamp(r + err[0] * weight)
                    g = clamp(g + err[1] * weight)
                    b = clamp(b + err[2] * weight)
                    pixels[nx, ny] = (r, g, b)

    # Convert final image into palette name grid
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(closest_color(pixels[x, y]))
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
    grid = apply_jarvis_judice_ninke(img)
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Jarvis-Judice-Ninke renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    process_image(args.image, args.output)
