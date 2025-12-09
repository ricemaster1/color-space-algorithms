from PIL import Image, ImageFilter
import webcolors
import os
import sys
import argparse

# ARMLITE color names
ARMLITE_COLORS = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
    'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
    'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue',
    'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta',
    'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
    'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen',
    'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
    'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
    'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon',
    'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue',
    'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue',
    'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream',
    'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred',
    'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna',
    'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen',
    'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
    'whitesmoke', 'yellow', 'yellowgreen'
]

ARMLITE_RGB = {name: webcolors.name_to_rgb(name) for name in ARMLITE_COLORS}

def closest_color(rgb):
    min_dist = float('inf')
    closest = None
    for name, color_rgb in ARMLITE_RGB.items():
        dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, color_rgb))
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest

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
