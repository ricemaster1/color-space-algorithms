from PIL import Image
import webcolors
import sys
import os
import argparse
import math

# ARMLite color palette
ARMLITE_COLORS = [
    'aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black',
    'blanchedalmond','blue','blueviolet','brown','burlywood','cadetblue','chartreuse',
    'chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue',
    'darkcyan','darkgoldenrod','darkgray','darkgreen','darkgrey','darkkhaki','darkmagenta',
    'darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen',
    'darkslateblue','darkslategray','darkslategrey','darkturquoise','darkviolet','deeppink',
    'deepskyblue','dimgray','dimgrey','dodgerblue','firebrick','floralwhite','forestgreen',
    'fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow',
    'grey','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush',
    'lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow',
    'lightgray','lightgreen','lightgrey','lightpink','lightsalmon','lightseagreen','lightskyblue',
    'lightslategray','lightslategrey','lightsteelblue','lightyellow','lime','limegreen','linen',
    'magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream',
    'mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid',
    'palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum',
    'powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen',
    'seashell','sienna','silver','skyblue','slateblue','slategray','slategrey','snow','springgreen',
    'steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow',
    'yellowgreen'
]

# RGB lookup
ARMLITE_RGB = {name: webcolors.name_to_rgb(name) for name in ARMLITE_COLORS}

def color_distance(c1, c2):
    return sum((a-b)**2 for a,b in zip(c1,c2))

def top_n_colors(rgb, n):
    sorted_colors = sorted(ARMLITE_RGB.items(), key=lambda x: color_distance(rgb, x[1]))
    return [name for name, _ in sorted_colors[:n]]

def node_based_quantization(img, n=3, iterations=3):
    width, height = img.size
    pixels = list(img.getdata())
    grid = [[None for _ in range(width)] for _ in range(height)]

    # Initialize nodes with top-N candidates
    for y in range(height):
        for x in range(width):
            grid[y][x] = top_n_colors(pixels[y*width + x], n)

    # Iteratively assign colors minimizing local energy
    for it in range(iterations):
        for y in range(height):
            for x in range(width):
                best_energy = float('inf')
                best_color = None
                for candidate in grid[y][x]:
                    rgb_c = ARMLITE_RGB[candidate]
                    # Energy = distance to original + neighbors
                    energy = color_distance(rgb_c, pixels[y*width + x])
                    for dx in [-1,0,1]:
                        for dy in [-1,0,1]:
                            nx, ny = x+dx, y+dy
                            if 0<=nx<width and 0<=ny<height and not (dx==0 and dy==0):
                                neighbor_color = grid[ny][nx][0]  # take current best
                                energy += 0.5 * color_distance(rgb_c, ARMLITE_RGB[neighbor_color])
                    if energy < best_energy:
                        best_energy = energy
                        best_color = candidate
                grid[y][x] = [best_color]  # lock in best candidate

    # Flatten to final pixel color names
    final_colors = [[grid[y][x][0] for x in range(width)] for y in range(height)]
    return final_colors

def generate_assembly(colors, output_path):
    width, height = 128, 96
    lines = ['; === Fullscreen Sprite ===',
             '    MOV R0, #2',
             '    STR R0, .Resolution',
             '    MOV R1, #.PixelScreen',
             '    MOV R6, #512 ; row stride (128*4)']
    for y in range(height):
        for x in range(width):
            offset = ((y * width) + x) * 4
            addr_line = f'    MOV R5, #{offset}\n    ADD R4, R1, R5'
            color_name = colors[y][x]
            write_line = f'    MOV R0, #.{color_name}\n    STR R0, [R4]   ; Pixel ({x},{y})'
            lines.append(addr_line)
            lines.append(write_line)
    lines.append('    HALT')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Assembly sprite file written to {output_path}")

def process_image(image_path, output_path, n=3):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128,96))
    final_colors = node_based_quantization(img, n=n)
    generate_assembly(final_colors, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARMLite node-based color quantizer')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', default='converted.s', help='Output assembly file path')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("Image not found.")
        sys.exit(1)

    # Ask user for top-N closest colors
    while True:
        try:
            n_input = int(input("Enter number of closest colors to consider (e.g., 1-10): "))
            if n_input < 1:
                print("N must be at least 1.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    process_image(args.image, args.output, n=n_input)
