#!/usr/bin/env python3
"""
True Color Converter for ARMlite
================================
Converts images to ARMlite assembly using full 24-bit RGB hex values.

NO quantization, NO palette limitation - uses the EXACT colors from your image!
This bypasses the 147 CSS3 named color restriction entirely.

Usage:
    python truecolor.py input.png [output.s]
    python truecolor.py input.png --resolution mid  # 64x48
    python truecolor.py input.png --resolution hi   # 128x96 (default)
"""

from __future__ import annotations

from PIL import Image
from datetime import datetime
import argparse
import os
import sys
from pathlib import Path

# Resolution presets
RESOLUTIONS = {
    'mid': (64, 48, 1),   # 64x48, .Resolution = 1
    'hi': (128, 96, 2),   # 128x96, .Resolution = 2
}


def rgb_to_hex(r: int, g: int, b: int) -> int:
    """Convert RGB tuple to 24-bit hex value."""
    return (r << 16) | (g << 8) | b


def apply_truecolor(img: Image.Image, resolution: str = 'hi') -> list[list[int]]:
    """
    Convert image to grid of hex color values.
    
    Args:
        img: PIL Image in RGB mode
        resolution: 'mid' (64x48) or 'hi' (128x96)
    
    Returns:
        2D grid of 24-bit hex color values
    """
    width, height, _ = RESOLUTIONS[resolution]
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    pixels = img.load()
    color_grid: list[list[int]] = []
    
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            r, g, b = pixels[x, y][:3]  # Handle RGBA too
            row.append(rgb_to_hex(r, g, b))
        color_grid.append(row)
    
    return color_grid


def generate_assembly(color_grid: list[list[int]], output_path: str, *,
                      image_path: str = '', resolution: str = 'hi'):
    """
    Generate ARMlite assembly file with true color hex values.
    
    Args:
        color_grid: 2D grid of 24-bit hex color values
        output_path: Path to write assembly file
        image_path: Original image path (for comments)
        resolution: Resolution preset used
    """
    height = len(color_grid)
    width = len(color_grid[0]) if height else 0
    _, _, res_value = RESOLUTIONS[resolution]
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    lines = [
        '; === True Color Fullscreen Sprite ===',
        f'; Generated: {timestamp}',
        f'; Source: {os.path.basename(image_path)}' if image_path else '',
        f'; Resolution: {width}x{height} (mode {res_value})',
        '; Uses raw 24-bit RGB hex values - NO palette quantization!',
        '',
        f'    MOV R0, #{res_value}',
        '    STR R0, .Resolution',
        '    MOV R1, #.PixelScreen',
    ]
    
    # Filter out empty comment lines
    lines = [line for line in lines if line]
    
    for y in range(height):
        for x in range(width):
            offset = ((y * width) + x) * 4
            hex_color = color_grid[y][x]
            lines.append(f'    MOV R5, #{offset}')
            lines.append('    ADD R4, R1, R5')
            lines.append(f'    MOV R0, #0x{hex_color:06X}')
            lines.append(f'    STR R0, [R4]   ; Pixel ({x},{y})')
    
    lines.append('    HALT')
    
    with open(output_path, 'w') as fh:
        fh.write('\n'.join(lines))
    
    total_pixels = width * height
    unique_colors = len(set(c for row in color_grid for c in row))
    print(f"✓ True color assembly written to {output_path}")
    print(f"  {total_pixels} pixels, {unique_colors} unique colors")
    print(f"  Resolution: {width}x{height}")


def process_image(image_path: str, output_path: str, resolution: str = 'hi'):
    """Process an image and generate assembly output."""
    img = Image.open(image_path).convert('RGB')
    color_grid = apply_truecolor(img, resolution)
    generate_assembly(color_grid, output_path, image_path=image_path, resolution=resolution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='True Color converter for ARMlite sprites - uses full 24-bit RGB!'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', help='Output assembly file path (default: truecolor.s)')
    parser.add_argument('--resolution', '-r', choices=['mid', 'hi'], default='hi',
                        help='Output resolution: mid (64x48) or hi (128x96, default)')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.image):
        print('Error: Image not found.')
        sys.exit(1)
    
    output_path = args.output
    if output_path:
        output_path = os.path.expanduser(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'truecolor.s')
        else:
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                print(f'Error: Output directory does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = 'truecolor.s'
    
    process_image(args.image, output_path, args.resolution)
