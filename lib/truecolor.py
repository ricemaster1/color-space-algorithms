"""
True Color utilities for ARMlite assembly generation.

This module provides functions to generate ARMlite assembly using full 24-bit
RGB hex values instead of the limited 147 CSS3 named colors.

Usage:
    from lib.truecolor import rgb_to_hex, generate_truecolor_assembly
"""

from __future__ import annotations

from datetime import datetime
import os
from typing import Sequence

# Resolution presets: (width, height, .Resolution value)
RESOLUTIONS = {
    'low': (32, 32, 0),   # 32x32 low-res (direct addressed)
    'mid': (64, 48, 1),   # 64x48 mid-res
    'hi': (128, 96, 2),   # 128x96 hi-res
}


def rgb_to_hex(r: int, g: int, b: int) -> int:
    """Convert RGB tuple to 24-bit hex value."""
    return (r << 16) | (g << 8) | b


def hex_to_rgb(hex_val: int) -> tuple[int, int, int]:
    """Convert 24-bit hex value to RGB tuple."""
    return ((hex_val >> 16) & 0xFF, (hex_val >> 8) & 0xFF, hex_val & 0xFF)


def generate_truecolor_assembly(
    color_grid: Sequence[Sequence[int]],
    output_path: str,
    *,
    image_path: str = '',
    resolution: str = 'hi',
    comment: str = '',
) -> None:
    """
    Generate ARMlite assembly file with true color hex values.
    
    Args:
        color_grid: 2D grid of 24-bit hex color values (0xRRGGBB)
        output_path: Path to write assembly file
        image_path: Original image path (for comments)
        resolution: Resolution preset ('low', 'mid', 'hi')
        comment: Additional comment to include in header
    """
    height = len(color_grid)
    width = len(color_grid[0]) if height else 0
    _, _, res_value = RESOLUTIONS.get(resolution, RESOLUTIONS['hi'])
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    lines = [
        '; === True Color Sprite ===',
        f'; Generated: {timestamp}',
    ]
    
    if image_path:
        lines.append(f'; Source: {os.path.basename(image_path)}')
    if comment:
        lines.append(f'; {comment}')
    
    lines.extend([
        f'; Resolution: {width}x{height} (mode {res_value})',
        '; Full 24-bit RGB - no palette quantization',
        '',
        f'    MOV R0, #{res_value}',
        '    STR R0, .Resolution',
        '    MOV R1, #.PixelScreen',
    ])
    
    for y in range(height):
        for x in range(width):
            offset = ((y * width) + x) * 4
            hex_color = color_grid[y][x]
            lines.append(f'    MOV R5, #{offset}')
            lines.append('    ADD R4, R1, R5')
            lines.append(f'    MOV R0, #0x{hex_color:06X}')
            lines.append(f'    STR R0, [R4]   ; ({x},{y})')
    
    lines.append('    HALT')
    
    with open(output_path, 'w') as fh:
        fh.write('\n'.join(lines))


def generate_truecolor_assembly_optimized(
    color_grid: Sequence[Sequence[int]],
    output_path: str,
    *,
    image_path: str = '',
    resolution: str = 'hi',
    comment: str = '',
) -> None:
    """
    Generate optimized ARMlite assembly using data section and loop.
    
    This produces much smaller assembly files by storing pixel data
    in a .DATA section and using a loop to draw.
    
    Args:
        color_grid: 2D grid of 24-bit hex color values (0xRRGGBB)
        output_path: Path to write assembly file
        image_path: Original image path (for comments)
        resolution: Resolution preset ('low', 'mid', 'hi')
        comment: Additional comment to include in header
    """
    height = len(color_grid)
    width = len(color_grid[0]) if height else 0
    _, _, res_value = RESOLUTIONS.get(resolution, RESOLUTIONS['hi'])
    total_pixels = width * height
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Flatten the grid to a list of hex values
    pixels = [color_grid[y][x] for y in range(height) for x in range(width)]
    unique_colors = len(set(pixels))
    
    lines = [
        '; === True Color Sprite (Optimized) ===',
        f'; Generated: {timestamp}',
    ]
    
    if image_path:
        lines.append(f'; Source: {os.path.basename(image_path)}')
    if comment:
        lines.append(f'; {comment}')
    
    lines.extend([
        f'; Resolution: {width}x{height} (mode {res_value})',
        f'; {total_pixels} pixels, {unique_colors} unique colors',
        '; Full 24-bit RGB - no palette quantization',
        '',
        f'    MOV R0, #{res_value}',
        '    STR R0, .Resolution',
        '',
        '    MOV R1, #.PixelScreen',
        '    MOV R2, #pixels',
        f'    MOV R3, #{total_pixels}',
        '    MOV R4, #0          ; pixel index',
        '',
        'draw_loop:',
        '    LDR R5, [R2]        ; load color from data',
        '    STR R5, [R1]        ; write to screen',
        '    ADD R1, R1, #4      ; next screen pixel',
        '    ADD R2, R2, #4      ; next data word',
        '    ADD R4, R4, #1',
        '    CMP R4, R3',
        '    BLT draw_loop',
        '',
        '    HALT',
        '',
        '.DATA',
        'pixels:',
    ])
    
    # Add pixel data
    for px in pixels:
        lines.append(f'    .WORD 0x{px:06X}')
    
    with open(output_path, 'w') as fh:
        fh.write('\n'.join(lines))
