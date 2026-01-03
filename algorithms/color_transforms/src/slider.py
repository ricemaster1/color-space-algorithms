"""
slider.py - Interactive weight adjustment for ARMLite sprite conversion

Generates an ARMLite assembly program with BOTH HSV and HSL pre-rendered versions.
Toggle between color spaces and enter weights via the I/O console.

ARMLite Controls:
  T: Toggle between HSV and HSL color space (instant switching)
  W: Enter new weights via console (integers scaled by 100)
  P: Print current weights to console  
  R: Reset to default weights

Weights are entered as integers scaled by 100:
  - 2.7 → enter 270
  - 0.42 → enter 42
  - 8.0 → enter 800

Usage:
  python slider.py image.png [output.s] [--hsv H,S,V] [--hsl H,S,L]
"""

from __future__ import annotations

from PIL import Image
import argparse
import colorsys
from datetime import datetime
import math
import os
import sys
from pathlib import Path
from typing import Callable

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB


# === Color space transforms ===

def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*(channel / 255.0 for channel in rgb))


def _rgb_to_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*(channel / 255.0 for channel in rgb))
    return (h, s, l)


def _palette_space(
    palette: dict[str, tuple[int, int, int]],
    transform: Callable[[tuple[int, int, int]], tuple[float, float, float]]
) -> dict[str, tuple[float, float, float]]:
    return {name: transform(rgb) for name, rgb in palette.items()}


def weighted_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    w: tuple[float, float, float]
) -> float:
    """Calculate weighted distance between two color space points."""
    dh = min(abs(a[0] - b[0]), 1.0 - abs(a[0] - b[0]))
    ds = abs(a[1] - b[1])
    dv = abs(a[2] - b[2])
    return math.sqrt((w[0] * dh) ** 2 + (w[1] * ds) ** 2 + (w[2] * dv) ** 2)


def render_image_data(
    img: Image.Image,
    transform: Callable[[tuple[int, int, int]], tuple[float, float, float]],
    palette_transformed: dict[str, tuple[float, float, float]],
    weights: tuple[float, float, float]
) -> list[int]:
    """Render image to palette colors, return list of RGB hex values."""
    width, height = img.size
    pixels = []
    
    for y in range(height):
        for x in range(width):
            rgb = img.getpixel((x, y))
            converted = transform(rgb)
            
            best_name = 'black'
            best_dist = float('inf')
            for name, target in palette_transformed.items():
                d = weighted_distance(converted, target, weights)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            
            color_value = ARMLITE_RGB[best_name]
            hex_value = (color_value[0] << 16) | (color_value[1] << 8) | color_value[2]
            pixels.append(hex_value)
    
    return pixels


# === Assembly generation ===

def generate_slider_assembly(
    img: Image.Image,
    output_path: str,
    hsv_weights: tuple[float, float, float],
    hsl_weights: tuple[float, float, float],
    image_name: str = ''
) -> None:
    """
    Generate an ARMLite assembly program with dual HSV/HSL images and console I/O.
    
    Features:
    - Pre-rendered HSV and HSL versions of the image
    - Toggle between color spaces with 'T' key
    - Enter weights via console with 'W' key (for copy/paste workflow)
    - Print current settings with 'P' key
    """
    
    width, height = img.size
    total_pixels = width * height
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Pre-render both color spaces
    hsv_transform = _rgb_to_hsv
    hsl_transform = _rgb_to_hsl
    hsv_palette = _palette_space(ARMLITE_RGB, hsv_transform)
    hsl_palette = _palette_space(ARMLITE_RGB, hsl_transform)
    
    print(f"Rendering HSV version with weights {hsv_weights}...")
    hsv_pixels = render_image_data(img, hsv_transform, hsv_palette, hsv_weights)
    
    print(f"Rendering HSL version with weights {hsl_weights}...")
    hsl_pixels = render_image_data(img, hsl_transform, hsl_palette, hsl_weights)
    
    lines = []
    
    # Header comments
    lines.extend([
        '; === Interactive Weight Slider with Console I/O ===',
        f'; Generated: {timestamp}',
        f'; Source: {image_name}' if image_name else '',
        f'; HSV weights: ({hsv_weights[0]}, {hsv_weights[1]}, {hsv_weights[2]})',
        f'; HSL weights: ({hsl_weights[0]}, {hsl_weights[1]}, {hsl_weights[2]})',
        ';',
        '; Controls:',
        ';   T: Toggle HSV/HSL color space',
        ';   W: Enter new weights (integers x100)',
        ';   P: Print current weights to console',
        ';   R: Reset to default weights',
        ';',
    ])
    lines = [l for l in lines if l]
    
    # === Main program ===
    lines.extend([
        '',
        '; === Initialize ===',
        '        MOV R0, #2',
        '        STR R0, .Resolution        ; Hi-res 128x96',
        '',
        '; Initialize state: 0=HSV, 1=HSL',
        '        MOV R0, #0',
        '        STR R0, ColorSpace',
        '',
        '; Initialize weights (scaled by 100)',
        f'        MOV R0, #{int(hsv_weights[0] * 100)}',
        '        STR R0, WeightH',
        f'        MOV R0, #{int(hsv_weights[1] * 100)}',
        '        STR R0, WeightS',
        f'        MOV R0, #{int(hsv_weights[2] * 100)}',
        '        STR R0, WeightV',
        '',
        '; Enable keyboard with all keys',
        '        MOV R0, #7                 ; bit 0=interrupts, bit 1=arrows, bit 2=all',
        '        STR R0, .KeyboardMask',
        '',
        '; Setup keyboard interrupt handler',
        '        MOV R0, #KeyHandler',
        '        STR R0, .KeyboardISR',
        '',
        '; Enable interrupts',
        '        MOV R0, #1',
        '        STR R0, .InterruptRegister',
        '',
        '; Print instructions',
        '        MOV R0, #WelcomeMsg',
        '        STR R0, .WriteString',
        '',
        '; Render initial frame (HSV)',
        '        BL RenderHSV',
        '',
        '; Main loop - wait for interrupts',
        'MainLoop:',
        '        B MainLoop',
        '',
    ])
    
    # === Keyboard interrupt handler ===
    lines.extend([
        '; === Keyboard Handler ===',
        'KeyHandler:',
        '        PUSH {R0-R7, LR}',
        '        LDR R0, .LastKeyAndReset',
        '',
        '; T/t - Toggle color space',
        '        CMP R0, #84               ; "T"',
        '        BEQ ToggleSpace',
        '        CMP R0, #116              ; "t"',
        '        BEQ ToggleSpace',
        '',
        '; W/w - Enter weights',
        '        CMP R0, #87               ; "W"',
        '        BEQ EnterWeights',
        '        CMP R0, #119              ; "w"',
        '        BEQ EnterWeights',
        '',
        '; P/p - Print weights',
        '        CMP R0, #80               ; "P"',
        '        BEQ PrintWeights',
        '        CMP R0, #112              ; "p"',
        '        BEQ PrintWeights',
        '',
        '; R/r - Reset',
        '        CMP R0, #82               ; "R"',
        '        BEQ ResetWeights',
        '        CMP R0, #114              ; "r"',
        '        BEQ ResetWeights',
        '',
        '        B KeyDone',
        '',
    ])
    
    # Toggle color space
    lines.extend([
        'ToggleSpace:',
        '        LDR R0, ColorSpace',
        '        CMP R0, #0',
        '        BEQ SwitchToHSL',
        '',
        '; Switch to HSV',
        '        MOV R0, #0',
        '        STR R0, ColorSpace',
        f'        MOV R0, #{int(hsv_weights[0] * 100)}',
        '        STR R0, WeightH',
        f'        MOV R0, #{int(hsv_weights[1] * 100)}',
        '        STR R0, WeightS',
        f'        MOV R0, #{int(hsv_weights[2] * 100)}',
        '        STR R0, WeightV',
        '        MOV R0, #HSVMsg',
        '        STR R0, .WriteString',
        '        BL RenderHSV',
        '        B KeyDone',
        '',
        'SwitchToHSL:',
        '        MOV R0, #1',
        '        STR R0, ColorSpace',
        f'        MOV R0, #{int(hsl_weights[0] * 100)}',
        '        STR R0, WeightH',
        f'        MOV R0, #{int(hsl_weights[1] * 100)}',
        '        STR R0, WeightS',
        f'        MOV R0, #{int(hsl_weights[2] * 100)}',
        '        STR R0, WeightV',
        '        MOV R0, #HSLMsg',
        '        STR R0, .WriteString',
        '        BL RenderHSL',
        '        B KeyDone',
        '',
    ])
    
    # Enter weights via console
    lines.extend([
        'EnterWeights:',
        '; Prompt for H weight',
        '        MOV R0, #PromptH',
        '        STR R0, .WriteString',
        '        LDR R0, .InputNum',
        '        STR R0, WeightH',
        '',
        '; Prompt for S weight',
        '        MOV R0, #PromptS',
        '        STR R0, .WriteString',
        '        LDR R0, .InputNum',
        '        STR R0, WeightS',
        '',
        '; Prompt for V/L weight',
        '        MOV R0, #PromptV',
        '        STR R0, .WriteString',
        '        LDR R0, .InputNum',
        '        STR R0, WeightV',
        '',
        '; Print confirmation',
        '        MOV R0, #WeightsSetMsg',
        '        STR R0, .WriteString',
        '        BL PrintCurrentWeights',
        '',
        '; Note: Image will not update (pre-rendered)',
        '        MOV R0, #NoUpdateMsg',
        '        STR R0, .WriteString',
        '',
        '        B KeyDone',
        '',
    ])
    
    # Print weights
    lines.extend([
        'PrintWeights:',
        '        BL PrintCurrentWeights',
        '        B KeyDone',
        '',
        'PrintCurrentWeights:',
        '        PUSH {LR}',
        '; Print color space',
        '        LDR R0, ColorSpace',
        '        CMP R0, #0',
        '        BEQ PrintHSVLabel',
        '        MOV R0, #HSLLabel',
        '        B PrintLabel',
        'PrintHSVLabel:',
        '        MOV R0, #HSVLabel',
        'PrintLabel:',
        '        STR R0, .WriteString',
        '',
        '; Print weights',
        '        MOV R0, #WeightsLabel',
        '        STR R0, .WriteString',
        '        LDR R0, WeightH',
        '        STR R0, .WriteSignedNum',
        '        MOV R0, #CommaMsg',
        '        STR R0, .WriteString',
        '        LDR R0, WeightS',
        '        STR R0, .WriteSignedNum',
        '        MOV R0, #CommaMsg',
        '        STR R0, .WriteString',
        '        LDR R0, WeightV',
        '        STR R0, .WriteSignedNum',
        '        MOV R0, #NewlineMsg',
        '        STR R0, .WriteString',
        '',
        '; Print Python command hint',
        '        MOV R0, #PythonHint',
        '        STR R0, .WriteString',
        '',
        '        POP {LR}',
        '        RET',
        '',
    ])
    
    # Reset weights
    lines.extend([
        'ResetWeights:',
        '        LDR R0, ColorSpace',
        '        CMP R0, #0',
        '        BEQ ResetHSV',
        '',
        '; Reset HSL',
        f'        MOV R0, #{int(hsl_weights[0] * 100)}',
        '        STR R0, WeightH',
        f'        MOV R0, #{int(hsl_weights[1] * 100)}',
        '        STR R0, WeightS',
        f'        MOV R0, #{int(hsl_weights[2] * 100)}',
        '        STR R0, WeightV',
        '        B ResetDone',
        '',
        'ResetHSV:',
        f'        MOV R0, #{int(hsv_weights[0] * 100)}',
        '        STR R0, WeightH',
        f'        MOV R0, #{int(hsv_weights[1] * 100)}',
        '        STR R0, WeightS',
        f'        MOV R0, #{int(hsv_weights[2] * 100)}',
        '        STR R0, WeightV',
        '',
        'ResetDone:',
        '        MOV R0, #ResetMsg',
        '        STR R0, .WriteString',
        '        BL PrintCurrentWeights',
        '        B KeyDone',
        '',
    ])
    
    # Key handler exit
    lines.extend([
        'KeyDone:',
        '        POP {R0-R7, LR}',
        '        RFE',
        '',
    ])
    
    # === Render functions ===
    lines.extend([
        '; === Render HSV image ===',
        'RenderHSV:',
        '        PUSH {R0-R3, LR}',
        '        MOV R1, #.PixelScreen',
        '        MOV R2, #ImageDataHSV',
        f'        MOV R3, #{total_pixels}',
        '',
        'RenderLoopHSV:',
        '        LDR R0, [R2]',
        '        STR R0, [R1]',
        '        ADD R1, R1, #4',
        '        ADD R2, R2, #4',
        '        SUB R3, R3, #1',
        '        CMP R3, #0',
        '        BGT RenderLoopHSV',
        '',
        '        POP {R0-R3, LR}',
        '        RET',
        '',
        '; === Render HSL image ===',
        'RenderHSL:',
        '        PUSH {R0-R3, LR}',
        '        MOV R1, #.PixelScreen',
        '        MOV R2, #ImageDataHSL',
        f'        MOV R3, #{total_pixels}',
        '',
        'RenderLoopHSL:',
        '        LDR R0, [R2]',
        '        STR R0, [R1]',
        '        ADD R1, R1, #4',
        '        ADD R2, R2, #4',
        '        SUB R3, R3, #1',
        '        CMP R3, #0',
        '        BGT RenderLoopHSL',
        '',
        '        POP {R0-R3, LR}',
        '        RET',
        '',
    ])
    
    # === Data section ===
    lines.extend([
        '; === Data ===',
        '        .DATA',
        'ColorSpace:     .WORD 0      ; 0=HSV, 1=HSL',
        'WeightH:        .WORD 0',
        'WeightS:        .WORD 0',
        'WeightV:        .WORD 0',
        '',
        'WelcomeMsg:     .ASCIZ "=== Slider Controls ===\\nT: Toggle HSV/HSL\\nW: Enter weights (x100)\\nP: Print weights\\nR: Reset\\n\\n"',
        'HSVMsg:         .ASCIZ "Switched to HSV\\n"',
        'HSLMsg:         .ASCIZ "Switched to HSL\\n"',
        'HSVLabel:       .ASCIZ "HSV "',
        'HSLLabel:       .ASCIZ "HSL "',
        'WeightsLabel:   .ASCIZ "weights (x100): "',
        'CommaMsg:       .ASCIZ ", "',
        'NewlineMsg:     .ASCIZ "\\n"',
        'PromptH:        .ASCIZ "Enter H weight (x100): "',
        'PromptS:        .ASCIZ "Enter S weight (x100): "',
        'PromptV:        .ASCIZ "Enter V/L weight (x100): "',
        'WeightsSetMsg:  .ASCIZ "Weights set to: "',
        'NoUpdateMsg:    .ASCIZ "(Note: Re-run Python with these weights to see changes)\\n"',
        'ResetMsg:       .ASCIZ "Reset to defaults.\\n"',
        'PythonHint:     .ASCIZ "Divide by 100 for Python: -w H/100,S/100,V/100\\n"',
        '',
    ])
    
    # === Image data ===
    lines.append('; === HSV Pre-rendered image data ===')
    lines.append('ImageDataHSV:')
    for hex_val in hsv_pixels:
        lines.append(f'        .WORD 0x{hex_val:06x}')
    
    lines.append('')
    lines.append('; === HSL Pre-rendered image data ===')
    lines.append('ImageDataHSL:')
    for hex_val in hsl_pixels:
        lines.append(f'        .WORD 0x{hex_val:06x}')
    
    lines.append('')
    
    # Write output
    with open(output_path, 'w') as fh:
        fh.write('\n'.join(lines))
    
    total_lines = len(lines)
    print(f"\nInteractive slider assembly written to {output_path}")
    print(f"Total lines: {total_lines}")
    print(f"\nLoad in ARMLite and use:")
    print(f"  T - Toggle between HSV and HSL")
    print(f"  W - Enter new weights via console")
    print(f"  P - Print current weights")
    print(f"  R - Reset to defaults")


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HSV/HSL slider for ARMLite sprites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ARMLite Controls:
  T         Toggle between HSV and HSL color space
  W         Enter new weights via console (as integers x100)
  P         Print current weights to console
  R         Reset to default weights

Weights are scaled by 100 (e.g., 2.7 → 270, 0.42 → 42).

Example:
  python slider.py photo.png                    # Use defaults
  python slider.py photo.png slider.s           # Specify output
  python slider.py photo.png --hsv 2.7,2.2,8.0  # Custom HSV weights
  python slider.py photo.png --hsl 0.42,0.8,1.5 # Custom HSL weights
"""
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', help='Output assembly file (default: slider.s)')
    parser.add_argument('--hsv', default='2.7,2.2,8.0', type=str, metavar='H,S,V',
                        help='HSV weights (default: 2.7,2.2,8.0)')
    parser.add_argument('--hsl', default='0.42,0.8,1.5', type=str, metavar='H,S,L',
                        help='HSL weights (default: 0.42,0.8,1.5)')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.image):
        print(f'Error: Image not found: {args.image}')
        sys.exit(1)
    
    # Parse HSV weights
    try:
        hsv_weights: tuple[float, float, float] = tuple(float(w.strip()) for w in args.hsv.split(','))  # type: ignore
    except Exception:
        print('Invalid HSV weights. Use comma-separated numbers, e.g. 2.7,2.2,8.0')
        sys.exit(1)
    if len(hsv_weights) != 3:
        print('HSV weights must contain exactly three values.')
        sys.exit(1)
    
    # Parse HSL weights
    try:
        hsl_weights: tuple[float, float, float] = tuple(float(w.strip()) for w in args.hsl.split(','))  # type: ignore
    except Exception:
        print('Invalid HSL weights. Use comma-separated numbers, e.g. 0.42,0.8,1.5')
        sys.exit(1)
    if len(hsl_weights) != 3:
        print('HSL weights must contain exactly three values.')
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = os.path.expanduser(args.output)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'slider.s')
    else:
        output_path = 'slider.s'
    
    # Load and resize image
    img = Image.open(args.image).convert('RGB')
    img = img.resize((128, 96))
    
    # Generate assembly
    generate_slider_assembly(
        img=img,
        output_path=output_path,
        hsv_weights=hsv_weights,
        hsl_weights=hsl_weights,
        image_name=os.path.basename(args.image)
    )


if __name__ == '__main__':
    main()
