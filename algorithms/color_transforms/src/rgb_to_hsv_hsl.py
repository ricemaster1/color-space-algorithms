from __future__ import annotations

from PIL import Image
import argparse
import colorsys
from datetime import datetime
import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Tuple

# Try numpy for fast auto-matching
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.palette import ARMLITE_RGB, ARMLITE_COLORS, closest_color


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*(channel / 255.0 for channel in rgb))


def _rgb_to_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*(channel / 255.0 for channel in rgb))
    return (h, s, l)


def _weighted_distance(weight_h: float, weight_s: float, weight_v: float):
    def metric(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dh = min(abs(a[0] - b[0]), 1.0 - abs(a[0] - b[0]))
        ds = abs(a[1] - b[1])
        dv = abs(a[2] - b[2])
        return math.sqrt((weight_h * dh) ** 2 + (weight_s * ds) ** 2 + (weight_v * dv) ** 2)

    return metric


def _palette_space(palette: dict[str, tuple[int, int, int]], transform: Callable[[tuple[int, int, int]], tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
    return {name: transform(rgb) for name, rgb in palette.items()}


def auto_match_weights(img: Image.Image, space: str = 'hsv') -> tuple[float, float, float]:
    """Auto-optimize weights to minimize RGB error between original and quantized.
    
    Uses numpy vectorized grid search for speed.
    Returns optimal (h_weight, s_weight, v_weight) tuple.
    """
    if not HAS_NUMPY:
        print("Warning: numpy not available, using default weights")
        return (2.7, 2.2, 8.0) if space == 'hsv' else (0.42, 0.8, 1.5)
    
    transform = _rgb_to_hsv if space == 'hsv' else _rgb_to_hsl
    palette = _palette_space(ARMLITE_RGB, transform)
    
    # Build numpy arrays for vectorized computation
    palette_names = list(palette.keys())
    palette_hsv = np.array([palette[n] for n in palette_names])  # (147, 3)
    palette_rgb = np.array([ARMLITE_RGB[n] for n in palette_names])  # (147, 3)
    
    # Convert image to numpy and transform to color space (vectorized)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    
    r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    
    # Hue calculation (common to HSV and HSL)
    hue = np.zeros_like(maxc)
    mask_r = (maxc == r) & (delta != 0)
    mask_g = (maxc == g) & (delta != 0)
    mask_b = (maxc == b) & (delta != 0)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    hue = hue / 6.0
    hue[hue < 0] += 1.0
    
    if space == 'hsv':
        sat = np.where(maxc != 0, delta / maxc, 0)
        val = maxc
        px_space = np.stack([hue, sat, val], axis=-1).reshape(-1, 3)
    else:
        light = (maxc + minc) / 2.0
        sat = np.zeros_like(light)
        mask = delta != 0
        sat[mask & (light <= 0.5)] = delta[mask & (light <= 0.5)] / (maxc[mask & (light <= 0.5)] + minc[mask & (light <= 0.5)])
        sat[mask & (light > 0.5)] = delta[mask & (light > 0.5)] / (2.0 - maxc[mask & (light > 0.5)] - minc[mask & (light > 0.5)])
        px_space = np.stack([hue, sat, light], axis=-1).reshape(-1, 3)
    
    orig_rgb = (img_arr * 255).reshape(-1, 3).astype(np.int16)  # (N, 3)
    n_pixels = len(orig_rgb)
    
    # Pre-compute differences (constant across all weight combos)
    dh_raw = np.abs(px_space[:, None, 0] - palette_hsv[None, :, 0])
    dh2 = np.minimum(dh_raw, 1.0 - dh_raw) ** 2  # Hue wrapping + square
    ds2 = (px_space[:, None, 1] - palette_hsv[None, :, 1]) ** 2
    dv2 = (px_space[:, None, 2] - palette_hsv[None, :, 2]) ** 2
    
    def evaluate_weights(weights_arr):
        """Evaluate all weight combinations, return errors array."""
        weights_sq = weights_arr ** 2
        n_weights = len(weights_arr)
        errors = np.zeros(n_weights, dtype=np.float64)
        chunk_size = 3000
        
        for start in range(0, n_pixels, chunk_size):
            end = min(start + chunk_size, n_pixels)
            chunk_dh2 = dh2[start:end]
            chunk_ds2 = ds2[start:end]
            chunk_dv2 = dv2[start:end]
            chunk_rgb = orig_rgb[start:end]
            
            dist2 = (weights_sq[:, 0, None, None] * chunk_dh2[None, :, :] +
                     weights_sq[:, 1, None, None] * chunk_ds2[None, :, :] +
                     weights_sq[:, 2, None, None] * chunk_dv2[None, :, :])
            
            best_idx = np.argmin(dist2, axis=2)
            matched_rgb = palette_rgb[best_idx]
            rgb_diff = chunk_rgb[None, :, :].astype(np.int16) - matched_rgb.astype(np.int16)
            errors += np.sum(rgb_diff.astype(np.float64) ** 2, axis=(1, 2))
        
        return errors
    
    # === STAGE 1: Coarse search (125 combos) ===
    h_coarse = np.array([0.5, 1.5, 2.5, 4.0, 5.0], dtype=np.float32)
    s_coarse = np.array([0.5, 1.5, 2.5, 4.0, 5.0], dtype=np.float32)
    v_coarse = np.array([1.0, 2.0, 4.0, 6.0, 10.0], dtype=np.float32)
    
    wh, ws, wv = np.meshgrid(h_coarse, s_coarse, v_coarse, indexing='ij')
    weights_coarse = np.stack([wh.ravel(), ws.ravel(), wv.ravel()], axis=1)
    
    errors_coarse = evaluate_weights(weights_coarse)
    best_coarse_idx = np.argmin(errors_coarse)
    best_coarse = weights_coarse[best_coarse_idx]
    
    # === STAGE 2: Fine search around best (125 combos with tighter spacing) ===
    h_fine = np.clip(best_coarse[0] + np.array([-0.5, -0.25, 0, 0.25, 0.5]), 0.1, 10.0).astype(np.float32)
    s_fine = np.clip(best_coarse[1] + np.array([-0.5, -0.25, 0, 0.25, 0.5]), 0.1, 10.0).astype(np.float32)
    v_fine = np.clip(best_coarse[2] + np.array([-1.0, -0.5, 0, 0.5, 1.0]), 0.1, 10.0).astype(np.float32)
    
    wh, ws, wv = np.meshgrid(h_fine, s_fine, v_fine, indexing='ij')
    weights_fine = np.stack([wh.ravel(), ws.ravel(), wv.ravel()], axis=1)
    
    errors_fine = evaluate_weights(weights_fine)
    best_fine_idx = np.argmin(errors_fine)
    best_w = weights_fine[best_fine_idx].tolist()
    best_error = errors_fine[best_fine_idx]
    
    avg_error = math.sqrt(best_error / n_pixels)
    print(f"Auto-matched {space.upper()} weights: ({best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}) - Avg RGB error: {avg_error:.1f}")
    return tuple(best_w)  # type: ignore


def apply_rgb_to_hsv_hsl(
    img: Image.Image,
    space: str = 'hsv',
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[list[str]]:
    width, height = img.size
    transform = _rgb_to_hsv if space == 'hsv' else _rgb_to_hsl
    metric = _weighted_distance(*weights)

    palette_space = _palette_space(ARMLITE_RGB, transform)
    grid: list[list[str]] = []

    for y in range(height):
        row: list[str] = []
        for x in range(width):
            rgb = img.getpixel((x, y))
            converted = transform(rgb)
            best_name = 'black'
            best_distance = float('inf')
            for name, target in palette_space.items():
                dist = metric(converted, target)
                if dist < best_distance:
                    best_distance = dist
                    best_name = name
            row.append(best_name)
        grid.append(row)
    return grid


def generate_assembly(color_grid, output_path, *, image_path: str = '', space: str = '', weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    height = len(color_grid)
    width = len(color_grid[0]) if height else 0
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '; === Fullscreen Sprite ===',
        f'; Generated: {timestamp}',
        f'; Source: {os.path.basename(image_path)}' if image_path else '',
        f'; Color space: {space.upper()} weights=({weights[0]}, {weights[1]}, {weights[2]})' if space else '',
        '    MOV R0, #2',
        '    STR R0, .Resolution',
        '    MOV R1, #.PixelScreen',
        '    MOV R6, #512 ; row stride (128 * 4)'
    ]
    # Filter out empty comment lines
    lines = [line for line in lines if line]
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


def process_image(image_path, output_path, space: str, weights: tuple[float, float, float], auto: bool = False):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    
    # Auto-match weights if requested
    if auto:
        weights = auto_match_weights(img, space)
    
    grid = apply_rgb_to_hsv_hsl(img, space=space, weights=weights)
    generate_assembly(grid, output_path, image_path=image_path, space=space, weights=weights)
    return weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RGB→HSV/HSL workflow renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', default='converted.s', help='Output assembly file path (default: converted.s)')
    parser.add_argument('-s','--space', choices=['hsv', 'hsl'], default='hsv', help='Color space to use for matching (default: hsv)')
    
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument('-w', '--weights', default=None, type=str, metavar='W1,W2,W3',
        help='Comma-separated weights for hue,sat,value/lightness matching. For HSV, recommended default is 2.7,2.2,8 and for HSL it is 0.42,0.8,1.5. Negative values are allowed. Setting all weights to 0,0,0 will produce a solid color.')
    weight_group.add_argument('-a', '--auto', action='store_true',
        help='Auto-match weights to minimize RGB error (requires numpy)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    # Determine weights, allowing negative values and proper defaults for HSL
    weights: tuple[float, float, float]
    if args.auto:
        # Will be computed later in process_image
        weights = (0.0, 0.0, 0.0)  # placeholder
    elif args.weights is not None:
        try:
            weights = tuple(float(w.strip()) for w in args.weights.split(','))  # type: ignore[assignment]
        except Exception:
            print('Invalid weights. Use comma-separated numbers, e.g. 1,1,0.5 or -1,1,1')
            sys.exit(1)
        if len(weights) != 3:
            print('Weights must contain exactly three values.')
            sys.exit(1)
    else:
        weights = (2.7, 2.2, 8.0) if args.space == 'hsv' else (0.42, 0.8, 1.5)

    # Process image (auto-match will compute weights if --auto)
    img_for_weights = Image.open(args.image).convert('RGB').resize((128, 96))
    if args.auto:
        weights = auto_match_weights(img_for_weights, args.space)

    # Build default filename from color space and weights
    default_filename = f'{args.space}_{weights[0]}_{weights[1]}_{weights[2]}.s'
    
    # If output is a directory, save with weight-based filename inside it
    output_path = args.output
    if output_path:
        output_path = os.path.expanduser(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, default_filename)
        else:
            # If parent directory doesn't exist, error out
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                print(f'Output directory does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = default_filename
    
    process_image(args.image, output_path, args.space, weights)
