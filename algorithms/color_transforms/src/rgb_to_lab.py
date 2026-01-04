from __future__ import annotations

from PIL import Image
import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

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

from lib import ARMLITE_RGB


# === Color Space Transforms ===

def _srgb_to_linear(channel: float) -> float:
    """Convert sRGB channel (0-255) to linear RGB (0-1)."""
    channel /= 255.0
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _rgb_to_xyz(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert sRGB to CIE XYZ (D65 illuminant)."""
    r_lin = _srgb_to_linear(rgb[0])
    g_lin = _srgb_to_linear(rgb[1])
    b_lin = _srgb_to_linear(rgb[2])

    # sRGB to XYZ matrix (D65)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    return (x, y, z)


def _xyz_to_lab(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert CIE XYZ to CIE L*a*b* (D65 illuminant)."""
    # D65 reference white
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: float) -> float:
        epsilon = 0.008856  # (6/29)^3
        kappa = 903.3       # (29/3)^3
        if t > epsilon:
            return t ** (1 / 3)
        return (kappa * t + 16) / 116

    fx = f(xyz[0] / xn)
    fy = f(xyz[1] / yn)
    fz = f(xyz[2] / zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return (L, a, b)


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert sRGB to CIE L*a*b* (D65 illuminant)."""
    return _xyz_to_lab(_rgb_to_xyz(rgb))


# === Distance Metrics ===

def _weighted_lab_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    weights: tuple[float, float, float]
) -> float:
    """Compute weighted Euclidean distance in L*a*b* space."""
    dL = (a[0] - b[0]) * weights[0]
    da = (a[1] - b[1]) * weights[1]
    db = (a[2] - b[2]) * weights[2]
    return math.sqrt(dL * dL + da * da + db * db)


# === Auto-Matching ===

def auto_match_weights(img: Image.Image) -> tuple[float, float, float]:
    """Auto-optimize weights to minimize RGB error between original and quantized.
    
    Uses numpy vectorized grid search for speed.
    Returns optimal (L_weight, a_weight, b_weight) tuple.
    """
    if not HAS_NUMPY:
        print("Warning: numpy not available, using default weights (1, 1, 1)")
        return (1.0, 1.0, 1.0)
    
    # Build numpy arrays for palette
    palette_names = list(ARMLITE_RGB.keys())
    palette_rgb = np.array([ARMLITE_RGB[n] for n in palette_names], dtype=np.float32)
    palette_lab = np.array([_rgb_to_lab(tuple(rgb)) for rgb in ARMLITE_RGB.values()], dtype=np.float32)
    
    # Convert image to numpy
    img_arr = np.array(img, dtype=np.float32)
    h, w = img_arr.shape[:2]
    n_pixels = h * w
    
    # Convert all pixels to L*a*b*
    img_flat = img_arr.reshape(-1, 3).astype(np.uint8)
    img_lab = np.array([_rgb_to_lab(tuple(px)) for px in img_flat], dtype=np.float32)
    
    def evaluate_weights(weight_array: np.ndarray) -> np.ndarray:
        """Evaluate total RGB error for multiple weight configurations."""
        errors = np.zeros(len(weight_array))
        for i, weights in enumerate(weight_array):
            # Compute weighted distances to all palette colors
            diff = img_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
            weighted_diff = diff * weights
            distances = np.sqrt(np.sum(weighted_diff ** 2, axis=2))
            
            # Find best match for each pixel
            best_indices = np.argmin(distances, axis=1)
            matched_rgb = palette_rgb[best_indices]
            
            # Compute RGB error
            rgb_error = np.sum((img_flat.astype(np.float32) - matched_rgb) ** 2)
            errors[i] = rgb_error
        return errors
    
    # Coarse grid search
    L_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    a_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    b_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    
    L_grid, a_grid, b_grid = np.meshgrid(L_vals, a_vals, b_vals, indexing='ij')
    weights_coarse = np.stack([L_grid.ravel(), a_grid.ravel(), b_grid.ravel()], axis=1)
    
    errors_coarse = evaluate_weights(weights_coarse)
    best_coarse_idx = np.argmin(errors_coarse)
    best_coarse = weights_coarse[best_coarse_idx]
    
    # Fine grid search around best coarse result
    L_fine = np.linspace(max(0.25, best_coarse[0] - 0.5), best_coarse[0] + 0.5, 5)
    a_fine = np.linspace(max(0.25, best_coarse[1] - 0.5), best_coarse[1] + 0.5, 5)
    b_fine = np.linspace(max(0.25, best_coarse[2] - 0.5), best_coarse[2] + 0.5, 5)
    
    L_grid, a_grid, b_grid = np.meshgrid(L_fine, a_fine, b_fine, indexing='ij')
    weights_fine = np.stack([L_grid.ravel(), a_grid.ravel(), b_grid.ravel()], axis=1)
    
    errors_fine = evaluate_weights(weights_fine)
    best_fine_idx = np.argmin(errors_fine)
    best_w = weights_fine[best_fine_idx].tolist()
    best_error = errors_fine[best_fine_idx]
    
    avg_error = math.sqrt(best_error / n_pixels)
    print(f"Auto-matched L*a*b* weights: ({best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}) - Avg RGB error: {avg_error:.1f}")
    return tuple(best_w)  # type: ignore


# === Core Processing ===

def apply_rgb_to_lab(
    img: Image.Image,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> list[list[str]]:
    """
    Convert image to color grid using L*a*b* color space matching.
    
    Args:
        img: PIL Image in RGB mode
        weights: Weights for (L*, a*, b*) matching
    
    Returns:
        2D grid of palette color names (147 colors)
    """
    width, height = img.size
    palette_lab = {name: _rgb_to_lab(rgb) for name, rgb in ARMLITE_RGB.items()}
    grid: list[list[str]] = []

    for y in range(height):
        row: list[str] = []
        for x in range(width):
            px = img.getpixel((x, y))
            if isinstance(px, int):
                px = (px, px, px)
            lab_src = _rgb_to_lab((px[0], px[1], px[2]))
            best_name = 'black'
            best_distance = float('inf')
            for name, lab in palette_lab.items():
                dist = _weighted_lab_distance(lab_src, lab, weights)
                if dist < best_distance:
                    best_distance = dist
                    best_name = name
            row.append(best_name)
        grid.append(row)
    return grid


def generate_assembly(
    color_grid: list[list[str]],
    output_path: str,
    *,
    image_path: str = '',
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Generate ARMlite assembly file with 147-color palette.
    
    Args:
        color_grid: 2D grid of color names
        output_path: Path to write assembly file
        image_path: Original image path (for comments)
        weights: Weights used for L*a*b* matching
    """
    height = len(color_grid)
    width = len(color_grid[0]) if height else 0
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    lines = [
        '; === Fullscreen Sprite (147-color palette) ===',
        f'; Generated: {timestamp}',
        f'; Source: {os.path.basename(image_path)}' if image_path else '',
        f'; Color space: L*a*b* weights=({weights[0]}, {weights[1]}, {weights[2]})',
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
    print(f"Palette assembly written to {output_path}")


def process_image(
    image_path: str,
    output_path: str,
    weights: tuple[float, float, float],
    auto: bool = False,
) -> tuple[float, float, float]:
    """
    Process an image and generate ARMlite assembly with 147-color palette.
    
    Args:
        image_path: Path to input image
        output_path: Path to write assembly file
        weights: Weights for L*a*b* matching
        auto: Auto-optimize weights
    
    Returns:
        Final weights used (may differ from input if auto=True)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    
    # Auto-match weights if requested
    if auto:
        weights = auto_match_weights(img)
    
    grid = apply_rgb_to_lab(img, weights=weights)
    generate_assembly(grid, output_path, image_path=image_path, weights=weights)
    return weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RGB→L*a*b* palette quantizer for ARMLite sprites. Converts images to 147-color CSS3 palette using perceptually uniform L*a*b* color space.'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', default=None, help='Output assembly file path (default: auto-generated)')
    parser.add_argument('-o', '--output-file', dest='output_file', default=None,
        help='Output assembly file path (alternative to positional argument)')
    
    # Weight options
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument('-w', '--weights', default=None, type=str, metavar='L,a,b',
        help='Comma-separated weights for L*,a*,b* channels (default: 1,1,1)')
    weight_group.add_argument('-a', '--auto', action='store_true',
        help='Auto-match weights to minimize RGB error (requires numpy)')
    
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    # Determine weights
    weights: tuple[float, float, float]
    if args.auto:
        weights = (0.0, 0.0, 0.0)  # placeholder, will be computed
    elif args.weights is not None:
        try:
            weights = tuple(float(w.strip()) for w in args.weights.split(','))  # type: ignore[assignment]
        except ValueError:
            print('Invalid weights. Use comma-separated numbers, e.g. 1,1,0.8')
            sys.exit(1)
        if len(weights) != 3:
            print('Weights must contain exactly three values.')
            sys.exit(1)
    else:
        weights = (1.0, 1.0, 1.0)

    # Auto-match weights if requested
    if args.auto:
        img_for_weights = Image.open(args.image).convert('RGB').resize((128, 96))
        weights = auto_match_weights(img_for_weights)

    # Build default filename
    default_filename = f'lab_{weights[0]}_{weights[1]}_{weights[2]}.s'
    
    # Determine output path: -o flag takes precedence over positional argument
    output_path = args.output_file if args.output_file else args.output
    if output_path:
        output_path = os.path.expanduser(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, default_filename)
        else:
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                print(f'Output directory does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = default_filename
    
    # Don't pass auto=True since we already computed weights
    process_image(args.image, output_path, weights, auto=False)
