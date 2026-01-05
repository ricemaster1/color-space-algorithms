from __future__ import annotations

from PIL import Image
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

# Ensure lib is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib import ARMLITE_RGB
from distance_ciede2000 import rgb_to_lab, delta_e_ciede2000
from distance_cie94 import delta_e_cie94
from distance_cie76 import delta_e_cie76


class NeoDeltaEQuantizer:
    def __init__(
        self,
        metric: str = 'adaptive',
        application: str = 'graphic_arts',
        chroma_threshold: float = 10.0,
        luminance_threshold: float = 12.0,
        hybrid_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
    ) -> None:
        self.metric = metric.lower()
        self.application = application
        self.chroma_threshold = chroma_threshold
        self.luminance_threshold = luminance_threshold
        self.hybrid_weights = hybrid_weights
        self._cache: dict[tuple[int, int, int], str] = {}
        self._palette = self._prepare_palette()

    def _prepare_palette(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for name, rgb in ARMLITE_RGB.items():
            lab = rgb_to_lab(rgb)
            chroma = math.hypot(lab[1], lab[2])
            entries.append({'name': name, 'lab': lab, 'chroma': chroma})
        return entries

    def _ensure_supported(self) -> None:
        allowed = {'adaptive', 'ciede2000', 'cie94', 'cie76', '2000', '94', '76', 'hybrid'}
        if self.metric not in allowed:
            raise ValueError(f"Unknown metric '{self.metric}'")

    def _delta(self, lab_src, entry) -> float:
        dest_lab = entry['lab']  # type: ignore[index]
        metric = self.metric
        if metric == 'adaptive':
            return self._adaptive_delta(lab_src, entry)
        if metric in {'ciede2000', '2000'}:
            return delta_e_ciede2000(lab_src, dest_lab)
        if metric in {'cie94', '94'}:
            return delta_e_cie94(
                lab_src,
                dest_lab,
                application='textiles' if self.application == 'textiles' else 'graphic_arts',
            )
        if metric in {'cie76', '76'}:
            return delta_e_cie76(lab_src, dest_lab)
        if metric == 'hybrid':
            de2000 = delta_e_ciede2000(lab_src, dest_lab)
            de94 = delta_e_cie94(
                lab_src,
                dest_lab,
                application='textiles' if self.application == 'textiles' else 'graphic_arts',
            )
            de76 = delta_e_cie76(lab_src, dest_lab)
            w0, w1, w2 = self.hybrid_weights
            return w0 * de2000 + w1 * de94 + w2 * de76
        raise ValueError(f"Unknown metric '{metric}'")

    def _adaptive_delta(self, lab_src, entry) -> float:
        dest_lab = entry['lab']  # type: ignore[index]
        src_chroma = math.hypot(lab_src[1], lab_src[2])
        dest_chroma = entry['chroma']  # type: ignore[index]
        delta2000 = delta_e_ciede2000(lab_src, dest_lab)
        delta94 = delta_e_cie94(
            lab_src,
            dest_lab,
            application='textiles' if self.application == 'textiles' else 'graphic_arts',
        )
        delta76 = delta_e_cie76(lab_src, dest_lab)

        if src_chroma <= self.chroma_threshold and dest_chroma <= self.chroma_threshold:
            return delta2000
        luminance_gap = abs(lab_src[0] - dest_lab[0])
        if luminance_gap > self.luminance_threshold:
            return 0.55 * delta2000 + 0.45 * delta94
        w0, w1, w2 = self.hybrid_weights
        return w0 * delta2000 + w1 * delta94 + w2 * delta76

    def map_color(self, rgb: tuple[int, int, int]) -> str:
        cached = self._cache.get(rgb)
        if cached is not None:
            return cached
        self._ensure_supported()
        lab_src = rgb_to_lab(rgb)
        best_entry = min(self._palette, key=lambda entry: self._delta(lab_src, entry))
        name = best_entry['name']  # type: ignore[index]
        self._cache[rgb] = name  # type: ignore[assignment]
        return name


def apply_distance_delta_e_neo(
    img: Image.Image,
    metric: str = 'adaptive',
    application: str = 'graphic_arts',
    chroma_threshold: float = 10.0,
    luminance_threshold: float = 12.0,
    hybrid_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> list[list[str]]:
    width, height = img.size
    quantizer = NeoDeltaEQuantizer(
        metric=metric,
        application=application,
        chroma_threshold=chroma_threshold,
        luminance_threshold=luminance_threshold,
        hybrid_weights=hybrid_weights,
    )
    pixels = list(img.getdata())
    grid: list[list[str]] = []
    idx = 0
    for _ in range(height):
        row: list[str] = []
        for _ in range(width):
            row.append(quantizer.map_color(pixels[idx]))
            idx += 1
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


def _parse_weights(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('hybrid weights must have three comma-separated values')
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('hybrid weights must be numeric') from exc
    total = sum(values)
    if total <= 0:
        raise argparse.ArgumentTypeError('hybrid weights must sum to a positive value')
    return tuple(value / total for value in values)  # type: ignore[return-value]


def process_image(
    image_path: str,
    output_path: str,
    metric: str,
    application: str,
    chroma_threshold: float,
    luminance_threshold: float,
    hybrid_weights: Tuple[float, float, float],
) -> None:
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 96))
    grid = apply_distance_delta_e_neo(
        img,
        metric=metric,
        application=application,
        chroma_threshold=chroma_threshold,
        luminance_threshold=luminance_threshold,
        hybrid_weights=hybrid_weights,
    )
    generate_assembly(grid, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Neo Delta E renderer for ARMLite sprites'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('output', nargs='?', default='converted.s', help='Output assembly file path (default: converted.s)')
    parser.add_argument(
        '-m', '--metric',
        default='adaptive',
        choices=['adaptive', 'ciede2000', 'cie94', 'cie76', 'hybrid', '2000', '94', '76'],
        help='Delta E metric selection (default: adaptive)'
    )
    parser.add_argument(
        '-ca', '--cie94-application',
        default='graphic_arts',
        choices=['graphic_arts', 'textiles'],
        help='Application weighting for CIE94-based calculations'
    )
    parser.add_argument(
        '-ct', '--chroma-threshold',
        type=float,
        default=10.0,
        help='Chroma threshold for adaptive blending (default: 10.0)'
    )
    parser.add_argument(
        '-lt', '--luminance-threshold',
        type=float,
        default=12.0,
        help='Luminance gap threshold for adaptive blending (default: 12.0)'
    )
    parser.add_argument(
        '-hw', '--hybrid-weights',
        type=_parse_weights,
        default=(0.6, 0.3, 0.1),
        help='Comma-separated weights for (CIEDE2000,CIE94,CIE76) when using hybrid metrics'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Image not found.')
        sys.exit(1)

    output_path = args.output
    if output_path:
        output_path = os.path.expanduser(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'converted.s')
        else:
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                print(f'Output directory does not exist: {parent}')
                sys.exit(1)
    else:
        output_path = 'converted.s'

    process_image(
        args.image,
        args.output,
        args.metric,
        args.cie94_application,
        args.chroma_threshold,
        args.luminance_threshold,
        args.hybrid_weights,
    )
