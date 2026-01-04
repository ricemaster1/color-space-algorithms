"""Shared library modules for ARMLite algorithms."""
from .palette import ARMLITE_COLORS, ARMLITE_RGB, closest_color, color_distance
from .truecolor import (
    RESOLUTIONS,
    rgb_to_hex,
    hex_to_rgb,
    generate_truecolor_assembly,
    generate_truecolor_assembly_optimized,
)

__all__ = [
    # Palette-based (147 named colors)
    'ARMLITE_COLORS',
    'ARMLITE_RGB',
    'closest_color',
    'color_distance',
    # True color (full 24-bit RGB)
    'RESOLUTIONS',
    'rgb_to_hex',
    'hex_to_rgb',
    'generate_truecolor_assembly',
    'generate_truecolor_assembly_optimized',
]
