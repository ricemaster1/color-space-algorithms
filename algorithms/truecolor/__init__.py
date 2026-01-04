"""True Color converter - uses full 24-bit RGB hex values, no palette quantization."""
from .src.truecolor import apply_truecolor, generate_assembly, process_image

__all__ = ['apply_truecolor', 'generate_assembly', 'process_image']
