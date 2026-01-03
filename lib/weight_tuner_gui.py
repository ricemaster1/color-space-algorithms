"""
weight_tuner_gui.py - ARMlite-style GUI for real-time HSV/HSL weight tuning

A Python tkinter application that mimics the ARMlite web interface layout
and provides live preview of weighted HSV/HSL color quantization.

Features:
- Real-time weight adjustment via sliders
- Toggle between HSV and HSL color spaces
- Original vs quantized side-by-side comparison
- Export to ARMlite assembly format (.s)
- ARMlite-inspired dark theme UI

Usage:
    python weight_tuner_gui.py [image_path]
    
If no image is provided, a file dialog will open.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import os
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from PIL import Image, ImageTk

# Ensure lib is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from palette import ARMLITE_RGB, ARMLITE_COLORS, closest_color, color_distance


# === Constants ===
# ARMlite supports multiple pixel modes:
# Mode 1: 16x12 (192 pixels)
# Mode 2: 32x24 (768 pixels)  
# Mode 3: 64x48 (3072 pixels)
# Mode 4: 128x96 (12288 pixels)
PIXEL_MODES = {
    1: (16, 12, 12),   # (width, height, scale)
    2: (32, 24, 8),
    3: (64, 48, 5),
    4: (128, 96, 3),
}
DEFAULT_MODE = 4

# Default weights (from conversation context)
HSV_DEFAULTS = (2.7, 2.2, 8.0)
HSL_DEFAULTS = (0.42, 0.8, 1.5)

# ARMlite-inspired colors
COLORS = {
    'bg_dark': '#1a1a2e',       # Main background
    'bg_panel': '#16213e',      # Panel background
    'bg_section': '#0f3460',    # Section background
    'text': '#ffffff',          # Main text (white for contrast)
    'text_dim': '#888888',      # Dimmed text
    'accent': '#e94560',        # Accent color (red)
    'accent2': '#00adb5',       # Secondary accent (teal)
    'border': '#1f4068',        # Border color
    'button_bg': '#4477aa',     # Button background (medium blue)
    'button_fg': '#000000',     # Button text (BLACK for macOS contrast)
    'button_hover': '#5588bb',  # Button hover
    'match_bg': '#22aa88',      # Match button (green-teal)
    'slider_trough': '#0a0a14', # Slider trough
}

# Try to import numpy for fast optimization
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# === Color space transforms ===

def rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))


def rgb_to_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))
    return (h, s, l)


def weighted_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    w: tuple[float, float, float]
) -> float:
    """Weighted distance with hue wrapping."""
    dh = min(abs(a[0] - b[0]), 1.0 - abs(a[0] - b[0]))
    ds = abs(a[1] - b[1])
    dv = abs(a[2] - b[2])
    return math.sqrt((w[0] * dh) ** 2 + (w[1] * ds) ** 2 + (w[2] * dv) ** 2)


def palette_to_space(
    transform: Callable[[tuple[int, int, int]], tuple[float, float, float]]
) -> dict[str, tuple[float, float, float]]:
    """Transform palette to HSV/HSL space."""
    return {name: transform(rgb) for name, rgb in ARMLITE_RGB.items()}


# Pre-compute transformed palettes
PALETTE_HSV = palette_to_space(rgb_to_hsv)
PALETTE_HSL = palette_to_space(rgb_to_hsl)

# Pre-compute numpy arrays for fast vectorized quantization
if HAS_NUMPY:
    # Palette names in order
    _PALETTE_NAMES = list(ARMLITE_RGB.keys())
    
    # HSV palette as (N, 3) array
    _PALETTE_HSV_NP = np.array([PALETTE_HSV[n] for n in _PALETTE_NAMES], dtype=np.float32)
    # HSL palette as (N, 3) array  
    _PALETTE_HSL_NP = np.array([PALETTE_HSL[n] for n in _PALETTE_NAMES], dtype=np.float32)
    # RGB palette as (N, 3) array for output
    _PALETTE_RGB_NP = np.array([ARMLITE_RGB[n] for n in _PALETTE_NAMES], dtype=np.uint8)


class ARMliteStyleApp:
    """Main application window with ARMlite-inspired styling."""
    
    def __init__(self, root: tk.Tk, image_path: Optional[str] = None):
        self.root = root
        self.root.title("ARMlite Weight Tuner")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.geometry("1000x750")
        
        # State
        self.source_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.current_space = 'hsv'
        self.weights = list(HSV_DEFAULTS)
        self.image_path = ""
        self.pixel_mode = DEFAULT_MODE
        
        # Photo references (prevent garbage collection)
        self._original_photo: Optional[ImageTk.PhotoImage] = None
        self._quantized_photo: Optional[ImageTk.PhotoImage] = None
        
        # Debounce timer for slider updates
        self._update_timer: Optional[str] = None
        
        # Caches for performance
        # Auto-match cache: (image_id, color_space) -> (weights, avg_error)
        self._match_cache: dict[tuple[int, str], tuple[list[float], float]] = {}
        # Quantized image cache: (image_id, color_space, weights_tuple) -> PIL.Image
        self._quant_cache: dict[tuple[int, str, tuple[float, ...]], Image.Image] = {}
        self._image_id: int = 0  # Incremented on each new image load
        
        self._setup_styles()
        self._create_layout()
        self._bind_shortcuts()
        
        if image_path:
            self._load_image(image_path)
    
    @property
    def img_width(self) -> int:
        return PIXEL_MODES[self.pixel_mode][0]
    
    @property
    def img_height(self) -> int:
        return PIXEL_MODES[self.pixel_mode][1]
    
    @property
    def display_scale(self) -> int:
        return PIXEL_MODES[self.pixel_mode][2]
    
    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind('<Control-o>', lambda e: self._browse_image())
        self.root.bind('<Control-s>', lambda e: self._export_assembly())
        self.root.bind('<r>', lambda e: self._reset_weights())
        self.root.bind('<R>', lambda e: self._reset_weights())
        for mode in PIXEL_MODES:
            self.root.bind(str(mode), lambda e, m=mode: self._set_pixel_mode(m))
    
    def _setup_styles(self):
        """Configure ttk styles for ARMlite theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure action button style (Load, Export, Reset)
        style.configure(
            'Action.TButton',
            background=COLORS['button_bg'],
            foreground=COLORS['button_fg'],
            borderwidth=2,
            focuscolor='none',
            padding=(12, 6),
            font=('Helvetica', 11, 'bold')
        )
        style.map('Action.TButton',
            background=[('active', COLORS['button_hover']), ('pressed', COLORS['button_hover'])],
            foreground=[('active', COLORS['button_fg'])]
        )
        
        # Configure match button style (green/teal)
        style.configure(
            'Match.TButton',
            background=COLORS['match_bg'],
            foreground=COLORS['button_fg'],
            borderwidth=2,
            focuscolor='none',
            padding=(12, 6),
            font=('Helvetica', 11, 'bold')
        )
        style.map('Match.TButton',
            background=[('active', COLORS['accent2']), ('pressed', COLORS['accent2'])],
            foreground=[('active', COLORS['button_fg'])]
        )
        
        # Configure scale (slider) style
        style.configure(
            'ARMlite.Horizontal.TScale',
            background=COLORS['bg_panel'],
            troughcolor=COLORS['slider_trough'],
            sliderthickness=20
        )
        
        # Configure label style
        style.configure(
            'ARMlite.TLabel',
            background=COLORS['bg_panel'],
            foreground=COLORS['text']
        )
        
        # Configure frame style
        style.configure(
            'ARMlite.TFrame',
            background=COLORS['bg_panel']
        )
    
    def _create_layout(self):
        """Create the main UI layout mimicking ARMlite."""
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top row: Image displays
        self._create_display_section(main_frame)
        
        # Middle row: Controls
        self._create_controls_section(main_frame)
        
        # Bottom row: Console output
        self._create_console_section(main_frame)
    
    def _create_panel(self, parent: tk.Frame, title: str) -> tk.Frame:
        """Create an ARMlite-style panel with title."""
        panel = tk.Frame(parent, bg=COLORS['bg_panel'], bd=1, relief=tk.SOLID)
        
        # Title bar
        title_bar = tk.Frame(panel, bg=COLORS['bg_section'], height=25)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        
        title_label = tk.Label(
            title_bar,
            text=title,
            bg=COLORS['bg_section'],
            fg=COLORS['text'],
            font=('Segoe UI', 10, 'bold')
        )
        title_label.pack(side=tk.LEFT, padx=8, pady=3)
        
        # Content area
        content = tk.Frame(panel, bg=COLORS['bg_panel'])
        content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        return panel, content
    
    def _create_display_section(self, parent: tk.Frame):
        """Create the image display section."""
        display_frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        canvas_w = self.img_width * self.display_scale
        canvas_h = self.img_height * self.display_scale
        
        # Original image panel
        orig_panel, orig_content = self._create_panel(display_frame, "Original (scaled)")
        orig_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(
            orig_content,
            width=canvas_w,
            height=canvas_h,
            bg='#000000',
            highlightthickness=0
        )
        self.original_canvas.pack(pady=5)
        
        # Original image info label
        self.original_info_label = tk.Label(
            orig_content,
            text=f"No image loaded",
            bg=COLORS['bg_panel'],
            fg=COLORS['text_dim'],
            font=('Consolas', 9)
        )
        self.original_info_label.pack()
        
        # Quantized image panel (mimics ARMlite pixel display)
        quant_panel, quant_content = self._create_panel(display_frame, "Input/Output (ARMlite Preview)")
        quant_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create pixel grid frame
        self.pixel_frame = tk.Frame(
            quant_content,
            bg='#000000',
            width=canvas_w,
            height=canvas_h
        )
        self.pixel_frame.pack(pady=5)
        self.pixel_frame.pack_propagate(False)
        
        self.quantized_canvas = tk.Canvas(
            self.pixel_frame,
            width=canvas_w,
            height=canvas_h,
            bg='#000000',
            highlightthickness=0
        )
        self.quantized_canvas.pack()
        
        # Pixel info label
        self.pixel_count_label = tk.Label(
            quant_content,
            text=f"Mode {self.pixel_mode}: {self.img_width}x{self.img_height} ({self.img_width * self.img_height} px)",
            bg=COLORS['bg_panel'],
            fg=COLORS['text_dim'],
            font=('Consolas', 9)
        )
        self.pixel_count_label.pack()
    
    def _create_controls_section(self, parent: tk.Frame):
        """Create the weight sliders and control buttons."""
        controls_panel, controls_content = self._create_panel(parent, "Processor (Weight Controls)")
        controls_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Top row: Load button and color space toggle
        button_frame = tk.Frame(controls_content, bg=COLORS['bg_panel'])
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load button
        load_btn = ttk.Button(
            button_frame,
            text="Load",
            command=self._browse_image,
            style='Action.TButton'
        )
        load_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Export button
        export_btn = ttk.Button(
            button_frame,
            text="Export .s",
            command=self._export_assembly,
            style='Action.TButton'
        )
        export_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Reset button
        reset_btn = ttk.Button(
            button_frame,
            text="Reset",
            command=self._reset_weights,
            style='Action.TButton'
        )
        reset_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Match button (auto-optimize weights)
        match_btn = ttk.Button(
            button_frame,
            text="Match",
            command=self._auto_match,
            style='Match.TButton'
        )
        match_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Color space toggle
        self.space_var = tk.StringVar(value='hsv')
        
        hsv_radio = tk.Radiobutton(
            button_frame,
            text="HSV",
            variable=self.space_var,
            value='hsv',
            command=self._on_space_change,
            bg=COLORS['bg_panel'],
            fg=COLORS['text'],
            selectcolor=COLORS['bg_section'],
            activebackground=COLORS['bg_panel'],
            activeforeground=COLORS['accent']
        )
        hsv_radio.pack(side=tk.LEFT, padx=(0, 5))
        
        hsl_radio = tk.Radiobutton(
            button_frame,
            text="HSL",
            variable=self.space_var,
            value='hsl',
            command=self._on_space_change,
            bg=COLORS['bg_panel'],
            fg=COLORS['text'],
            selectcolor=COLORS['bg_section'],
            activebackground=COLORS['bg_panel'],
            activeforeground=COLORS['accent']
        )
        hsl_radio.pack(side=tk.LEFT)
        
        # Weight display label
        self.weight_display = tk.Label(
            button_frame,
            text="",
            bg=COLORS['bg_panel'],
            fg=COLORS['accent2'],
            font=('Consolas', 10, 'bold')
        )
        self.weight_display.pack(side=tk.RIGHT)
        
        # Sliders frame
        sliders_frame = tk.Frame(controls_content, bg=COLORS['bg_panel'])
        sliders_frame.pack(fill=tk.X)
        
        # Create weight sliders
        self.slider_vars = []
        self.slider_entries = []
        slider_names = ['H (Hue)', 'S (Saturation)', 'V/L (Value/Lightness)']
        
        for i, name in enumerate(slider_names):
            row = tk.Frame(sliders_frame, bg=COLORS['bg_panel'])
            row.pack(fill=tk.X, pady=3)
            
            # Label
            label = tk.Label(
                row,
                text=name,
                bg=COLORS['bg_panel'],
                fg=COLORS['text'],
                width=22,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            # Slider variable
            var = tk.DoubleVar(value=self.weights[i])
            self.slider_vars.append(var)
            
            # Slider
            slider = ttk.Scale(
                row,
                from_=0.0,
                to=10.0,
                variable=var,
                orient=tk.HORIZONTAL,
                style='ARMlite.Horizontal.TScale',
                command=lambda v, idx=i: self._on_slider_change(idx)
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            
            # Editable entry for weight value
            entry_var = tk.StringVar(value=f"{self.weights[i]:.2f}")
            entry = tk.Entry(
                row,
                textvariable=entry_var,
                bg=COLORS['bg_section'],
                fg=COLORS['accent2'],
                insertbackground=COLORS['accent'],
                width=7,
                font=('Consolas', 10),
                justify='center',
                relief='flat',
                highlightthickness=1,
                highlightbackground=COLORS['border'],
                highlightcolor=COLORS['accent']
            )
            entry.pack(side=tk.LEFT)
            entry.bind('<Return>', lambda e, idx=i: self._on_entry_change(idx))
            entry.bind('<FocusOut>', lambda e, idx=i: self._on_entry_change(idx))
            self.slider_entries.append(entry_var)
        
        # Pixel mode selector
        mode_frame = tk.Frame(controls_content, bg=COLORS['bg_panel'])
        mode_frame.pack(fill=tk.X, pady=(10, 0))
        
        mode_label = tk.Label(
            mode_frame,
            text="Pixel Mode:",
            bg=COLORS['bg_panel'],
            fg=COLORS['text'],
            width=22,
            anchor='w'
        )
        mode_label.pack(side=tk.LEFT)
        
        self.mode_var = tk.IntVar(value=self.pixel_mode)
        
        for mode in [1, 2, 3, 4]:
            w, h, _ = PIXEL_MODES[mode]
            mode_radio = tk.Radiobutton(
                mode_frame,
                text=f"{mode} ({w}×{h})",
                variable=self.mode_var,
                value=mode,
                command=self._on_mode_change,
                bg=COLORS['bg_panel'],
                fg=COLORS['text'],
                selectcolor=COLORS['bg_section'],
                activebackground=COLORS['bg_panel'],
                activeforeground=COLORS['accent']
            )
            mode_radio.pack(side=tk.LEFT, padx=(0, 8))
        
        self._update_weight_display()
    
    def _on_mode_change(self):
        """Handle pixel mode change."""
        new_mode = self.mode_var.get()
        if new_mode != self.pixel_mode:
            self._set_pixel_mode(new_mode)
            self._log(f"Pixel mode changed to {new_mode} ({self.img_width}×{self.img_height})")
            
            # Update original info label if image is loaded
            if self.source_image:
                src_w, src_h = self.source_image.size
                self.original_info_label.config(
                    text=f"{src_w}×{src_h} → {self.img_width}×{self.img_height}"
                )
            
            self._update_displays()
    
    def _create_console_section(self, parent: tk.Frame):
        """Create the console output section."""
        console_panel, console_content = self._create_panel(parent, "Console")
        console_panel.pack(fill=tk.BOTH, expand=True)
        
        self.console = tk.Text(
            console_content,
            height=8,
            bg='#0a0a14',
            fg='#00ff00',
            insertbackground='#00ff00',
            font=('Consolas', 10),
            state=tk.DISABLED
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self._log("ARMlite Weight Tuner initialized")
        self._log("Shortcuts: Ctrl+O=Load, Ctrl+S=Export, R=Reset, 1-4=Mode")
        self._log("LOAD an image to begin")
    
    def _log(self, message: str):
        """Log a message to the console."""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
    
    def _set_pixel_mode(self, mode: int):
        """Change the pixel mode and resize displays."""
        if mode == self.pixel_mode:
            return
        
        self.pixel_mode = mode
        self.mode_var.set(mode)  # Sync radio button UI
        w, h = self.img_width, self.img_height
        scale = self.display_scale
        canvas_w, canvas_h = w * scale, h * scale
        
        # Resize canvases
        self.original_canvas.config(width=canvas_w, height=canvas_h)
        self.quantized_canvas.config(width=canvas_w, height=canvas_h)
        self.pixel_frame.config(width=canvas_w, height=canvas_h)
        
        # Update info label
        self.pixel_count_label.config(
            text=f"Mode {mode}: {w}x{h} ({w * h} px)"
        )
        
        self._log(f"Switched to Mode {mode}: {w}x{h}")
        
        # Re-process image if loaded
        if self.source_image:
            self.display_image = self._fit_to_armlite(self.source_image)
            # Invalidate caches (image dimensions changed)
            self._image_id += 1
            self._match_cache.clear()
            self._quant_cache.clear()
            self._schedule_update()
    
    def _browse_image(self):
        """Open file dialog to select an image."""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self._load_image(path)
    
    def _load_image(self, path: str):
        """Load and process an image."""
        try:
            self.source_image = Image.open(path).convert('RGB')
            self.image_path = path
            
            # Scale to ARMlite size (64x48) preserving aspect ratio with padding
            self.display_image = self._fit_to_armlite(self.source_image)
            
            # Invalidate caches for new image
            self._image_id += 1
            self._match_cache.clear()
            self._quant_cache.clear()
            
            filename = os.path.basename(path)
            src_w, src_h = self.source_image.size
            self._log(f"Loaded: {filename} ({src_w}x{src_h})")
            self._log(f"Scaled to: {self.display_image.size[0]}x{self.display_image.size[1]}")
            
            # Update original info label
            self.original_info_label.config(
                text=f"{src_w}×{src_h} → {self.img_width}×{self.img_height}"
            )
            
            self._update_displays()
            
        except Exception as e:
            self._log(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Could not load image:\n{e}")
    
    def _fit_to_armlite(self, img: Image.Image) -> Image.Image:
        """Resize image to fit current ARMlite pixel mode."""
        return img.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
    
    def _on_space_change(self):
        """Handle color space toggle."""
        new_space = self.space_var.get()
        if new_space != self.current_space:
            self.current_space = new_space
            
            # Switch to appropriate default weights
            if new_space == 'hsv':
                self.weights = list(HSV_DEFAULTS)
            else:
                self.weights = list(HSL_DEFAULTS)
            
            # Update sliders and entries
            for i, var in enumerate(self.slider_vars):
                var.set(self.weights[i])
                self.slider_entries[i].set(f"{self.weights[i]:.2f}")
            
            self._log(f"Switched to {new_space.upper()} color space")
            self._update_weight_display()
            self._schedule_update()
    
    def _on_slider_change(self, index: int):
        """Handle slider value change."""
        value = self.slider_vars[index].get()
        self.weights[index] = value
        self.slider_entries[index].set(f"{value:.2f}")
        self._update_weight_display()
        self._schedule_update()
    
    def _on_entry_change(self, index: int):
        """Handle manual entry of weight value."""
        try:
            value = float(self.slider_entries[index].get())
            value = max(0.0, min(10.0, value))  # Clamp to slider range
            self.weights[index] = value
            self.slider_vars[index].set(value)
            self.slider_entries[index].set(f"{value:.2f}")
            self._update_weight_display()
            self._schedule_update()
        except ValueError:
            # Revert to current weight if invalid
            self.slider_entries[index].set(f"{self.weights[index]:.2f}")
    
    def _update_weight_display(self):
        """Update the weight display label."""
        space = self.current_space.upper()
        w = self.weights
        self.weight_display.config(
            text=f"{space}: ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})"
        )
    
    def _schedule_update(self):
        """Schedule a display update with debouncing."""
        if self._update_timer:
            self.root.after_cancel(self._update_timer)
        # Short debounce - numpy makes rendering fast enough for near-realtime
        self._update_timer = self.root.after(16, self._update_displays)  # ~60fps target
    
    def _update_displays(self):
        """Update both image displays."""
        self._update_timer = None
        
        if self.display_image is None:
            return
        
        # Update original display
        self._draw_original()
        
        # Update quantized display
        self._draw_quantized()
    
    def _draw_original(self):
        """Draw the original image (scaled up)."""
        if self.display_image is None:
            return
        
        # Scale up for display
        display = self.display_image.resize(
            (self.img_width * self.display_scale, self.img_height * self.display_scale),
            Image.Resampling.NEAREST
        )
        
        self._original_photo = ImageTk.PhotoImage(display)
        self.original_canvas.delete("all")
        self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self._original_photo)
    
    def _draw_quantized(self):
        """Draw the quantized image using ARMlite palette."""
        if self.display_image is None:
            return
        
        w, h = self.img_width, self.img_height
        weights_tuple = (round(self.weights[0], 2), round(self.weights[1], 2), round(self.weights[2], 2))
        
        # Check quantization cache
        cache_key = (self._image_id, self.current_space, weights_tuple)
        if cache_key in self._quant_cache:
            quantized = self._quant_cache[cache_key]
        else:
            # Compute and cache
            weights = np.array(self.weights, dtype=np.float32) if HAS_NUMPY else tuple(self.weights)
            
            if HAS_NUMPY:
                # Fast vectorized quantization
                quantized = self._quantize_numpy(w, h, weights)
            else:
                # Fallback to slow per-pixel loop
                quantized = self._quantize_slow(w, h, weights)
            
            # Cache result (limit cache size to avoid memory bloat)
            if len(self._quant_cache) > 100:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._quant_cache))
                del self._quant_cache[oldest_key]
            self._quant_cache[cache_key] = quantized
        
        # Scale up for display
        display = quantized.resize(
            (w * self.display_scale, h * self.display_scale),
            Image.Resampling.NEAREST
        )
        
        self._quantized_photo = ImageTk.PhotoImage(display)
        self.quantized_canvas.delete("all")
        self.quantized_canvas.create_image(0, 0, anchor=tk.NW, image=self._quantized_photo)
    
    def _quantize_numpy(self, width: int, height: int, weights: 'np.ndarray') -> Image.Image:
        """Fast numpy-vectorized quantization."""
        # Get image as numpy array (H, W, 3)
        img_arr = np.array(self.display_image, dtype=np.float32) / 255.0
        
        # Convert RGB to HSV or HSL
        if self.current_space == 'hsv':
            # Vectorized RGB->HSV
            r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
            maxc = np.maximum(np.maximum(r, g), b)
            minc = np.minimum(np.minimum(r, g), b)
            v = maxc
            s = np.where(maxc != 0, (maxc - minc) / maxc, 0)
            
            # Hue calculation
            delta = maxc - minc
            hue = np.zeros_like(maxc)
            mask_r = (maxc == r) & (delta != 0)
            mask_g = (maxc == g) & (delta != 0)
            mask_b = (maxc == b) & (delta != 0)
            hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
            hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
            hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
            hue = hue / 6.0
            hue[hue < 0] += 1.0
            
            converted = np.stack([hue, s, v], axis=-1)  # (H, W, 3)
            palette_np = _PALETTE_HSV_NP
        else:
            # Vectorized RGB->HSL
            r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
            maxc = np.maximum(np.maximum(r, g), b)
            minc = np.minimum(np.minimum(r, g), b)
            light = (maxc + minc) / 2.0
            
            delta = maxc - minc
            s = np.zeros_like(light)
            mask = delta != 0
            s[mask & (light <= 0.5)] = delta[mask & (light <= 0.5)] / (maxc[mask & (light <= 0.5)] + minc[mask & (light <= 0.5)])
            s[mask & (light > 0.5)] = delta[mask & (light > 0.5)] / (2.0 - maxc[mask & (light > 0.5)] - minc[mask & (light > 0.5)])
            
            # Hue calculation (same as HSV)
            hue = np.zeros_like(maxc)
            mask_r = (maxc == r) & (delta != 0)
            mask_g = (maxc == g) & (delta != 0)
            mask_b = (maxc == b) & (delta != 0)
            hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
            hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
            hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
            hue = hue / 6.0
            hue[hue < 0] += 1.0
            
            converted = np.stack([hue, s, light], axis=-1)  # (H, W, 3)
            palette_np = _PALETTE_HSL_NP
        
        # Reshape for broadcasting: (H*W, 1, 3) vs (1, N_colors, 3)
        pixels = converted.reshape(-1, 1, 3)  # (H*W, 1, 3)
        palette = palette_np.reshape(1, -1, 3)  # (1, N, 3)
        
        # Compute weighted distances with hue wrapping
        dh_raw = np.abs(pixels[:,:,0] - palette[:,:,0])
        dh = np.minimum(dh_raw, 1.0 - dh_raw)  # Hue wrapping
        ds = np.abs(pixels[:,:,1] - palette[:,:,1])
        dv = np.abs(pixels[:,:,2] - palette[:,:,2])
        
        # Weighted squared distance (sqrt not needed for argmin)
        dist = (weights[0] * dh) ** 2 + (weights[1] * ds) ** 2 + (weights[2] * dv) ** 2
        
        # Find closest palette index for each pixel
        closest_idx = np.argmin(dist, axis=1)  # (H*W,)
        
        # Map to RGB output
        output = _PALETTE_RGB_NP[closest_idx].reshape(height, width, 3)
        
        return Image.fromarray(output, 'RGB')
    
    def _quantize_slow(self, w: int, h: int, weights: tuple) -> Image.Image:
        """Slow fallback quantization without numpy."""
        transform = rgb_to_hsv if self.current_space == 'hsv' else rgb_to_hsl
        palette = PALETTE_HSV if self.current_space == 'hsv' else PALETTE_HSL
        
        quantized = Image.new('RGB', (w, h))
        
        for y in range(h):
            for x in range(w):
                px = self.display_image.getpixel((x, y))
                if isinstance(px, int):
                    rgb: tuple[int, int, int] = (px, px, px)
                else:
                    rgb = (px[0], px[1], px[2])
                converted = transform(rgb)
                
                # Find closest palette color
                best_name = 'black'
                best_dist = float('inf')
                for name, target in palette.items():
                    d = weighted_distance(converted, target, weights)
                    if d < best_dist:
                        best_dist = d
                        best_name = name
                
                quantized.putpixel((x, y), ARMLITE_RGB[best_name])
        
        return quantized
    
    def _reset_weights(self):
        """Reset weights to defaults for current color space."""
        if self.current_space == 'hsv':
            self.weights = list(HSV_DEFAULTS)
        else:
            self.weights = list(HSL_DEFAULTS)
        
        for i, var in enumerate(self.slider_vars):
            var.set(self.weights[i])
            self.slider_entries[i].set(f"{self.weights[i]:.2f}")
        
        self._log(f"Reset to {self.current_space.upper()} defaults: {tuple(self.weights)}")
        self._update_weight_display()
        self._schedule_update()
    
    def _auto_match(self):
        """Fast auto-optimize weights using numpy vectorized grid search."""
        if self.display_image is None:
            messagebox.showwarning("No Image", "Load an image first")
            return
        
        # Check cache first
        cache_key = (self._image_id, self.current_space)
        if cache_key in self._match_cache:
            cached_weights, cached_error = self._match_cache[cache_key]
            self.weights = list(cached_weights)
            for i, var in enumerate(self.slider_vars):
                var.set(self.weights[i])
                self.slider_entries[i].set(f"{self.weights[i]:.2f}")
            self._log(f"[cached] {self.current_space.upper()}: ({self.weights[0]:.2f}, {self.weights[1]:.2f}, {self.weights[2]:.2f})")
            self._log(f"Avg RGB error: {cached_error:.1f}")
            self._update_weight_display()
            self._schedule_update()
            return
        
        if not HAS_NUMPY:
            self._log("numpy not installed - using fallback grid search")
            self._auto_match_grid()
            return
        
        self._log("Optimizing weights (fast grid search)...")
        self.root.update()
        
        # Get transform and palette for current color space
        transform = rgb_to_hsv if self.current_space == 'hsv' else rgb_to_hsl
        palette = PALETTE_HSV if self.current_space == 'hsv' else PALETTE_HSL
        
        # Build numpy arrays for vectorized computation
        palette_names = list(palette.keys())
        palette_hsv = np.array([palette[n] for n in palette_names])  # (147, 3)
        palette_rgb = np.array([ARMLITE_RGB[n] for n in palette_names])  # (147, 3)
        
        # Cache pixel data as numpy arrays
        orig_rgb_list = []
        px_space_list = []
        for y in range(self.img_height):
            for x in range(self.img_width):
                px = self.display_image.getpixel((x, y))
                if isinstance(px, (tuple, list)) and len(px) >= 3:
                    rgb = (px[0], px[1], px[2])
                    orig_rgb_list.append(rgb)
                    px_space_list.append(transform(rgb))
        
        orig_rgb = np.array(orig_rgb_list)  # (N, 3)
        px_space = np.array(px_space_list)  # (N, 3)
        
        def compute_error(weights):
            """Compute total RGB error for given weights (vectorized)."""
            w = np.array(weights)
            
            # Compute weighted distance from each pixel to each palette color
            # Handle hue wrapping
            dh = np.abs(px_space[:, None, 0] - palette_hsv[None, :, 0])
            dh = np.minimum(dh, 1.0 - dh)  # Hue wrapping
            ds = np.abs(px_space[:, None, 1] - palette_hsv[None, :, 1])
            dv = np.abs(px_space[:, None, 2] - palette_hsv[None, :, 2])
            
            # Weighted distance: (N, 147)
            dist = np.sqrt((w[0] * dh)**2 + (w[1] * ds)**2 + (w[2] * dv)**2)
            
            # Find best match for each pixel
            best_idx = np.argmin(dist, axis=1)  # (N,)
            matched_rgb = palette_rgb[best_idx]  # (N, 3)
            
            # Compute RGB error
            rgb_diff = orig_rgb - matched_rgb
            total_error = np.sum(rgb_diff ** 2)
            return total_error
        
        # Grid search over weight space
        # Use finer grid for better results
        h_vals = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        s_vals = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        v_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        
        # Start from space-appropriate defaults (not current slider values)
        # This ensures consistent results regardless of slider state
        default_w = HSV_DEFAULTS if self.current_space == 'hsv' else HSL_DEFAULTS
        best_w = list(default_w)
        best_error = compute_error(best_w)
        
        for h in h_vals:
            for s in s_vals:
                for v in v_vals:
                    err = compute_error([h, s, v])
                    if err < best_error:
                        best_error = err
                        best_w = [h, s, v]
        
        # Apply optimized weights
        self.weights = best_w
        for i, var in enumerate(self.slider_vars):
            var.set(self.weights[i])
            self.slider_entries[i].set(f"{self.weights[i]:.2f}")
        
        # Report result and cache
        avg_error = math.sqrt(best_error / len(orig_rgb))
        self._match_cache[cache_key] = (list(best_w), avg_error)
        self._log(f"Optimized {self.current_space.upper()}: ({self.weights[0]:.2f}, {self.weights[1]:.2f}, {self.weights[2]:.2f})")
        self._log(f"Avg RGB error: {avg_error:.1f}")
        self._update_weight_display()
        self._schedule_update()
    
    def _auto_match_grid(self):
        """Grid search fallback when numpy is not available."""
        transform = rgb_to_hsv if self.current_space == 'hsv' else rgb_to_hsl
        palette = PALETTE_HSV if self.current_space == 'hsv' else PALETTE_HSL
        
        # Cache pixel data
        pixels = []
        for y in range(self.img_height):
            for x in range(self.img_width):
                px = self.display_image.getpixel((x, y))
                if isinstance(px, (tuple, list)) and len(px) >= 3:
                    rgb = (px[0], px[1], px[2])
                    pixels.append(transform(rgb))
        
        def evaluate(w):
            total = 0.0
            for px_space in pixels:
                best_dist = float('inf')
                for pal_space in palette.values():
                    d = weighted_distance(px_space, pal_space, w)
                    if d < best_dist:
                        best_dist = d
                total += best_dist
            return total
        
        # Coarse grid search
        best_w = self.weights[:]
        best_score = evaluate(tuple(best_w))
        
        for h in [0.5, 1.0, 2.0, 3.0, 5.0]:
            for s in [0.5, 1.0, 2.0, 3.0, 5.0]:
                for v in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
                    score = evaluate((h, s, v))
                    if score < best_score:
                        best_score = score
                        best_w = [h, s, v]
        
        self.weights = best_w
        for i, var in enumerate(self.slider_vars):
            var.set(self.weights[i])
            self.slider_entries[i].set(f"{self.weights[i]:.2f}")
        
        self._log(f"Grid search {self.current_space.upper()}: ({self.weights[0]:.2f}, {self.weights[1]:.2f}, {self.weights[2]:.2f})")
        self._update_weight_display()
        self._schedule_update()
    
    def _export_assembly(self):
        """Export current quantized image to ARMlite assembly."""
        if self.display_image is None:
            messagebox.showwarning("No Image", "Load an image first")
            return
        
        # Generate default filename
        space = self.current_space
        w = self.weights
        default_name = f"{space}_{w[0]:.1f}_{w[1]:.1f}_{w[2]:.1f}.s"
        
        filetypes = [("Assembly files", "*.s"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(
            defaultextension=".s",
            filetypes=filetypes,
            initialfile=default_name
        )
        
        if not path:
            return
        
        try:
            self._generate_assembly(path)
            self._log(f"Exported: {os.path.basename(path)}")
            messagebox.showinfo("Export Complete", f"Saved to:\n{path}")
        except Exception as e:
            self._log(f"Export error: {e}")
            messagebox.showerror("Export Error", str(e))
    
    def _generate_assembly(self, output_path: str):
        """Generate ARMlite assembly file."""
        transform = rgb_to_hsv if self.current_space == 'hsv' else rgb_to_hsl
        palette = PALETTE_HSV if self.current_space == 'hsv' else PALETTE_HSL
        weights = (self.weights[0], self.weights[1], self.weights[2])
        
        w, h = self.img_width, self.img_height
        
        # Quantize image
        pixels = []
        for y in range(h):
            for x in range(w):
                px = self.display_image.getpixel((x, y))
                if isinstance(px, int):
                    rgb: tuple[int, int, int] = (px, px, px)
                else:
                    rgb = (px[0], px[1], px[2])
                converted = transform(rgb)
                
                best_name = 'black'
                best_dist = float('inf')
                for name, target in palette.items():
                    d = weighted_distance(converted, target, weights)
                    if d < best_dist:
                        best_dist = d
                        best_name = name
                
                color = ARMLITE_RGB[best_name]
                hex_val = (color[0] << 16) | (color[1] << 8) | color[2]
                pixels.append(hex_val)
        
        # Generate assembly
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        source_name = os.path.basename(self.image_path) if self.image_path else "unknown"
        
        lines = [
            f"// Generated by weight_tuner_gui.py",
            f"// Timestamp: {timestamp}",
            f"// Source: {source_name}",
            f"// Color space: {self.current_space.upper()}",
            f"// Weights: ({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f})",
            f"// Pixel mode: {self.pixel_mode} ({w}x{h})",
            "",
            f"// Set pixel mode to {w}x{h}",
            f"MOV R0, #{self.pixel_mode}",
            "STR R0, .WritePixelMode",
            "",
            "// Set up base address for pixel data",
            "LDR R1, =pixels",
            "MOV R2, #0              // pixel index",
            f"MOV R3, #{w * h}        // total pixels",
            "",
            "draw_loop:",
            "    LDR R4, [R1], #4    // load color, advance pointer",
            "    STR R2, .WritePixelX // set X from index (auto-wraps)",
            "    STR R4, .WritePixelColour",
            "    ADD R2, R2, #1",
            "    CMP R2, R3",
            "    BLT draw_loop",
            "",
            "HALT",
            "",
            ".DATA",
            "pixels:",
        ]
        
        # Add pixel data
        for px in pixels:
            lines.append(f".WORD 0x{px:06X}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='ARMlite-style GUI for HSV/HSL weight tuning'
    )
    parser.add_argument('image', nargs='?', help='Path to input image')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ARMliteStyleApp(root, args.image)
    root.mainloop()


if __name__ == '__main__':
    main()
