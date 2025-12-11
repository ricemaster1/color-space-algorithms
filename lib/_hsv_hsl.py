"""HSV/HSL geometry scaffold using Matplotlib.

This module rebuilds the HSV cylinder and HSL double-cone with clean slice
controls so we can iterate inside Matplotlib before exporting meshes to Blender.

References
---------
- Linus Östholm, *blender-plots*, https://github.com/Linusnie/blender-plots
- Vignolini Lab, *PyLlama*, https://github.com/VignoliniLab/PyLlama
- Jake VanderPlas, *Python Data Science Handbook*,
  https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(slots=True)
class RenderConfig:
    """Configuration bundle for the HSV/HSL figure."""

    hsv_theta_steps: int = 360
    hsv_value_steps: int = 160
    hsl_theta_steps: int = 360
    hsl_lightness_steps: int = 200
    slice_center_angle: float = 0.0
    slice_width: float = np.deg2rad(65.0)  # HSV slice width
    hsl_slice_width: float = np.deg2rad(45.0)  # HSL slice width (smaller to show more color)
    slice_hue_override: Optional[float] = None  # set to 5/6 for magenta, etc.
    hsv_slice_value_span: float = 1.0  # 1.0 = full-height wall gradient; shrink to compress
    draw_hsv_top_cap: bool = True
    draw_hsv_bottom_cap: bool = True
    output_path: Path = Path("hsv_hsl_.svg")

    def slice_half_width(self) -> float:
        return max(self.slice_width / 2.0, 0.0)
    
    def hsl_slice_half_width(self) -> float:
        return max(self.hsl_slice_width / 2.0, 0.0)


def hue_from_angle(angle: float) -> float:
    """Map an angle in radians to a hue fraction in [0, 1)."""

    return ((angle % (2 * np.pi)) / (2 * np.pi)) % 1.0


def slice_mask(theta_grid: np.ndarray, center_angle: float, half_width: float) -> np.ndarray:
    """Boolean mask that keeps cells outside the slice window."""

    wrapped = (theta_grid - center_angle + np.pi) % (2 * np.pi) - np.pi
    return np.abs(wrapped) > half_width


def hls_to_rgb_array(h: np.ndarray, l: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Vectorized wrapper over colorsys.hls_to_rgb."""

    vectorized = np.vectorize(colorsys.hls_to_rgb, otypes=[float, float, float])
    r, g, b = vectorized(h, l, s)
    return np.stack([r, g, b], axis=-1)


def slice_boundary_angles(cfg: RenderConfig) -> Tuple[float, float]:
    hw = cfg.slice_half_width()
    return cfg.slice_center_angle - hw, cfg.slice_center_angle + hw


def hsl_slice_boundary_angles(cfg: RenderConfig) -> Tuple[float, float]:
    hw = cfg.hsl_slice_half_width()
    return cfg.slice_center_angle - hw, cfg.slice_center_angle + hw


def _camera_direction(elev: float = 25, azim: float = 35) -> np.ndarray:
    """Get normalized camera direction vector for painter's algorithm."""
    return np.array([
        np.cos(np.radians(azim)) * np.cos(np.radians(elev)),
        np.sin(np.radians(azim)) * np.cos(np.radians(elev)),
        np.sin(np.radians(elev))
    ])


def _sort_polys_by_depth(verts: list, colors: list, cam_dir: np.ndarray) -> Tuple[list, list]:
    """Sort polygons by depth using painter's algorithm (furthest first)."""
    def poly_depth(quad):
        center = np.mean(quad, axis=0)
        return np.dot(center, cam_dir)
    
    sorted_indices = sorted(range(len(verts)), key=lambda i: poly_depth(verts[i]))
    return [verts[i] for i in sorted_indices], [colors[i] for i in sorted_indices]


def slice_hues(cfg: RenderConfig) -> Tuple[float, float]:
    if cfg.slice_hue_override is not None:
        return (cfg.slice_hue_override, cfg.slice_hue_override)
    lower_angle, upper_angle = slice_boundary_angles(cfg)
    return hue_from_angle(lower_angle), hue_from_angle(upper_angle)


def plot_hsv_cylinder(ax, cfg: RenderConfig) -> None:
    all_verts = []
    all_colors = []
    
    # Build cylinder polygons (excluding sliced section)
    left_angle, right_angle = slice_boundary_angles(cfg)
    # Go from right angle around to left angle (the visible part)
    theta_start = right_angle
    theta_end = left_angle + 2 * np.pi
    
    n_theta = cfg.hsv_theta_steps
    n_value = cfg.hsv_value_steps
    theta = np.linspace(theta_start, theta_end, n_theta)
    value = np.linspace(0.0, 1.0, n_value)
    
    for i in range(n_value - 1):
        for j in range(n_theta - 1):
            t0, t1 = theta[j], theta[j + 1]
            v0, v1 = value[i], value[i + 1]
            
            quad = [
                (np.cos(t0), np.sin(t0), v0),
                (np.cos(t1), np.sin(t1), v0),
                (np.cos(t1), np.sin(t1), v1),
                (np.cos(t0), np.sin(t0), v1),
            ]
            all_verts.append(quad)
            
            t_mid = (t0 + t1) / 2
            v_mid = (v0 + v1) / 2
            h = hue_from_angle(t_mid)
            hsv = np.array([[[h, 1.0, v_mid]]])
            rgb = hsv_to_rgb(hsv)[0, 0]
            all_colors.append(rgb)
    
    # Add wall polygons
    _add_hsv_wall_polys(all_verts, all_colors, left_angle, cfg)
    _add_hsv_wall_polys(all_verts, all_colors, right_angle, cfg)
    
    # Add cap polygons
    if cfg.draw_hsv_bottom_cap:
        _add_hsv_disk_polys(all_verts, all_colors, z=0.0, value_level=0.0, cfg=cfg)
    if cfg.draw_hsv_top_cap:
        _add_hsv_disk_polys(all_verts, all_colors, z=1.0, value_level=1.0, cfg=cfg)
    
    # Sort by camera depth and render
    cam_dir = _camera_direction()
    sorted_verts, sorted_colors = _sort_polys_by_depth(all_verts, all_colors, cam_dir)
    
    poly = Poly3DCollection(sorted_verts, facecolors=sorted_colors, edgecolors='none')
    ax.add_collection3d(poly)
    
    # Add annotations
    _add_hue_labels(ax, radius=1.2, z=0.5)
    _add_vertical_arrow(ax, x=-0.2, y=-1.3, z0=0.0, z1=1.0, label="Value", color='white')
    _add_saturation_arrow(ax, angle=np.radians(200), z=1.05, r0=0.0, r1=1.0, color='white')

    ax.set_title("HSV cylinder", pad=10, color='white')


def _add_hsv_disk_polys(all_verts: list, all_colors: list, z: float, value_level: float, cfg: RenderConfig) -> None:
    """Add HSV disk cap polygons to the collection for unified z-sorting."""
    left_angle, right_angle = slice_boundary_angles(cfg)
    theta_start = right_angle
    theta_end = left_angle + 2 * np.pi
    
    n_theta = cfg.hsv_theta_steps // 4  # Reduce resolution for caps
    n_radius = cfg.hsv_value_steps // 4
    theta = np.linspace(theta_start, theta_end, n_theta)
    radius = np.linspace(0.0, 1.0, n_radius)
    
    for i in range(n_radius - 1):
        for j in range(n_theta - 1):
            r0, r1 = radius[i], radius[i + 1]
            t0, t1 = theta[j], theta[j + 1]
            
            quad = [
                (r0 * np.cos(t0), r0 * np.sin(t0), z),
                (r1 * np.cos(t0), r1 * np.sin(t0), z),
                (r1 * np.cos(t1), r1 * np.sin(t1), z),
                (r0 * np.cos(t1), r0 * np.sin(t1), z),
            ]
            all_verts.append(quad)
            
            t_mid = (t0 + t1) / 2
            r_mid = (r0 + r1) / 2
            h = hue_from_angle(t_mid)
            hsv = np.array([[[h, r_mid, value_level]]])
            rgb = hsv_to_rgb(hsv)[0, 0]
            all_colors.append(rgb)


def _add_hsv_wall_polys(all_verts: list, all_colors: list, angle: float, cfg: RenderConfig) -> None:
    """Add HSV slice wall polygons to the collection for unified z-sorting."""
    n = cfg.hsv_value_steps
    hue = cfg.slice_hue_override if cfg.slice_hue_override is not None else hue_from_angle(angle)
    
    sat = np.linspace(0.0, 1.0, n)
    value = np.linspace(0.0, 1.0, n)
    
    for i in range(n - 1):
        for j in range(n - 1):
            s0, s1 = sat[j], sat[j + 1]
            v0, v1 = value[i], value[i + 1]
            
            x0, y0 = s0 * np.cos(angle), s0 * np.sin(angle)
            x1, y1 = s1 * np.cos(angle), s1 * np.sin(angle)
            
            quad = [
                (x0, y0, v0),
                (x1, y1, v0),
                (x1, y1, v1),
                (x0, y0, v1),
            ]
            all_verts.append(quad)
            
            s_mid = (s0 + s1) / 2
            v_mid = (v0 + v1) / 2
            hsv = np.array([[[hue, s_mid, v_mid]]])
            rgb = hsv_to_rgb(hsv)[0, 0]
            all_colors.append(rgb)


def plot_hsl_double_cone(ax, cfg: RenderConfig) -> None:
    all_verts = []
    all_colors = []
    
    # Build double-cone polygons (excluding sliced section)
    left_angle, right_angle = hsl_slice_boundary_angles(cfg)
    theta_start = right_angle
    theta_end = left_angle + 2 * np.pi
    
    n_theta = cfg.hsl_theta_steps
    n_light = cfg.hsl_lightness_steps
    theta = np.linspace(theta_start, theta_end, n_theta)
    lightness = np.linspace(0.0, 1.0, n_light)
    
    for i in range(n_light - 1):
        for j in range(n_theta - 1):
            t0, t1 = theta[j], theta[j + 1]
            l0, l1 = lightness[i], lightness[i + 1]
            
            # Radius depends on lightness (double cone shape)
            r0 = 1.0 - abs(2.0 * l0 - 1.0)
            r1 = 1.0 - abs(2.0 * l1 - 1.0)
            
            quad = [
                (r0 * np.cos(t0), r0 * np.sin(t0), l0),
                (r0 * np.cos(t1), r0 * np.sin(t1), l0),
                (r1 * np.cos(t1), r1 * np.sin(t1), l1),
                (r1 * np.cos(t0), r1 * np.sin(t0), l1),
            ]
            all_verts.append(quad)
            
            t_mid = (t0 + t1) / 2
            l_mid = (l0 + l1) / 2
            r_mid = 1.0 - abs(2.0 * l_mid - 1.0)
            h = hue_from_angle(t_mid)
            rgb = colorsys.hls_to_rgb(h, l_mid, r_mid)
            all_colors.append(rgb)
    
    # Add wall polygons
    _add_hsl_wall_polys(all_verts, all_colors, left_angle, cfg)
    _add_hsl_wall_polys(all_verts, all_colors, right_angle, cfg)
    
    # Sort by camera depth and render
    cam_dir = _camera_direction()
    sorted_verts, sorted_colors = _sort_polys_by_depth(all_verts, all_colors, cam_dir)
    
    poly = Poly3DCollection(sorted_verts, facecolors=sorted_colors, edgecolors='none')
    ax.add_collection3d(poly)
    
    # Add annotations
    _add_hue_labels(ax, radius=1.2, z=0.5)
    _add_vertical_arrow(ax, x=-0.2, y=-1.3, z0=0.0, z1=1.0, label="Lightness", color='white')
    _add_saturation_arrow(ax, angle=np.radians(200), z=0.5, r0=0.0, r1=1.0, color='white')

    ax.set_title("HSL double-cone", pad=10, color='white')


def _add_hsl_wall_polys(all_verts: list, all_colors: list, angle: float, cfg: RenderConfig) -> None:
    """Add HSL slice wall polygons to the collection for unified z-sorting."""
    n = cfg.hsl_lightness_steps
    hue = cfg.slice_hue_override if cfg.slice_hue_override is not None else hue_from_angle(angle)
    
    lightness = np.linspace(0.0, 1.0, n)
    sat = np.linspace(0.0, 1.0, n)
    
    for i in range(n - 1):
        for j in range(n - 1):
            s0, s1 = sat[j], sat[j + 1]
            l0, l1 = lightness[i], lightness[i + 1]
            
            # Radius depends on lightness (double cone shape)
            r_max_0 = 1.0 - abs(2.0 * l0 - 1.0)
            r_max_1 = 1.0 - abs(2.0 * l1 - 1.0)
            
            r00 = s0 * r_max_0
            r10 = s1 * r_max_0
            r01 = s0 * r_max_1
            r11 = s1 * r_max_1
            
            quad = [
                (r00 * np.cos(angle), r00 * np.sin(angle), l0),
                (r10 * np.cos(angle), r10 * np.sin(angle), l0),
                (r11 * np.cos(angle), r11 * np.sin(angle), l1),
                (r01 * np.cos(angle), r01 * np.sin(angle), l1),
            ]
            all_verts.append(quad)
            
            s_mid = (s0 + s1) / 2
            l_mid = (l0 + l1) / 2
            rgb = colorsys.hls_to_rgb(hue, l_mid, s_mid)
            all_colors.append(rgb)


def configure_axes(ax, zoom: float = 1.0) -> None:
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=25, azim=35)
    # Zoom in by adjusting limits
    lim = 1.0 / zoom
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.1 / zoom, (1.0 + 0.1) / zoom)


def _add_hue_labels(ax, radius: float = 1.15, z: float = 0.5) -> None:
    """Add hue color labels around the shape (0°/Red, 120°/Green, 240°/Blue)."""
    labels = [
        (0, "0° Red", "red"),
        (120, "120° Green", "green"),
        (240, "240° Blue", "blue"),
    ]
    for deg, text, color in labels:
        rad = np.radians(deg)
        x, y = radius * np.cos(rad), radius * np.sin(rad)
        ax.text(x, y, z, text, color=color, fontsize=8, ha='center', va='center',
                fontweight='bold')


def _add_vertical_arrow(ax, x: float, y: float, z0: float, z1: float, label: str, color: str = 'white') -> None:
    """Add a vertical arrow with label for Value/Lightness scale."""
    from mpl_toolkits.mplot3d.art3d import Line3D
    # Draw arrow line
    ax.plot([x, x], [y, y], [z0, z1], color=color, linewidth=2)
    # Arrow head
    ax.plot([x, x], [y, y], [z1 - 0.05, z1], color=color, linewidth=3)
    # Labels at ends
    ax.text(x, y, z0 - 0.08, "0", color=color, fontsize=9, ha='center', va='top')
    ax.text(x, y, z1 + 0.08, "1", color=color, fontsize=9, ha='center', va='bottom')
    # Arrow label
    ax.text(x - 0.15, y, (z0 + z1) / 2, label, color=color, fontsize=10, ha='right', va='center',
            rotation=90)


def _add_saturation_arrow(ax, angle: float, z: float, r0: float, r1: float, color: str = 'white') -> None:
    """Add a radial arrow with label for Saturation scale."""
    x0, y0 = r0 * np.cos(angle), r0 * np.sin(angle)
    x1, y1 = r1 * np.cos(angle), r1 * np.sin(angle)
    # Draw arrow line
    ax.plot([x0, x1], [y0, y1], [z, z], color=color, linewidth=2)
    # Labels
    ax.text(x0, y0, z, "0", color=color, fontsize=9, ha='center', va='center')
    ax.text(x1 * 1.1, y1 * 1.1, z, "1", color=color, fontsize=9, ha='center', va='center')
    # Label in middle
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(xm, ym, z + 0.12, "Saturation", color=color, fontsize=9, ha='center', va='bottom')


def main() -> None:
    cfg = RenderConfig()
    fig = plt.figure(figsize=(16, 8))
    
    # Create dark gray radial gradient background
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.set_aspect('auto')
    
    # Radial gradient: lighter center, darker edges
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    # Distance from center, normalized
    R = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) / 0.7
    R = np.clip(R, 0, 1)
    # Gradient from light gray (center) to dark gray (edges)
    gradient = 0.35 - 0.15 * R  # 0.35 at center, 0.20 at edges
    rgb_bg = np.stack([gradient, gradient, gradient], axis=-1)
    ax_bg.imshow(rgb_bg, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    ax_bg.axis('off')

    ax_hsv = fig.add_subplot(1, 2, 1, projection="3d")
    ax_hsv.set_facecolor((0, 0, 0, 0))  # Transparent to show gradient
    ax_hsv.patch.set_alpha(0)
    plot_hsv_cylinder(ax_hsv, cfg)
    configure_axes(ax_hsv, zoom=1.2)

    ax_hsl = fig.add_subplot(1, 2, 2, projection="3d")
    ax_hsl.set_facecolor((0, 0, 0, 0))  # Transparent to show gradient
    ax_hsl.patch.set_alpha(0)
    plot_hsl_double_cone(ax_hsl, cfg)
    configure_axes(ax_hsl, zoom=1.2)

    fig.suptitle("HSV vs HSL scaffold", fontsize=18, color='white', y=0.95)
    fig.savefig(cfg.output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    print(f"Saved figure to {cfg.output_path.resolve()}")


if __name__ == "__main__":
    main()
