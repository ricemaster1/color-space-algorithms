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
    slice_width: float = np.deg2rad(130.0)
    slice_hue_override: Optional[float] = None  # set to 5/6 for magenta, etc.
    hsv_slice_value_span: float = 1.0  # 1.0 = full-height wall gradient; shrink to compress
    draw_hsv_top_cap: bool = True
    draw_hsv_bottom_cap: bool = True
    output_path: Path = Path("hsv_hsl_.svg")

    def slice_half_width(self) -> float:
        return max(self.slice_width / 2.0, 0.0)


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
    
    # Sort by camera depth and render
    cam_dir = _camera_direction()
    sorted_verts, sorted_colors = _sort_polys_by_depth(all_verts, all_colors, cam_dir)
    
    poly = Poly3DCollection(sorted_verts, facecolors=sorted_colors, edgecolors='none')
    ax.add_collection3d(poly)

    # Optional caps (drawn separately - they don't have z-ordering issues)
    if cfg.draw_hsv_bottom_cap:
        plot_hsv_disk(ax, z=0.0, value_level=0.0, cfg=cfg)
    if cfg.draw_hsv_top_cap:
        plot_hsv_disk(ax, z=1.0, value_level=1.0, cfg=cfg)

    ax.set_title("HSV cylinder", pad=10)


def plot_hsv_disk(ax, z: float, value_level: float, cfg: RenderConfig) -> None:
    radius = np.linspace(0.0, 1.0, cfg.hsv_value_steps)
    theta = np.linspace(0.0, 2 * np.pi, cfg.hsv_theta_steps)
    radius_grid, theta_grid = np.meshgrid(radius, theta)

    x = radius_grid * np.cos(theta_grid)
    y = radius_grid * np.sin(theta_grid)
    z_grid = np.full_like(x, z)

    hues = (theta_grid / (2 * np.pi)) % 1.0
    hsv = np.stack([hues, radius_grid, np.full_like(radius_grid, value_level)], axis=-1)
    colors = hsv_to_rgb(hsv)

    mask = slice_mask(theta_grid, cfg.slice_center_angle, cfg.slice_half_width())
    x = np.where(mask, x, np.nan)
    y = np.where(mask, y, np.nan)
    z_grid = np.where(mask, z_grid, np.nan)

    ax.plot_surface(x, y, z_grid, facecolors=colors, linewidth=0, antialiased=False, shade=False)


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
    left_angle, right_angle = slice_boundary_angles(cfg)
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

    ax.set_title("HSL double-cone", pad=10)


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


def configure_axes(ax) -> None:
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=25, azim=35)


def main() -> None:
    cfg = RenderConfig()
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)

    ax_hsv = fig.add_subplot(1, 2, 1, projection="3d")
    plot_hsv_cylinder(ax_hsv, cfg)
    configure_axes(ax_hsv)

    ax_hsl = fig.add_subplot(1, 2, 2, projection="3d")
    plot_hsl_double_cone(ax_hsl, cfg)
    configure_axes(ax_hsl)

    fig.suptitle("HSV vs HSL scaffold", fontsize=16)
    fig.savefig(cfg.output_path, transparent=True)
    print(f"Saved figure to {cfg.output_path.resolve()}")


if __name__ == "__main__":
    main()
