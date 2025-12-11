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


def slice_hues(cfg: RenderConfig) -> Tuple[float, float]:
    if cfg.slice_hue_override is not None:
        return (cfg.slice_hue_override, cfg.slice_hue_override)
    lower_angle, upper_angle = slice_boundary_angles(cfg)
    return hue_from_angle(lower_angle), hue_from_angle(upper_angle)


def plot_hsv_cylinder(ax, cfg: RenderConfig) -> None:
    theta = np.linspace(0.0, 2 * np.pi, cfg.hsv_theta_steps)
    value = np.linspace(0.0, 1.0, cfg.hsv_value_steps)
    theta_grid, value_grid = np.meshgrid(theta, value)

    x = np.cos(theta_grid)
    y = np.sin(theta_grid)
    z = value_grid

    hues = (theta_grid / (2 * np.pi)) % 1.0
    sat = np.ones_like(hues)
    hsv = np.stack([hues, sat, value_grid], axis=-1)
    colors = hsv_to_rgb(hsv)

    mask = slice_mask(theta_grid, cfg.slice_center_angle, cfg.slice_half_width())
    x = np.where(mask, x, np.nan)
    y = np.where(mask, y, np.nan)
    z = np.where(mask, z, np.nan)

    ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False, shade=False)

    # Optional caps reuse the same mask logic so top and bottom align perfectly.
    if cfg.draw_hsv_bottom_cap:
        plot_hsv_disk(ax, z=0.0, value_level=0.0, cfg=cfg)
    if cfg.draw_hsv_top_cap:
        plot_hsv_disk(ax, z=1.0, value_level=1.0, cfg=cfg)

    plot_hsv_slice_plane(ax, cfg)
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


def _draw_hsv_wall(ax, angle: float, hue: float, n: int) -> None:
    """Draw a single HSV slice wall at the given angle with the given hue."""
    sat = np.linspace(0.0, 1.0, n)
    value = np.linspace(0.0, 1.0, n)
    sat_grid, value_grid = np.meshgrid(sat, value)
    
    hsv = np.stack([
        np.full_like(sat_grid, hue),
        sat_grid,
        value_grid,
    ], axis=-1)
    colors = hsv_to_rgb(hsv)
    
    x = sat_grid * np.cos(angle)
    y = sat_grid * np.sin(angle)
    z = value_grid
    
    ax.plot_surface(
        x, y, z,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def plot_hsv_slice_plane(ax, cfg: RenderConfig) -> None:
    """Draw the inner slice walls as rectangular cross-sections."""
    n = cfg.hsv_value_steps
    
    for angle in slice_boundary_angles(cfg):
        hue = cfg.slice_hue_override if cfg.slice_hue_override is not None else hue_from_angle(angle)
        _draw_hsv_wall(ax, angle, hue, n)


def plot_hsl_double_cone(ax, cfg: RenderConfig) -> None:
    theta = np.linspace(0.0, 2 * np.pi, cfg.hsl_theta_steps)
    lightness = np.linspace(0.0, 1.0, cfg.hsl_lightness_steps)
    theta_grid, lightness_grid = np.meshgrid(theta, lightness)

    radius = 1.0 - np.abs(2.0 * lightness_grid - 1.0)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = lightness_grid

    hues = (theta_grid / (2 * np.pi)) % 1.0
    sat = radius
    colors = hls_to_rgb_array(hues, lightness_grid, sat)

    mask = slice_mask(theta_grid, cfg.slice_center_angle, cfg.slice_half_width())
    x = np.where(mask, x, np.nan)
    y = np.where(mask, y, np.nan)
    z = np.where(mask, z, np.nan)

    ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False, shade=False)
    plot_hsl_slice_plane(ax, cfg)
    ax.set_title("HSL double-cone", pad=10)


def _draw_hsl_wall(ax, angle: float, hue: float, n: int) -> None:
    """Draw a single HSL slice wall at the given angle with the given hue."""
    lightness = np.linspace(0.0, 1.0, n)
    sat = np.linspace(0.0, 1.0, n)
    sat_grid, lightness_grid = np.meshgrid(sat, lightness)
    
    radius_max = 1.0 - np.abs(2.0 * lightness_grid - 1.0)
    radius = sat_grid * radius_max
    
    colors = hls_to_rgb_array(
        np.full_like(lightness_grid, hue),
        lightness_grid,
        sat_grid,
    )
    
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = lightness_grid
    
    ax.plot_surface(
        x, y, z,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def plot_hsl_slice_plane(ax, cfg: RenderConfig) -> None:
    """Draw the inner slice walls for the HSL double-cone."""
    n = cfg.hsl_lightness_steps
    
    for angle in slice_boundary_angles(cfg):
        hue = cfg.slice_hue_override if cfg.slice_hue_override is not None else hue_from_angle(angle)
        _draw_hsl_wall(ax, angle, hue, n)


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
