from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


SLICE_OPENING_ANGLE = np.deg2rad(80)  # angular width removed to reveal a slice
SLICE_FACE_ANGLE = -0.5  # angle of the visible inner wall (what we stand in front of)
SLICE_MAX_ANGLE = SLICE_FACE_ANGLE + SLICE_OPENING_ANGLE
SLICE_FACE_HUE = ((SLICE_FACE_ANGLE % (2 * np.pi)) / (2 * np.pi)) % 1.0


def _hls_to_rgb_array(h: np.ndarray, l: np.ndarray, s: np.ndarray) -> np.ndarray:
    hls_to_rgb = np.vectorize(colorsys.hls_to_rgb, otypes=[float, float, float])
    r, g, b = hls_to_rgb(h, l, s)
    return np.stack([r, g, b], axis=-1)


def _slice_mask(theta_values: np.ndarray) -> np.ndarray:
    wrapped = (theta_values - SLICE_FACE_ANGLE) % (2 * np.pi)
    return (wrapped >= 0.0) & (wrapped <= SLICE_OPENING_ANGLE)


def _apply_slice_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta_grid: np.ndarray):
    mask = _slice_mask(theta_grid)
    return (
        np.where(mask, np.nan, x),
        np.where(mask, np.nan, y),
        np.where(mask, np.nan, z),
    )


def _plot_hsv_cylinder(ax):
    theta = np.linspace(0, 2 * np.pi, 360)
    value = np.linspace(0.0, 1.0, 120)
    theta_grid, value_grid = np.meshgrid(theta, value)

    x = np.cos(theta_grid)
    y = np.sin(theta_grid)
    z = value_grid

    hue_norm = (theta_grid / (2 * np.pi)) % 1.0
    sat = np.ones_like(hue_norm)
    hsv = np.stack([hue_norm, sat, value_grid], axis=-1)
    colors = hsv_to_rgb(hsv)

    x, y, z = _apply_slice_mask(x, y, z, theta_grid)

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    _plot_hsv_disk(ax, z=0.0, value_level=0.0)
    _plot_hsv_disk(ax, z=1.0, value_level=1.0)
    _plot_hsv_slice_plane(ax)
    _plot_slice_boundaries(ax, shape='hsv')
    ax.set_title('HSV Cylinder', pad=16)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _plot_hsv_disk(ax, z: float, value_level: float):
    radius = np.linspace(0.0, 1.0, 120)
    theta = np.linspace(0, 2 * np.pi, 360)
    radius_grid, theta_grid = np.meshgrid(radius, theta)

    x = radius_grid * np.cos(theta_grid)
    y = radius_grid * np.sin(theta_grid)
    z_grid = np.full_like(x, z)

    hue_norm = (theta_grid / (2 * np.pi)) % 1.0
    sat = radius_grid
    val = np.full_like(radius_grid, value_level)
    hsv = np.stack([hue_norm, sat, val], axis=-1)
    colors = hsv_to_rgb(hsv)

    x, y, z_grid = _apply_slice_mask(x, y, z_grid, theta_grid)

    ax.plot_surface(
        x,
        y,
        z_grid,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def _plot_hsl_double_cone(ax):
    theta = np.linspace(0, 2 * np.pi, 360)
    lightness = np.linspace(0.0, 1.0, 160)
    theta_grid, lightness_grid = np.meshgrid(theta, lightness)

    radius = 1.0 - np.abs(2.0 * lightness_grid - 1.0)
    radius = np.clip(radius, 0.0, 1.0)

    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = lightness_grid

    hue_norm = (theta_grid / (2 * np.pi)) % 1.0
    sat = radius  # saturation tapers toward the axis, keeping the apex white
    colors = _hls_to_rgb_array(hue_norm, lightness_grid, sat)

    x, y, z = _apply_slice_mask(x, y, z, theta_grid)

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.scatter([0], [0], [0], color='black', s=60, depthshade=False)
    ax.scatter([0], [0], [1], color='white', edgecolors='black', s=60, depthshade=False)
    _plot_hsl_slice_plane(ax)
    _plot_slice_boundaries(ax, shape='hsl')
    ax.set_title('HSL Double Cone', pad=16)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _plot_hsv_slice_plane(ax, angle: float = SLICE_FACE_ANGLE):
    sat = np.linspace(0.0, 1.0, 200)
    value = np.linspace(0.0, 1.0, 200)
    sat_grid, value_grid = np.meshgrid(sat, value)

    hue = SLICE_FACE_HUE
    hsv = np.stack(
        [
            np.full_like(sat_grid, hue),
            sat_grid,
            value_grid,
        ],
        axis=-1,
    )
    colors = hsv_to_rgb(hsv)

    x = sat_grid * np.cos(angle)
    y = sat_grid * np.sin(angle)
    z = value_grid

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def _plot_hsl_slice_plane(ax, angle: float = SLICE_FACE_ANGLE):
    lightness = np.linspace(0.0, 1.0, 200)
    saturation = np.linspace(0.0, 1.0, 200)
    sat_grid, lightness_grid = np.meshgrid(saturation, lightness)

    hue = SLICE_FACE_HUE
    colors = _hls_to_rgb_array(
        np.full_like(lightness_grid, hue),
        lightness_grid,
        sat_grid,
    )

    radius_max = 1.0 - np.abs(2.0 * lightness_grid - 1.0)
    radius = sat_grid * radius_max
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = lightness_grid

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def _plot_slice_boundaries(ax, shape: str):
    boundary_angles = (
        SLICE_FACE_ANGLE,
        SLICE_MAX_ANGLE,
    )
    if shape == 'hsv':
        z = np.linspace(0.0, 1.0, 80)
        for angle in boundary_angles:
            x = np.full_like(z, np.cos(angle))
            y = np.full_like(z, np.sin(angle))
            ax.plot3D(x, y, z, color='#333333', linewidth=1.4, linestyle='--')
    elif shape == 'hsl':
        lightness = np.linspace(0.0, 1.0, 120)
        radius = 1.0 - np.abs(2.0 * lightness - 1.0)
        for angle in boundary_angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            ax.plot3D(x, y, lightness, color='#333333', linewidth=1.4, linestyle='--')


def main():
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax_hsv = fig.add_subplot(1, 2, 1, projection='3d')
    ax_hsl = fig.add_subplot(1, 2, 2, projection='3d')

    _plot_hsv_cylinder(ax_hsv)
    _plot_hsl_double_cone(ax_hsl)

    for ax in (ax_hsv, ax_hsl):
        ax.view_init(elev=25, azim=35)

    output_path = Path('hsv_hsl_.svg')
    plt.savefig(output_path, transparent=True)
    print(f'Saved diagram to {output_path.resolve()}')


if __name__ == '__main__':
    main()
