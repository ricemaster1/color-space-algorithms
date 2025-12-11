#!/usr/bin/env python3
"""
hsv_hsl_figures_publication.py

Ultra-clean figure generator for HSV cylinder and HSL double-cone.
Generates publication-quality PNGs and a 2D SVG schematic.

Features:
- Correct inner slice plane sampling (saturation vs radius vs value/lightness)
- Clean masking of the removed wedge
- Higher sampling density and anti-alias control
- Annotation utilities and export targets
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import colorsys

# ----------------------
# Configuration
# ----------------------

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# Geometry / slice
SLICE_CENTER_ANGLE = 1.5                # radians
SLICE_HALF_ANGLE = np.deg2rad(40)       # half wedge angle
SLICE_FACE_ANGLE = SLICE_CENTER_ANGLE + SLICE_HALF_ANGLE
SLICE_FACE_HUE = ((SLICE_FACE_ANGLE % (2 * np.pi)) / (2 * np.pi)) % 1.0

# Sampling resolution (increase for higher fidelity)
CYLINDER_THETA_SAMPLES = 720
CYLINDER_Z_SAMPLES = 240

DISK_RADIUS_SAMPLES = 720
DISK_ANGLE_SAMPLES = 720

SLICE_S_SAMPLES = 360
SLICE_VL_SAMPLES = 360

# Figure output sizes
PNG_DPI = 300
FIGSIZE = (12, 6)   # inches

# ----------------------
# Helpers
# ----------------------


def hue_wrap_delta(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    Minimal circular difference between hue arrays in [0,1] domain.
    Returns absolute delta in fraction of full circle.
    """
    raw = np.abs(h1 - h2)
    return np.minimum(raw, 1.0 - raw)


def hsl_to_rgb_array(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Vectorized HSL (colorsys uses H, L, S order) -> RGB in [0,1].
    h,s,l arrays broadcast-compatible.
    """
    # colorsys.hls_to_rgb expects scalars; vectorize
    flat_h = h.ravel()
    flat_s = s.ravel()
    flat_l = l.ravel()
    fn = np.vectorize(colorsys.hls_to_rgb, otypes=[float, float, float])
    r, g, b = fn(flat_h, flat_l, flat_s)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = rgb.reshape(h.shape + (3,))
    return rgb


def hsv_array_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Use matplotlib's hsv_to_rgb on an NxMx3 array assembled from h,s,v in [0,1].
    """
    stacked = np.stack([h, s, v], axis=-1)
    return hsv_to_rgb(stacked)


def slice_mask(theta_grid: np.ndarray, center: float = SLICE_CENTER_ANGLE,
               half_angle: float = SLICE_HALF_ANGLE) -> np.ndarray:
    """
    Boolean mask True if point is inside the removed wedge (should be hidden).
    theta_grid in radians.
    """
    wrapped = (theta_grid - center + np.pi) % (2 * np.pi) - np.pi
    return np.abs(wrapped) < half_angle


# ----------------------
# Geometry generation
# ----------------------


def make_hsv_cylinder(res_theta: int = CYLINDER_THETA_SAMPLES,
                      res_z: int = CYLINDER_Z_SAMPLES) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parametric HSV cylinder lateral surface.
    Returns x,y,z coordinates and facecolors (RGB).
    Cylinder: radius = 1
    Hue: theta / (2*pi)
    Sat: 1 (lateral surface)
    Value: z
    """
    theta = np.linspace(0, 2 * np.pi, res_theta)
    z = np.linspace(0.0, 1.0, res_z)
    theta_grid, z_grid = np.meshgrid(theta, z)  # shape (res_z, res_theta)

    x = np.cos(theta_grid)
    y = np.sin(theta_grid)
    hue = (theta_grid / (2 * np.pi)) % 1.0
    sat = np.ones_like(hue)
    val = z_grid

    colors = hsv_array_to_rgb(hue, sat, val)  # (res_z, res_theta, 3)
    return x, y, z_grid, colors


def make_hsv_endcap(z_level: float, res_r: int = DISK_RADIUS_SAMPLES, res_theta: int = DISK_ANGLE_SAMPLES):
    """
    Disk at z = z_level for HSV: saturation = radius, hue = theta, value = z_level.
    Produces radial saturation gradient.
    """
    r = np.linspace(0, 1.0, res_r)
    theta = np.linspace(0, 2 * np.pi, res_theta)
    r_grid, theta_grid = np.meshgrid(r, theta)

    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    z = np.full_like(x, z_level)

    hue = (theta_grid / (2 * np.pi)) % 1.0
    sat = r_grid
    val = np.full_like(r_grid, z_level)

    colors = hsv_array_to_rgb(hue, sat, val)
    return x, y, z, colors, theta_grid


def make_hsv_slice_plane(face_angle: float = SLICE_FACE_ANGLE,
                         hue_face: float = SLICE_FACE_HUE,
                         res_s: int = SLICE_S_SAMPLES,
                         res_v: int = SLICE_VL_SAMPLES):
    """
    The inner exposed face for HSV: a rectangular domain (saturation, value)
    mapped to a flat plane at angle = face_angle with 0<=s<=1 and 0<=v<=1.
    This is the correct 'inner wall' with radial saturation gradient.
    """
    s = np.linspace(0.0, 1.0, res_s)
    v = np.linspace(0.0, 1.0, res_v)
    s_grid, v_grid = np.meshgrid(s, v)

    hue_grid = np.full_like(s_grid, hue_face)
    colors = hsv_array_to_rgb(hue_grid, s_grid, v_grid)

    x = s_grid * np.cos(face_angle)
    y = s_grid * np.sin(face_angle)
    z = v_grid

    return x, y, z, colors, s_grid, v_grid


def make_hsl_double_cone(res_theta: int = CYLINDER_THETA_SAMPLES, res_l: int = CYLINDER_Z_SAMPLES):
    """
    HSL double cone geometry: radius depends on L: radius = 1 - |2L - 1|.
    Saturation follows radius profile to preserve white apex at L=0 and L=1.
    """
    theta = np.linspace(0, 2 * np.pi, res_theta)
    L = np.linspace(0.0, 1.0, res_l)
    theta_grid, L_grid = np.meshgrid(theta, L)

    radius = 1.0 - np.abs(2.0 * L_grid - 1.0)
    radius = np.clip(radius, 0.0, 1.0)

    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = L_grid

    hue = (theta_grid / (2 * np.pi)) % 1.0
    sat = radius  # saturation tapers toward axis
    colors = hsl_to_rgb_array(hue, sat, L_grid)

    return x, y, z, colors


def make_hsl_slice_plane(face_angle: float = SLICE_FACE_ANGLE,
                         hue_face: float = SLICE_FACE_HUE,
                         res_s: int = SLICE_S_SAMPLES,
                         res_l: int = SLICE_VL_SAMPLES):
    """
    HSL slice plane: saturated perimeter is limited by available radius for each lightness.
    We construct saturation fraction s_frac in [0,1] then multiply by radius_max = 1 - |2L - 1|
    """
    s_frac = np.linspace(0.0, 1.0, res_s)
    L = np.linspace(0.0, 1.0, res_l)
    s_grid, L_grid = np.meshgrid(s_frac, L)

    radius_max = 1.0 - np.abs(2.0 * L_grid - 1.0)
    radius = s_grid * radius_max

    hue_grid = np.full_like(radius, hue_face)
    sat_grid = s_grid * 1.0  # saturation fraction; actual chroma depends on radius_max in geometry

    # Convert by forming H, L, S where S is the effective saturation fraction.
    colors = hsl_to_rgb_array(hue_grid, sat_grid, L_grid)

    x = radius * np.cos(face_angle)
    y = radius * np.sin(face_angle)
    z = L_grid

    return x, y, z, colors, s_grid, L_grid


# ----------------------
# Rendering utilities
# ----------------------


def apply_slice_mask_xy(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta_grid: np.ndarray):
    """
    Mask coordinates and return NaN-ed arrays where the wedge is removed.
    theta_grid in radians matches the x,y param domain.
    """
    mask = slice_mask(theta_grid)
    return (
        np.where(mask, np.nan, x),
        np.where(mask, np.nan, y),
        np.where(mask, np.nan, z),
    )


def render_hsv_axis(ax):
    # Cylinder body
    x, y, z, colors = make_hsv_cylinder()
    # Mask the wedge
    # Need theta_grid used in cylinder creation to mask correctly
    theta = np.linspace(0, 2 * np.pi, CYLINDER_THETA_SAMPLES)
    z_lin = np.linspace(0.0, 1.0, CYLINDER_Z_SAMPLES)
    theta_grid, z_grid = np.meshgrid(theta, z_lin)
    x_masked, y_masked, z_masked = apply_slice_mask_xy(x, y, z, theta_grid)

    ax.plot_surface(
        x_masked, y_masked, z_masked,
        rcount=colors.shape[0], ccount=colors.shape[1],
        facecolors=colors, linewidth=0, antialiased=False, shade=False, edgecolor='none'
    )

    # Inner face: true saturation vs value rectangle
    xs, ys, zs, face_colors, s_grid, v_grid = make_hsv_slice_plane()
    ax.plot_surface(
        xs, ys, zs,
        rcount=face_colors.shape[0], ccount=face_colors.shape[1],
        facecolors=face_colors, linewidth=0, antialiased=False, shade=False, edgecolor='none'
    )

    # Caps: top and bottom disks (masked)
    dx, dy, dz, cap_colors, theta_grid_disk = make_hsv_endcap(0.0)
    dx_mask, dy_mask, dz_mask = apply_slice_mask_xy(dx, dy, dz, theta_grid_disk)
    ax.plot_surface(dx_mask, dy_mask, dz_mask, facecolors=cap_colors, linewidth=0, antialiased=False, shade=False, edgecolor='none')

    dx2, dy2, dz2, cap_colors2, theta_grid_disk2 = make_hsv_endcap(1.0)
    dx2_mask, dy2_mask, dz2_mask = apply_slice_mask_xy(dx2, dy2, dz2, theta_grid_disk2)
    ax.plot_surface(dx2_mask, dy2_mask, dz2_mask, facecolors=cap_colors2, linewidth=0, antialiased=False, shade=False, edgecolor='none')

    # Slice boundary outlines
    boundary_angles = (SLICE_CENTER_ANGLE + SLICE_HALF_ANGLE, SLICE_CENTER_ANGLE - SLICE_HALF_ANGLE)
    z_line = np.linspace(0.0, 1.0, 200)
    for angle in boundary_angles:
        x_line = np.cos(angle) * np.ones_like(z_line)
        y_line = np.sin(angle) * np.ones_like(z_line)
        ax.plot3D(x_line, y_line, z_line, color='#222222', linewidth=1.0, linestyle='--')

    ax.set_title("HSV Cylinder (saturation radius, value upward)")
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_hsl_axis(ax):
    x, y, z, colors = make_hsl_double_cone()
    # For masking we need theta_grid for cone domain
    theta = np.linspace(0, 2 * np.pi, CYLINDER_THETA_SAMPLES)
    L = np.linspace(0.0, 1.0, CYLINDER_Z_SAMPLES)
    theta_grid, L_grid = np.meshgrid(theta, L)
    radius = 1.0 - np.abs(2.0 * L_grid - 1.0)
    x_cone = radius * np.cos(theta_grid)
    y_cone = radius * np.sin(theta_grid)
    z_cone = L_grid
    x_mask, y_mask, z_mask = apply_slice_mask_xy(x_cone, y_cone, z_cone, theta_grid)

    ax.plot_surface(
        x_mask, y_mask, z_mask,
        rcount=colors.shape[0], ccount=colors.shape[1],
        facecolors=colors, linewidth=0, antialiased=False, shade=False, edgecolor='none'
    )

    # Inner slice: HSL valid region mapped to face plane
    xs, ys, zs, face_colors, s_grid, l_grid = make_hsl_slice_plane()
    ax.plot_surface(
        xs, ys, zs,
        rcount=face_colors.shape[0], ccount=face_colors.shape[1],
        facecolors=face_colors, linewidth=0, antialiased=False, shade=False, edgecolor='none'
    )

    # Outline boundaries along cone surface
    boundary_angles = (SLICE_CENTER_ANGLE + SLICE_HALF_ANGLE, SLICE_CENTER_ANGLE - SLICE_HALF_ANGLE)
    for angle in boundary_angles:
        # radius for each L
        Ls = np.linspace(0.0, 1.0, 300)
        radius_line = 1.0 - np.abs(2.0 * Ls - 1.0)
        x_line = radius_line * np.cos(angle)
        y_line = radius_line * np.sin(angle)
        ax.plot3D(x_line, y_line, Ls, color='#222222', linewidth=1.0, linestyle='--')

    ax.set_title("HSL Double Cone (lightness vertical)")
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


# ----------------------
# Vector 2D Schematic export
# ----------------------


def export_2d_schematic_svg(path: Path):
    """
    Export a vector-like 2D illustration (hue wheel, small HSV vs HSL panels) as SVG.
    This uses imshow for the color arrays but saves as SVG. Many backends will embed
    raster data in SVG. It still serves as a precise schematic for papers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    # Hue wheel (top-down)
    r = np.linspace(0, 1, 600)
    theta = np.linspace(0, 2 * np.pi, 1200)
    r_grid, theta_grid = np.meshgrid(r, theta)
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    hue = (theta_grid / (2 * np.pi)) % 1.0
    sat = r_grid
    val = np.ones_like(r_grid) * 1.0
    wheel = hsv_array_to_rgb(hue, sat, val)
    axes[0].imshow(wheel, origin='lower', extent=[-1, 1, -1, 1])
    axes[0].set_title("Hue Wheel")
    axes[0].axis('off')

    # HSV cross-section (s vs v) at face hue
    s = np.linspace(0, 1, 400)
    v = np.linspace(0, 1, 400)
    s_grid, v_grid = np.meshgrid(s, v)
    hue_face = np.full_like(s_grid, SLICE_FACE_HUE)
    hsv_face = hsv_array_to_rgb(hue_face, s_grid, v_grid)
    axes[1].imshow(hsv_face, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[1].set_title("HSV Slice (saturation vs value)")
    axes[1].set_xlabel("saturation")
    axes[1].set_ylabel("value")

    # HSL cross-section (s_frac vs lightness) at face hue
    s_frac = np.linspace(0, 1, 400)
    L = np.linspace(0, 1, 400)
    s_frac_grid, L_grid = np.meshgrid(s_frac, L)
    # For HSL we render as s_frac -> real chroma overlay is geometry dependent; show HLS sampling
    hsl_face = hsl_to_rgb_array(np.full_like(s_frac_grid, SLICE_FACE_HUE), s_frac_grid, L_grid)
    axes[2].imshow(hsl_face, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[2].set_title("HSL Slice (s_frac vs lightness)")
    axes[2].set_xlabel("saturation fraction")
    axes[2].set_ylabel("lightness")

    plt.savefig(path, format='svg')
    plt.close(fig)


# ----------------------
# Main entry
# ----------------------


def main():
    # Create figure and axes
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    ax_hsv = fig.add_subplot(1, 2, 1, projection='3d')
    ax_hsl = fig.add_subplot(1, 2, 2, projection='3d')

    # Render both axes with correct geometry and slicing
    render_hsv_axis(ax_hsv)
    render_hsl_axis(ax_hsl)

    # Nice camera and export
    for ax in (ax_hsv, ax_hsl):
        ax.view_init(elev=30, azim=35)

    png_out = OUT_DIR / "hsv_hsl_cylinders.png"
    plt.savefig(png_out, dpi=PNG_DPI)
    plt.close()

    # Export a vector-like 2D schematic for inclusion in papers
    svg_out = OUT_DIR / "hsv_hsl_schematic.svg"
    export_2d_schematic_svg(svg_out)

    print(f"Saved raster figure to {png_out.resolve()}")
    print(f"Saved schematic SVG to {svg_out.resolve()}")


if __name__ == "__main__":
    main()