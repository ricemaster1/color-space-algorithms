"""Debug script to test wall rendering with cylinder."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

n = 50
slice_center = 0.0
slice_half = np.radians(65)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Draw the cylinder first
theta_start = slice_center + slice_half  # +65°
theta_end = slice_center - slice_half + 2 * np.pi  # 295°
theta = np.linspace(theta_start, theta_end, 180)
value_cyl = np.linspace(0, 1, 50)
theta_grid, value_grid_cyl = np.meshgrid(theta, value_cyl)

x_cyl = np.cos(theta_grid)
y_cyl = np.sin(theta_grid)
z_cyl = value_grid_cyl

hues_cyl = (theta_grid / (2 * np.pi)) % 1.0
hsv_cyl = np.stack([hues_cyl, np.ones_like(hues_cyl), value_grid_cyl], axis=-1)
colors_cyl = hsv_to_rgb(hsv_cyl)

ax.plot_surface(x_cyl, y_cyl, z_cyl, facecolors=colors_cyl, shade=False, linewidth=0)

# Draw top cap
r = np.linspace(0, 1, n)
theta_cap = np.linspace(theta_start, theta_end, 180)
r_grid, theta_cap_grid = np.meshgrid(r, theta_cap)
x_cap = r_grid * np.cos(theta_cap_grid)
y_cap = r_grid * np.sin(theta_cap_grid)
z_cap = np.ones_like(x_cap)
hues_cap = (theta_cap_grid / (2 * np.pi)) % 1.0
hsv_cap = np.stack([hues_cap, r_grid, np.ones_like(r_grid)], axis=-1)
colors_cap = hsv_to_rgb(hsv_cap)
ax.plot_surface(x_cap, y_cap, z_cap, facecolors=colors_cap, shade=False, linewidth=0)

# Draw walls last
sat = np.linspace(0, 1, n)
value = np.linspace(0, 1, n)
sat_grid, value_grid = np.meshgrid(sat, value)

# Left wall: at angle (center - half) = -65 degrees = 295 degrees
angle1 = slice_center - slice_half
hue1 = ((angle1 % (2 * np.pi)) / (2 * np.pi)) % 1.0
hsv1 = np.stack([np.full_like(sat_grid, hue1), sat_grid, value_grid], axis=-1)
colors1 = hsv_to_rgb(hsv1)
x1 = sat_grid * np.cos(angle1)
y1 = sat_grid * np.sin(angle1)
ax.plot_surface(x1, y1, value_grid, facecolors=colors1, shade=False, linewidth=0)

# Right wall: at angle (center + half) = +65 degrees
angle2 = slice_center + slice_half
hue2 = ((angle2 % (2 * np.pi)) / (2 * np.pi)) % 1.0
hsv2 = np.stack([np.full_like(sat_grid, hue2), sat_grid, value_grid], axis=-1)
colors2 = hsv_to_rgb(hsv2)
x2 = sat_grid * np.cos(angle2)
y2 = sat_grid * np.sin(angle2)
ax.plot_surface(x2, y2, value_grid, facecolors=colors2, shade=False, linewidth=0)

ax.set_title(f"Left hue={hue1:.3f} (magenta), Right hue={hue2:.3f} (yellow)")
ax.view_init(elev=25, azim=35)
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
fig.savefig("debug_walls.png", dpi=150)
print(f"Left angle: {np.degrees(angle1):.1f}°, hue={hue1:.3f}")
print(f"Right angle: {np.degrees(angle2):.1f}°, hue={hue2:.3f}")
print("Saved debug_walls.png")
