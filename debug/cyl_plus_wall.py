"""Minimal test - cylinder + left wall only with proper z-sorting."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

slice_half = np.radians(65)
angle = -slice_half  # -65 degrees
hue = 0.819  # magenta

all_verts = []
all_colors = []

# First add the cylinder polygons
def add_cylinder_polys(theta_start, theta_end, n_theta, n_value):
    theta = np.linspace(theta_start, theta_end, n_theta)
    value = np.linspace(0, 1, n_value)
    
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
            h = (t_mid / (2 * np.pi)) % 1.0
            hsv = np.array([[[h, 1.0, v_mid]]])
            rgb = hsv_to_rgb(hsv)[0, 0]
            all_colors.append(rgb)

theta_start = slice_half  # +65
theta_end = -slice_half + 2 * np.pi  # 295
add_cylinder_polys(theta_start, theta_end, 60, 20)

# Add the left wall polygons
n = 30
sat = np.linspace(0, 1, n)
value = np.linspace(0, 1, n)

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

# Sort all polygons by distance from camera (painter's algorithm)
# Camera is at roughly (1, 1, 1) direction based on view_init(elev=25, azim=35)
cam_dir = np.array([np.cos(np.radians(35)) * np.cos(np.radians(25)),
                    np.sin(np.radians(35)) * np.cos(np.radians(25)),
                    np.sin(np.radians(25))])

def poly_depth(quad):
    center = np.mean(quad, axis=0)
    return np.dot(center, cam_dir)

# Sort by depth (furthest first for painter's algorithm)
sorted_indices = sorted(range(len(all_verts)), key=lambda i: poly_depth(all_verts[i]))
sorted_verts = [all_verts[i] for i in sorted_indices]
sorted_colors = [all_colors[i] for i in sorted_indices]

poly = Poly3DCollection(sorted_verts, facecolors=sorted_colors, edgecolors='none')
ax.add_collection3d(poly)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
ax.set_title(f"Z-sorted: Cylinder + left wall (hue={hue})")
ax.view_init(elev=25, azim=35)
ax.set_box_aspect((1, 1, 1))
fig.savefig("cyl_plus_wall.png", dpi=150)
print("Saved cyl_plus_wall.png - wall should be MAGENTA")
