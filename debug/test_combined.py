"""Test everything using Poly3DCollection only."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

n = 30
slice_center = 0.0
slice_half = np.radians(65)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def make_hsv_wall(angle, hue, n):
    sat = np.linspace(0, 1, n)
    value = np.linspace(0, 1, n)
    
    verts = []
    colors = []
    
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
            verts.append(quad)
            
            s_mid = (s0 + s1) / 2
            v_mid = (v0 + v1) / 2
            hsv = np.array([[[hue, s_mid, v_mid]]])
            rgb = hsv_to_rgb(hsv)[0, 0]
            colors.append(rgb)
    
    return verts, colors

def make_cylinder_polys(theta_start, theta_end, n_theta, n_value):
    """Create cylinder using Poly3DCollection."""
    theta = np.linspace(theta_start, theta_end, n_theta)
    value = np.linspace(0, 1, n_value)
    
    verts = []
    colors = []
    
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
            verts.append(quad)
            
            t_mid = (t0 + t1) / 2
            v_mid = (v0 + v1) / 2
            hue = (t_mid / (2 * np.pi)) % 1.0
            hsv = np.array([[[hue, 1.0, v_mid]]])
            rgb = hsv_to_rgb(hsv)[0, 0]
            colors.append(rgb)
    
    return verts, colors

# Draw cylinder using Poly3DCollection
theta_start = slice_center + slice_half
theta_end = slice_center - slice_half + 2 * np.pi
verts_cyl, colors_cyl = make_cylinder_polys(theta_start, theta_end, 90, 30)
poly_cyl = Poly3DCollection(verts_cyl, facecolors=colors_cyl, edgecolors='none')
ax.add_collection3d(poly_cyl)

# Left wall
angle1 = slice_center - slice_half
hue1 = ((angle1 % (2 * np.pi)) / (2 * np.pi)) % 1.0
verts1, colors1 = make_hsv_wall(angle1, hue1, n)
poly1 = Poly3DCollection(verts1, facecolors=colors1, edgecolors='none')
ax.add_collection3d(poly1)

# Right wall
angle2 = slice_center + slice_half
hue2 = ((angle2 % (2 * np.pi)) / (2 * np.pi)) % 1.0
verts2, colors2 = make_hsv_wall(angle2, hue2, n)
poly2 = Poly3DCollection(verts2, facecolors=colors2, edgecolors='none')
ax.add_collection3d(poly2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
ax.set_title(f"All Poly3D: Left={hue1:.3f}, Right={hue2:.3f}")
ax.view_init(elev=25, azim=35)
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
fig.savefig("test_combined.png", dpi=150)
print(f"Left hue={hue1:.3f}, Right hue={hue2:.3f}")
print("Saved test_combined.png")
