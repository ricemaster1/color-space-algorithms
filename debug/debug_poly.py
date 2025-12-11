"""Debug using Poly3DCollection for explicit face control."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

n = 20  # smaller for Poly3DCollection
slice_center = 0.0
slice_half = np.radians(65)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def make_wall_polys(angle, hue, n):
    """Create polygons for a wall at given angle with given hue."""
    sat = np.linspace(0, 1, n)
    value = np.linspace(0, 1, n)
    
    verts = []
    colors = []
    
    for i in range(n - 1):
        for j in range(n - 1):
            # Four corners of this quad
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
            
            # Color at center of this quad
            s_mid = (s0 + s1) / 2
            v_mid = (v0 + v1) / 2
            hsv = np.array([[[hue, s_mid, v_mid]]])
            rgb = hsv_to_rgb(hsv)[0, 0]
            colors.append(rgb)
    
    return verts, colors

# Left wall
angle1 = slice_center - slice_half
hue1 = ((angle1 % (2 * np.pi)) / (2 * np.pi)) % 1.0
verts1, colors1 = make_wall_polys(angle1, hue1, n)
poly1 = Poly3DCollection(verts1, facecolors=colors1, edgecolors='none')
ax.add_collection3d(poly1)

# Right wall  
angle2 = slice_center + slice_half
hue2 = ((angle2 % (2 * np.pi)) / (2 * np.pi)) % 1.0
verts2, colors2 = make_wall_polys(angle2, hue2, n)
poly2 = Poly3DCollection(verts2, facecolors=colors2, edgecolors='none')
ax.add_collection3d(poly2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
ax.set_title(f"Poly3D: Left hue={hue1:.3f}, Right hue={hue2:.3f}")
ax.view_init(elev=25, azim=35)
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
fig.savefig("debug_poly.png", dpi=150)
print(f"Left hue={hue1:.3f} (should be magenta)")
print(f"Right hue={hue2:.3f} (should be yellow)")
print("Saved debug_poly.png")
