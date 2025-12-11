"""Minimal test - just cylinder + left wall."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

angle = np.radians(-65)
hue = 0.819  # magenta

# Just ONE wall - no cylinder
n = 30
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

poly = Poly3DCollection(verts, facecolors=colors, edgecolors='none')
ax.add_collection3d(poly)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
ax.set_title(f"Single wall at -65deg, hue={hue}")
ax.view_init(elev=25, azim=35)
ax.set_box_aspect((1, 1, 1))
fig.savefig("single_wall.png", dpi=150)
print("Saved single_wall.png - should be MAGENTA")
