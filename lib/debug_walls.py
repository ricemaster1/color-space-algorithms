"""Debug script to test wall rendering."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

n = 50
sat = np.linspace(0, 1, n)
value = np.linspace(0, 1, n)
sat_grid, value_grid = np.meshgrid(sat, value)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection='3d')

# Wall 1: Magenta (hue=0.819) at angle -65 degrees
hue1 = 0.819
angle1 = np.radians(-65)
hsv1 = np.stack([np.full_like(sat_grid, hue1), sat_grid, value_grid], axis=-1)
colors1 = hsv_to_rgb(hsv1)
x1 = sat_grid * np.cos(angle1)
y1 = sat_grid * np.sin(angle1)
ax.plot_surface(x1, y1, value_grid, facecolors=colors1, shade=False)

# Wall 2: Yellow/orange (hue=0.181) at angle +65 degrees
hue2 = 0.181
angle2 = np.radians(65)
hsv2 = np.stack([np.full_like(sat_grid, hue2), sat_grid, value_grid], axis=-1)
colors2 = hsv_to_rgb(hsv2)
x2 = sat_grid * np.cos(angle2)
y2 = sat_grid * np.sin(angle2)
ax.plot_surface(x2, y2, value_grid, facecolors=colors2, shade=False)

ax.set_title(f"Wall1: hue={hue1:.3f} (magenta)  Wall2: hue={hue2:.3f} (yellow)")
ax.view_init(elev=25, azim=35)
fig.savefig("debug_walls.png", dpi=100)
print("Saved debug_walls.png")
