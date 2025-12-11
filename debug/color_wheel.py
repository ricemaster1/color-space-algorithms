import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 360)
r = 1

fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
for deg in range(360):
    color = plt.cm.hsv(deg/360)
    ax.bar(np.deg2rad(deg), 1, width=np.deg2rad(1), color=color)

ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig("hue_wheel.svg")