import numpy as np
import matplotlib.pyplot as plt
import spectra

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def hsv_to_rgb(h, s, v):
    c = spectra.hsv(h, s, v)
    r, g, b = c.rgb
    return (r, g, b)

def hsl_to_rgb(h, s, l):
    c = spectra.hsl(h, s, l)
    r, g, b = c.rgb
    return (r, g, b)

def cylinder_slice(res=512, mode="hsv", level=1.0):
    """
    Generates a horizontal slice of an HSL or HSV cylinder.
    level = V (HSV) or L (HSL)
    """
    # Create S (radius) and H (angle)
    w = res
    h = res

    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    A = (np.degrees(np.arctan2(Y, X)) + 360) % 360

    # Only fill inside unit circle
    img = np.ones((h, w, 3))
    for i in range(h):
        for j in range(w):
            if R[i, j] <= 1:
                S = R[i, j]
                H = A[i, j]
                if mode == "hsv":
                    rgb = hsv_to_rgb(H, S, level)
                else:
                    rgb = hsl_to_rgb(H, S, level)
                img[i, j] = rgb
            else:
                img[i, j] = (1, 1, 1)  # background
    return img

# ---------------------------------------------------------
# Draw cylinders: three slices each, stacked vertically
# ---------------------------------------------------------

def draw_cylinder(mode="hsv", title="HSV Cylinder"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    levels = [0.25, 0.5, 0.75]

    for ax, L in zip(axes, levels):
        img = cylinder_slice(res=512, mode=mode, level=L)
        ax.imshow(img, origin='lower')
        ax.set_title(f"{mode.upper()} Slice {L:.2f}")
        ax.axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{mode}_cylinder.png", dpi=200)
    plt.show()

# ---------------------------------------------------------
# Top-down hue wheel
# ---------------------------------------------------------

def draw_hue_wheel():
    img = cylinder_slice(res=600, mode="hsv", level=1.0)
    plt.figure(figsize=(5,5))
    plt.imshow(img, origin='lower')
    plt.title("Hue Wheel (Shared HSL/HSV)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("hue_wheel.png", dpi=200)
    plt.show()

# ---------------------------------------------------------
# Side-by-side comparison of HSV vs HSL at mid-slice
# ---------------------------------------------------------

def draw_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    hsv_img = cylinder_slice(res=600, mode="hsv", level=0.5)
    hsl_img = cylinder_slice(res=600, mode="hsl", level=0.5)

    axes[0].imshow(hsv_img)
    axes[1].imshow(hsl_img)

    axes[0].set_title("HSV Slice (V = 0.5)")
    axes[1].set_title("HSL Slice (L = 0.5)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("hsv_vs_hsl_comparison.png", dpi=200)
    plt.show()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    draw_cylinder(mode="hsv", title="HSV Cylinder Slices (V)")
    draw_cylinder(mode="hsl", title="HSL Cylinder Slices (L)")
    draw_hue_wheel()
    draw_comparison()