# `rgb_to_lab.py`

The `algorithms/color_transforms/rgb_to_lab/rgb_to_lab.py` module converts images to ARMlite-compatible sprites using the **CIE L*a*b*** color space for perceptually uniform palette matching.

> **Note:** This document uses LaTeX math notation (matrices, piecewise functions, etc.) that GitHub's Markdown renderer may not display correctly. For the best reading experience, use VS Code with the Markdown Preview Enhanced extension, Typora, or another full-featured Markdown viewer.

---

## CLI reference (template)
| Option | Description |
| --- | --- |
| `image` | Path to input image (required). Any format PIL can open. |
| `output` | Assembly output path. Defaults to `lab.s`. If a directory is provided, uses auto-generated filename. |
| `-w / --weights L,a,b` | Comma-separated floats controlling the L*, a*, b* weighting metric. Default: `1,1,1`. |
| `-a / --auto` | **Auto-match weights** to minimize RGB error using numpy vectorized grid search. Requires numpy. |
| `-o / --output-file` | Output path (alternative to positional argument). |
| `-h / --help` | Show help message. |

### Sample usage

```bash
# Basic conversion with default weights (1,1,1)
python rgb_to_lab.py photo.png

# Custom weights emphasizing lightness
python rgb_to_lab.py photo.png -w 2,1,1

# Auto-optimize weights for best color match
python rgb_to_lab.py photo.png --auto

# Specify output path
python rgb_to_lab.py photo.png -o output/sprite.s

# Via armlite.py
armlite.py convert photo.png -a rgb_to_lab -O output/
armlite.py convert photo.png -a rgb_to_lab --algo-extra="--auto"
```

<!--
![Auto-matched LAB result](img/auto-lab-fig.1.png)
<small>Figure 1: Auto-matched LAB weights preserve perceptual color differences while minimizing overall RGB error.</small>
-->

---

## Weight Tuner GUI (placeholder)

_This section will describe the GUI for interactive weight exploration, preview, and export. (To be completed)_

---

## Color Theory Background

### Why L*a*b*?

The **CIE L*a*b*** (CIELAB) color space was designed to be **perceptually uniform**—meaning that a given numerical change in color values produces a roughly equal perceived change in color, regardless of where in the color space you are.

- In RGB, equal numeric steps do not produce equal perceived differences
- In HSV/HSL, hue differences near gray are exaggerated
- L*a*b* is device-independent and models human vision more closely

### The Transform Pipeline

```
sRGB → Linear RGB → CIE XYZ → CIE L*a*b*
```

1. **sRGB → Linear RGB**: Remove gamma correction
   $$
   C_{lin} = \begin{cases}
     \frac{C}{12.92} & C \leq 0.04045 \\
     \left(\frac{C + 0.055}{1.055}\right)^{2.4} & C > 0.04045
   \end{cases}
   $$

2. **Linear RGB → XYZ**: Apply 3×3 matrix (D65 illuminant)
   $$
   \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} =
   \begin{bmatrix}
     0.4124564 & 0.3575761 & 0.1804375 \\
     0.2126729 & 0.7151522 & 0.0721750 \\
     0.0193339 & 0.1191920 & 0.9503041
   \end{bmatrix}
   \begin{bmatrix} R_{lin} \\ G_{lin} \\ B_{lin} \end{bmatrix}
   $$

3. **XYZ → L*a*b***: Nonlinear transform with reference white
   $$
   f(t) = \begin{cases}
     t^{1/3} & t > \varepsilon \\
     \frac{\kappa t + 16}{116} & t \leq \varepsilon
   \end{cases}
   $$
   where $\varepsilon = 0.008856$, $\kappa = 903.3$

   $$
   L^* = 116 f\left(\frac{Y}{Y_n}\right) - 16 \\
   a^* = 500 \left[f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right] \\
   b^* = 200 \left[f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right)\right]
   $$

### D65 Reference White

The script uses the **D65 standard illuminant** (average daylight):
- $X_n = 0.95047$
- $Y_n = 1.00000$
- $Z_n = 1.08883$

---

## L*a*b* vs HSV/HSL

| Aspect | L*a*b* | HSV/HSL |
|--------|--------|---------|
| **Perceptual uniformity** | ✓ Designed for it | ✗ Not uniform |
| **Hue handling** | Implicit in a*/b* | Explicit H channel |
| **Lightness accuracy** | L* correlates well with human perception | V/L less accurate |
| **Computation cost** | Higher (matrix + nonlinear) | Lower |
| **Best for** | Photo-realistic images | Graphic art, icons |

### When to Use Which

- **L*a*b***: Photographs, images with subtle color gradients, skin tones
- **HSV/HSL**: Graphic art, images with distinct hue regions, cartoons

---

## Implementation Details

### Distance Metric

The script uses **weighted Euclidean distance** in L*a*b* space:

$$
d = \sqrt{(\Delta L^* \cdot w_L)^2 + (\Delta a^* \cdot w_a)^2 + (\Delta b^* \cdot w_b)^2}
$$

This is a simplified version of the CIE76 color difference formula. More sophisticated formulas exist (CIE94, CIEDE2000) but add complexity for marginal gains in a 147-color palette context.

### Quantization Process

For each pixel:
1. Convert pixel RGB → L*a*b*
2. Compute weighted distance to all 147 palette colors (pre-converted to L*a*b*)
3. Select the palette color with minimum distance
4. Output the color name to the assembly grid

### Performance Notes

- Palette L*a*b* values are computed once and cached
- Auto-match uses numpy vectorization for ~10× speedup
- Image is resized to 128×96 before processing

---

## Sources

See [papers/references.bib](../papers/references.bib) for full BibTeX entries.

1. **CIE 15:2004** — *Colorimetry*, 3rd Edition. Commission Internationale de l'Éclairage, 2004. ISBN 3-901-906-33-9. The authoritative reference defining CIELAB (L*a*b*) color space and standard illuminants.
   - [Full text (Archive.org)](https://archive.org/details/gov.law.cie.15.2004)
2. **ICC.1:2004-10** — *Image technology colour management — Architecture, profile format and data structure* (Profile version 4.2.0.0). International Color Consortium, 2006. Defines Lab as profile connection space.
   - [Specification PDF](https://www.color.org/icc1v42.pdf)
3. MacEvoy, B. "Modern Color Models — CIELUV." *Handprint*. Comprehensive explanation of CIELAB/CIELUV history and color science.
   - [handprint.com](https://www.handprint.com/HP/WCL/color7.html#CIELUV)
4. Lindbloom, B. "Uniform Perceptual Lab (UPLab)." Discussion of perceptual non-uniformities in CIELAB and proposed improvements.
   - [brucelindbloom.com](http://www.brucelindbloom.com/index.html?UPLab.html)
5. Lindbloom, B. "Working Space Information." 3D representations of L*a*b* gamut and RGB working space comparisons.
   - [brucelindbloom.com](http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html)
6. Jain, A. K. (1989). *Fundamentals of Digital Image Processing*. Prentice Hall. ISBN 0-13-336165-9. Covers perceptual color differences and Euclidean distance in L*a*b*.
   - [Full text (Archive.org)](https://archive.org/details/fundamentalsofdi0000jain)
