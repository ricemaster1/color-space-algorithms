# `rgb_to_hsv_hsl.py` 
# Deep Dive

> **Note:** This document uses LaTeX math notation (matrices, piecewise functions, etc.) that GitHub's Markdown renderer may not display correctly. For the best reading experience, use VS Code with the Markdown Preview Enhanced extension, Typora, or another full-featured Markdown viewer.

The `algorithms/color_transforms/src/rgb_to_hsv_hsl.py` module converts arbitrary imagery into ARMLite-compatible sprites by matching pixels in HSV or HSL space before emitting `.Resolution`, `.PixelScreen`, and per-pixel stores. This document narrates the color theory behind the script, explains every function, and provides research references plus assets you can reuse in blog posts or release notes.

---

## Cylindrical color context

### The RGB monitor gamut

Before exploring alternative color models, it's worth defining exactly what we mean by "RGB." Smith describes it as "the gamut of colors spanned by the red, green, and blue (RGB) electron guns exciting their respective phosphors. It is called the RGB monitor gamut" [@smith1978hsv, p. 12]. This physical grounding matters: RGB isn't an abstract mathematical space but "a set of alternative models of the RGB monitor gamut based on the perceptual variables hue (H), saturation (S), and value (V) or brightness (L)" [@smith1978hsv, p. 12].

Smith frames color mathematically: "a color is a vector in a (finite) 3-dimensional space where the dimensions R, G and B are called the primaries" [@smith1978hsv, p. 12]. This vector-space view means any linear transformation of the primaries yields another valid (if perhaps less intuitive) color space.

Digital displays approximate the continuous analog signal with discrete steps: "In computer graphics, the guns are digitally controlled, the full analog range of each gun being approximated by $n = 2^m$ distinct equally spaced values. For $m \ge 8$, most humans cannot perceive the difference between analog or digital control, the discrete and continuous become one perceptually" [@smith1978hsv, p. 12]. This perceptual equivalence at 8 bits per channel is why 24-bit "true color" became the standard—and why ARMLite's palette, even with far fewer colors, can still produce recognizable imagery when dithered intelligently.

#### Gamma correction and the linearity assumption

Smith's analysis assumes the monitor behaves linearly, but CRTs are inherently nonlinear: "the light intensity emitted by the cathode ray tube in a television monitor is a nonlinear function of its driving voltages. Hence the assumption of linearity implies the existence of a black box between the numbers used for digital input and the numbers actually used to digitally control the input voltages. This black box compensates for the nonlinearity of the cathode ray tube. It can be implemented as a simple lookup table and is called a gamma-correction, or compensation, table" [@smith1978hsv, p. 12].

This distinction matters for color transforms: the HSV and HSL formulas operate on *linear* RGB values. When working with typical 8-bit images (which are gamma-encoded for display), accurate color manipulation requires linearizing first, transforming, then re-encoding. For palette matching in ARMLite sprites, the error introduced by skipping linearization is usually acceptable—but for scientific visualization or color-critical work, proper gamma handling is essential.

### Why RGB isn't enough

RGB defines color as a point inside a cube—red, green, and blue axes meeting at right angles. That geometry mirrors how monitors mix light, but it fights against how humans *think* about color. We don't say "a bit more green channel"; we say "make it warmer" or "less saturated."

The RGB cube also has physical limits. As Joblove & Greenberg note, "the particular characteristics of three human receptor systems make it impossible for any such set of three primary colors to duplicate all colors" [@joblove1978, p. 20]. Colors outside the "color triangle" formed by the chosen primaries would require negative RGB values—mathematically valid but physically unrealizable. Still, "all colors which can be created can therefore be represented within a cubic volume in the all-positive octant of an orthogonal three-space whose axes are the rgb primaries" [@joblove1978, p. 21].

An important concept here is **metamerism**: "Different spectral distributions can produce the same color sensation, or receptor responses. This phenomenon is known as metamerism; 'different' colors which appear the same as a result are called metamers" [@joblove1978, p. 20]. This is why we can represent a continuous spectrum with just three channels—and why two images can look identical on screen yet differ when printed.

The relationship between additive (RGB) and subtractive (CMY) primaries follows a simple inversion:

$$
[r \; g \; b] = [1 \; 1 \; 1] - [c \; m \; y]
$$

**Chromaticity**—the quality of color independent of brightness—"is the function of the ratios between the primary colors" [@joblove1978, p. 21]. This ratio-based thinking leads naturally to the "angular component" Joblove & Greenberg describe: wrapping hue around a cylinder so that color relationships become geometric angles rather than cube coordinates.

### Alternative primaries: the IQY transformation

Because color forms a linear vector space, we can transform between coordinate systems. Smith notes that "a new set of primaries I, Q, and Y also form a linear space" [@smith1978hsv, p. 12]. The transformation from RGB to IQY (used in NTSC television encoding) is:

$$
\begin{bmatrix}
I \\
Q \\
Y
\end{bmatrix}
=
\begin{bmatrix}
0.60 & -0.28 & -0.32 \\
0.21 & -0.52 & 0.31 \\
0.30 & 0.59 & 0.11
\end{bmatrix}
\begin{bmatrix}
R \\
G \\
B
\end{bmatrix}
$$

This matrix illustrates a key principle: the choice of primaries is arbitrary as long as the transformation is invertible. YIQ separates luminance (Y) from chrominance (I, Q), which allowed early color television to remain backward-compatible with black-and-white sets. HSV and HSL take this idea further by making one axis explicitly perceptual (hue as an angle) rather than optimizing for broadcast engineering.

### The cylindrical reshape

HSV and HSL reshape the cube into cylinders. Both place **hue** on a circular axis (0–360°), so related colors—orange next to red, cyan next to blue—live as neighbors rather than at opposite corners. The vertical axis becomes **value** (HSV) or **lightness** (HSL), and the radial axis becomes **saturation**. The critical difference between the two models is what happens at the top:

- **HSV cylinder:** Full value (V = 1) means maximum brightness of each hue. Pure white sits at the center of the top cap, while saturated colors ring the edge. The bottom collapses to black regardless of hue or saturation.
- **HSL double-cone:** Lightness (L = 0.5) is where colors are most vivid; L = 0 is black and L = 1 is white. The shape tapers to points at both poles, which is why the cross-section forms a double cone.

This rearrangement is what makes HSV and HSL feel intuitive: artists can tweak hue without disturbing brightness, or desaturate toward gray without shifting the underlying color. The figure below—rendered with matplotlib—shows both models sliced open so you can see the interior gradient from white (center) to saturated hue (edge).

The CSS Color Module Level 4 specification [@w3c-color4] codifies how modern browsers interpret `hsl()` syntax, while Joblove & Greenberg's SIGGRAPH paper [@joblove1978] and Alvy Ray Smith's gamut analysis [@smith1978hsv] formalize the math. Together, they anchor the intuitive picture to rigorous colorimetry.

![HSL and HSV models](img/hsv_hsl_.svg)

<small>Diagram by the author, generated with matplotlib.</small>

### The hexcone model: Smith's geometric derivation

Smith calls the HSV model the **hexcone model** because of its geometric construction. The key insight comes from projecting the RGB cube along its main diagonal:

> "If the colorcube is projected along its main diagonal (the gray axis) onto a plane perpendicular to the diagonal, a hexagonal disk (a hexagon and its interior) results. The interior points are those colors one would see looking at the colorcube along its gray axis in the direction from white to black. For each value of gray, there is an associated subcube of the colorcube. Corresponding to each subcube—ie, to each gray value—is a projection as before. As the gray level changes from 0 (black) to 1 (white), one moves from one hexagonal disk to the next. Each disk is larger than the preceding one, with the disk for black being a point. This is the hexcone." [@smith1978hsv, p. 13]

The hexcone model "is an attempt to transform the RGB colorcube dimensions into a set of dimensions modeling the artist's method of mixing. These are called hue, saturation, and value (HSV). Varying H corresponds to traversing the color circle. Decreasing S (desaturation) corresponds to increasing whiteness, and decreasing V (devaluation) corresponds to increasing blackness" [@smith1978hsv, p. 13].

#### The color bar interpretation

Smith provides an elegant visual explanation: "A color is represented by three bars. It is obtained by mixing R, G, and B in the proportions implied by the lengths of the three bars. V is simply the height of the tallest bar. If X is the height of the smallest bar, then $(X, X, X)$ is the gray which is desaturating the color. Subtracting the 'DC-level' of gray from the color leaves the hue information as a proportional mix of two primaries" [@smith1978hsv, p. 13].

This leads to a key observation: "a color is a mixture of at most three primaries, a hue of at most two primaries, and a primary, of course, of one primary" [@smith1978hsv, p. 13].

#### The saturation formula derivation

Smith derives saturation geometrically from the hexagonal disk. For a point P in the disk, saturation S is "the ratio $WP / WP'$, where $WP'$ is the intersection of the extension of WP with the nearest side of the triangle." Working through the geometry:

$$S = \frac{V - \min(R, G, B)}{V}$$

"Notice that $S = 1$ implies at least one of R, G, or B is 0. For disk 1, this bounding hexagon may be identified with the color circle" [@smith1978hsv, p. 14].

#### Handling the achromatic singularity

The gray axis presents a special case: "Special care must be exercised at the singular points $S = 0$—ie, where $R = G = B$, the gray, or achromatic, axis of the hexcone. Hue is not defined along this axis. Often the hue is simply immaterial at such a gray point. A practice which frequently succeeds is to define H at a singularity to be what it was as a result of the last call to the transform. Smooth traversals of the gamut tend to leave H at a reasonable definition using this technique" [@smith1978hsv, p. 13].

### Value versus brightness: a critical distinction

Smith emphasizes that value (V) and brightness/lightness (L) measure fundamentally different things:

> "The distinction between value and brightness is important. It is illustrated by this example: Red, white, and yellow all have the same value (no blackness), but red has one third the brightness of white (using definition $L_u$), and one half the brightness of yellow. The principal distinction between the two is the manner in which the pure (fully saturated) hues are treated. There is a plane containing all the pure hues in HSV space, but not in HSL space. Hence V would be used where the pure hues are to be given equal weight—eg, in a painting program. L would be used where colors must be distinguished by their brightness—eg, in choosing colors for an animated cartoon such that the colors are distinguishable even on a black-and-white television receiver." [@smith1978hsv, p. 12]

This distinction directly impacts palette selection for ARMLite: if your sprite needs to read well in grayscale (accessibility, monochrome displays), HSL matching preserves luminance contrast. If you want vibrant, saturated colors to have equal visual weight, HSV is the better choice.

---

## Formulae and functions

Normalize each pixel’s channels to `[0,1]` so `r = R/255`, `g = G/255`, `b = B/255`. Following the construction laid out by Joblove & Greenberg and later summarized by Smith [@joblove1978; @smith1978hsv], define

$$
\begin{aligned}
C_{\max} &= \max(r,g,b), \\
C_{\min} &= \min(r,g,b), \\
\Delta &= C_{\max} - C_{\min}.
\end{aligned}
$$

### HSV cylinder

$$
\begin{aligned}
V &= C_{\max},\\
S &= \begin{cases}
0 & \text{if } C_{\max} = 0,\\
\dfrac{\Delta}{C_{\max}} & \text{otherwise},
\end{cases}\\
H &= \begin{cases}
0 & \text{if } \Delta = 0,\\
60^\circ \bmod 360^\circ \times \begin{cases}
\dfrac{g-b}{\Delta} & C_{\max} = r,\\
2 + \dfrac{b-r}{\Delta} & C_{\max} = g,\\
4 + \dfrac{r-g}{\Delta} & C_{\max} = b.
\end{cases}
\end{cases}
\end{aligned}
$$

### HSL double cone

$$
\begin{aligned}
L &= \frac{C_{\max} + C_{\min}}{2},\\
S &= \begin{cases}
0 & \text{if } \Delta = 0,\\
\dfrac{\Delta}{1 - |2L - 1|} & \text{otherwise},
\end{cases}
\end{aligned}
$$

with the hue branch identical to HSV. All trigonometric work inside `colorsys` happens on normalized angles (`H \in [0,1)`), so the script later wraps hue distances in that same domain.

### The triangle model: generalized brightness

Smith's **triangle model** generalizes the double-cone to arbitrary brightness definitions. The normalized color coordinates are:

$$r = \frac{w_R R}{L}, \quad g = \frac{w_G G}{L}, \quad b = \frac{w_B B}{L}$$

where the generalized brightness is:

$$L = w_R R + w_G G + w_B B$$

with weights satisfying $w_R + w_G + w_B = 1$ and $w_R, w_G, w_B \ge 0$. "All such normalized colors fall in the plane $r + g + b = 1$ and are bounded by the equilateral triangle" [@smith1978hsv, p. 15].

Two cases are particularly important:

1. **Unbiased case**: $w_R = w_G = w_B = \tfrac{1}{3}$, giving $L_u = (R + G + B) / 3$
2. **NTSC case**: $w_R = 0.30$, $w_G = 0.59$, $w_B = 0.11$, giving $L_n = Y$ (luminance)

"The gray points, $R = G = B$, all map into $W = (w_R, w_G, w_B)$." In the unbiased case, the gray point is at the centroid $(\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{1}{3})$. In the NTSC case, "the gray point $(0.30, 0.59, 0.11)$ is 'biased' away from the centroid of the equilateral triangle" [@smith1978hsv, p. 15].

Smith connects this to classical colorimetry: "The triangle so obtained is an example of what is known in color theory as a chromaticity diagram. The most famous such diagram is that from 1931 of the CIE (Commission Internationale de l'Eclairage). It includes the entire human color gamut" [@smith1978hsv, p. 16]. The RGB monitor gamut appears as a triangular subset within the CIE diagram—demonstrating that displays can only reproduce a fraction of human-visible colors.

### Reconstructing RGB from hue

To reverse the process—going from a hue angle back to its pure RGB color on the unit hexagon—Joblove & Greenberg provide a set of piecewise linear functions. Given normalized hue $h \in [0,1)$, the RGB components of the fully saturated color are [@joblove1978, p. 22]:

$$
r = \begin{cases}
1 & \text{if } h \le \tfrac{1}{6} \text{ or } h > \tfrac{5}{6} \\
2 - 6h & \text{if } \tfrac{1}{6} \le h \le \tfrac{2}{6} \\
0 & \text{if } \tfrac{2}{6} \le h \le \tfrac{4}{6} \\
6h - 4 & \text{if } \tfrac{4}{6} \le h \le \tfrac{5}{6}
\end{cases}
$$

$$
g = \begin{cases}
6h & \text{if } 0 \le h \le \tfrac{1}{6} \\
1 & \text{if } \tfrac{1}{6} \le h \le \tfrac{3}{6} \\
4 - 6h & \text{if } \tfrac{3}{6} \le h \le \tfrac{4}{6} \\
0 & \text{if } \tfrac{4}{6} \le h < 1
\end{cases}
$$

$$
b = \begin{cases}
0 & \text{if } 0 \le h \le \tfrac{2}{6} \\
6h - 2 & \text{if } \tfrac{2}{6} \le h \le \tfrac{3}{6} \\
1 & \text{if } \tfrac{3}{6} \le h \le \tfrac{5}{6} \\
6 - 6h & \text{if } \tfrac{5}{6} \le h < 1
\end{cases}
$$

Each channel ramps linearly across $\tfrac{1}{6}$ of the hue circle (60°), creating the familiar six-sector structure: red → yellow → green → cyan → blue → magenta → red. The factor of 6 appears because the hue wheel is partitioned into six equal segments, each spanning $\tfrac{1}{6}$ of the normalized range.

However, these piecewise linear ramps have a perceptual cost: "Because the function is non-continuous, hue series will exhibit Mach banding (the illusion of overly light or dark areas) at the discontinuities due to the tendency of the human visual system to enhance such variations in luminance for any of the three receptor systems" [@joblove1978, p. 22].

To mitigate this, Joblove & Greenberg propose a cosine-smoothed variant that eliminates the sharp corners:

$$
[r \; g \; b] = \frac{[1 \; 1 \; 1] + \cos\bigl([1 \; 1 \; 1] - [r' \; g' \; b']\bigr)\pi}{2}
$$

where $r'$, $g'$, and $b'$ are the piecewise linear values from the functions above. The cosine term smooths the linear ramps into sinusoidal curves, producing gradual transitions at the sector boundaries and reducing the perceived Mach bands.

### Intensity and chroma scaling

To produce colors at arbitrary intensity and saturation levels—not just the fully saturated hues on the hexagon boundary—Joblove & Greenberg introduce a two-branch formula that scales by relative chroma $c$ and intensity $i$ (each on the range 0 to 1) [@joblove1978, p. 22]:

$$
[r \; g \; b] = \begin{cases}
\bigl([\tfrac{1}{2} \; \tfrac{1}{2} \; \tfrac{1}{2}] + c([r' \; g' \; b'] - [\tfrac{1}{2} \; \tfrac{1}{2} \; \tfrac{1}{2}])\bigr) \cdot 2i & \text{if } i \le \tfrac{1}{2} \\
\bigl([\tfrac{1}{2} \; \tfrac{1}{2} \; \tfrac{1}{2}] + c([r' \; g' \; b'] - [\tfrac{1}{2} \; \tfrac{1}{2} \; \tfrac{1}{2}])\bigr) + \bigl([\tfrac{1}{2} \; \tfrac{1}{2} \; \tfrac{1}{2}] - c([r' \; g' \; b'] - [\tfrac{1}{2} \; \tfrac{1}{2} \; \tfrac{1}{2}])\bigr) \cdot (2 - 2i) & \text{if } i \ge \tfrac{1}{2}
\end{cases}
$$

where "$c$ is the relative chroma and $i$ is the intensity (each on the range 0 to 1), and $[r' \; g' \; b']$ is computed using an equation such as" the piecewise or cosine-smoothed forms above [@joblove1978, p. 22].

This construction traces its lineage to **The Munsell Color System (1905)**, which introduced the perceptual coordinates still used today: **Munsell hue**, **Munsell value** (lightness relative to a reference white), and **Munsell chroma** (colorfulness relative to gray). As Joblove & Greenberg describe: "The grays are linearly interpolated on a straight line (the cylindrical axis) from black to white; the colors which are of maximum chroma for their respective hues (for the given rgb display space) are located on a circle centered on the gray axis and perpendicularly intersecting it halfway between the black point and white point" [@joblove1978, p. 22]. This geometric arrangement—gray axis at the center, saturated hues at maximum radius, intensity controlling vertical position—is exactly the double-cone structure that HSL inherits.

### From double-cone to cylinder: the HSV construction

The double-cone arises naturally from chroma and intensity, but a different parameterization yields the HSV cylinder. Joblove & Greenberg explain [@joblove1978, p. 22]:

> In a color space defined by hue, chroma, and intensity, all the colors of a given saturation define a conical surface whose apex is the black point (at which the saturation is undefined). A color space may be defined in which the radial component is directly related to saturation by letting that component be
>
> $$s = \frac{\max(r,g,b) - \min(r,g,b)}{\max(r,g,b)}$$
>
> Furthermore, the axial component can be specified to correspond to that component of color which is equal (and maximal) for all the colors representing the maximum intensities for all chromaticities and zero for black. In other words, the circle of maximum-chroma colors can be located so its center intersects the cylindrical axis at the white point.

This axial component is what they call **value**. In this color space, "the color solid is a right circular cylinder whose 'top' base is a circularized chromaticity diagram, whose 'bottom' base is black, and whose cylindrical surface" contains the desaturated variants of each hue [@joblove1978, p. 22]. The key insight is that by defining saturation relative to `max(r,g,b)` rather than the full black-to-white span, the apex of the cone lifts to form a flat top—hence the cylinder rather than the double-cone of HSL.

The cylindrical surface "contains the maximum-saturation colors for all hues and lightnesses" [@joblove1978, p. 23]. For a color in this space, the RGB reconstruction is:

$$
[r \; g \; b] = \bigl([1 \; 1 \; 1] + s \cdot ([r' \; g' \; b'] - [1 \; 1 \; 1])\bigr) \cdot v
$$

"where $s$ is the saturation and $v$ is the 'value' (each on the range 0 to 1), and $[r' \; g' \; b']$ is computed using" the piecewise or cosine-smoothed hue equations [@joblove1978, p. 23].

#### Triangular variants and CIE coordinates

The circular cross-section is a convenience, not a requirement. Joblove & Greenberg note: "Alternative variants of this arrangement are possible. If the bases of the cylinder, instead of being circles, are triangles with the primaries at the vertices, any section corresponds to the color triangle in the chromaticity diagram" [@joblove1978, p. 23].

This observation opens a path toward standards-based color: "This suggests the possibility of defining a color space based on the CIE (Commission Internationale de l'Éclairage or International Commission on Illumination) x,y chromaticity coordinates, the standard system established by that organization in 1931. Existing transformations could then be used to work in other CIE standard coordinate systems" [@joblove1978, p. 23]—a prescient comment given that CIE L\*a\*b\* and CIEDE2000 are now the workhorses of perceptual color difference calculations.

### Weighted palette metric

The closure returned by `_weighted_distance` implements

$$
\begin{aligned}
d_H &= \min\left(|h_a - h_b|, 1 - |h_a - h_b|\right),\\
d_S &= |s_a - s_b|,\\
d_V &= |v_a - v_b|,\\
d &= \sqrt{(w_h d_H)^2 + (w_s d_S)^2 + (w_v d_V)^2},
\end{aligned}
$$

which mirrors the perceptual heuristics Poynton recommends for highlight-preserving workflows [@poynton2003]. Hue wrap keeps magenta seams from appearing between 359° and 0°, while the user-facing weights translate artistic intent directly into the scoring function.

Once the palette vectors are cached in HSV/HSL space (`_palette_space`), the entire pixel search becomes a geometric problem on cylinders rather than a brute RGB cube walk, and that simple reframing is what lets this script feel like a tool with opinions rather than a generic converter.

### Computational considerations: hexcone vs. triangle

Smith explicitly addresses performance in the 1978 paper, and his observations remain relevant:

> "The transform pair derived from the hexcone model (RGB to HSV) require no trigonometric or other expensive functions. Hence they are quite fast, a fact of considerable importance when they are to be performed at the pixel level in a frame buffer." [@smith1978hsv, p. 19]

In contrast, "the triangle model transforms (RGB to HSL) are too slow to be used in software form in interactive situations such as painting because of the function calls to sqrt(), arctan(), and cos()" [@smith1978hsv, p. 19]. Smith suggests that "approximations to these functions (eg, linear interpolation between values in a lookup table for cos()) would lead to speedier response, especially if implemented in microcode" [@smith1978hsv, p. 19].

For ARMLite sprite conversion, both models are fast enough—we process at most 12,288 pixels (128×96), and Python's `colorsys` module uses the optimized hexcone algorithms. The choice between HSV and HSL should be driven by the artistic goal, not performance.

### Practical application: tint painting

Smith describes a compelling use case that illuminates why these transforms matter:

> "In the RGB paint program at NYIT there is a type of painting called tint paint. Here the user selects a color to paint with. Its tint (H and S) is extracted by use of the RGB to HSV transform. Now painting in a frame buffer can be thought of as overwriting a small 2-dimensional subset of a large 2-dimensional array... Tint painting is the following variation on simple painting: At a point (pixel) about to be written in the frame buffer, an RGB to HSV transform is performed to extract the value V there. A new color is formed from the tint the user selected and V of the pixel. An application of the HSV to RGB transform converts the color to usable form, and then it is written into the pixel." [@smith1978hsv, p. 19]

This technique—preserving value while replacing hue and saturation—is exactly what colorization workflows do today. For ARMLite, a similar approach could let artists "paint" a grayscale sprite with palette-constrained hues while respecting the original shading.

---

## Why run HSV/HSL on ARMLite sprites?
- **Hue preservation:** HSV/HSL treat hue as an angle on a cylinder (0–360°). Matching colors in that space keeps reds grouped with reds even when luminance shifts, which avoids palette "hue flipping" that happens with plain RGB distances.
- **Intentional saturation control:** Designers can bias the `--weights` parameter toward saturation to keep neon palettes intact, or toward value/lightness to produce pastel shading for UI sprites.
- **Predictable gradients:** Value (HSV) maps directly to perceived brightness for emissive displays, while Lightness (HSL) keeps mid-tones centered. Choosing between them lets you dial in dithers that work with, not against, later diffusion stages.
- **Drop-in assembly:** The script already emits valid ARMLite assembly, so it can be first in a pipeline or stand alone for quick palette experiments.

---

## Sources 
- **HSV/HSL origins:** Joblove & Greenberg’s *Color Spaces for Computer Graphics* (SIGGRAPH 1978) details the intuitive hue-saturation-lightness cylinders used here [@joblove1978]. Their work inspired modern paint-pickers and underpins the formulas exposed by Python's `colorsys`.
- **Gamut transform math:** Alvy Ray Smith's *Color Gamut Transform Pairs* (SIGGRAPH 1978) derives both the hexcone (HSV) and triangle (HSL) models with complete RGB↔HSV and RGB↔HSL algorithms [@smith1978hsv]. Smith's hexcone transforms—used successfully at Xerox PARC and NYIT for frame buffer painting programs since 1974—avoid trigonometric functions entirely, making them fast enough for real-time pixel manipulation. The triangle model generalizes brightness to include NTSC luminance ($Y = 0.30R + 0.59G + 0.11B$) as a special case.
- **Digital video practice:** Charles Poynton's *Digital Video and HDTV* (Morgan Kaufmann, 2003) recommends HSV-style processing for highlight preservation before quantization [@poynton2003]. The same advice applies to sprite art where saturated highlights matter.
- **Perceptual caveats:** Bruce Lindbloom's colorimetry reference documents why HSV/HSL are not perceptually uniform [@lindbloom]; weighting hue and saturation compensates for that non-linearity, which is why the script exposes explicit weights.

---

## Implementation walkthrough
| Function | Purpose |
| --- | --- |
| `_rgb_to_hsv` / `_rgb_to_hsl` | Wrap `colorsys` helpers, normalizing RGB to `[0,1]` floats before returning `(h, s, v/l)` tuples. |
| `_weighted_distance` | Builds a closure over hue/saturation/value weights. Hue distance wraps around the circle (`min(|Δh|, 1-|Δh|)`) so reds near 0° and 360° still match. |
| `_palette_space` | Precomputes HSV/HSL representations for every ARMLite palette entry to avoid recomputing transforms per pixel. |
| `apply_rgb_to_hsv_hsl` | Resizes the image to `128×96`, walks every pixel, converts it, and picks the closest palette entry via the weighted metric. Returns the grid of ARMLite color names. |
| `generate_assembly` | Writes the `.Resolution`, `.PixelScreen`, and per-pixel `STR` instructions expected by Peter Higginson's simulator. |
| `process_image` | Orchestrates the PIL load, resize, conversion, and final write. The CLI funnels into this function.

Key design notes:
- The search is brute-force but the palette is only 256 entries, so Python-level loops remain fast (~0.15 s on an M1 for 128×96 inputs).
- Hue wrap ensures `h=0.99` and `h=0.01` behave as neighbors, critical for sprites heavy in magentas/reds.
- Path to any raster image PIL can open. It will be resized to 128×96 with antialiasing. Resizing happens before conversion to guarantee a consistent ARMLite screen fit. If you need custom resolutions, change the `(128, 96)` tuple and adjust the `generate_assembly` stride constant.

---

## CLI reference
| Option | Description |
| --- | --- |
| `-o / --output` | Assembly output path. Defaults to `converted.s`. |
| `--space {hsv,hsl}` | Selects HSV (default) or HSL matching. HSV is better for emissive/glassy looks; HSL balances lightness for flat UI assets. |
| `--weights H,S,V` | Comma-separated floats controlling the weighting metric. Use `1,1,1` to treat all channels evenly, or bias toward hue (`3,1,0.5`) for cel shading, etc. |

Example: preserve saturated blues with strong hue weight and route into CIEDE2000 downstream.

```bash
$ python algorithms/color_transforms/src/rgb_to_hsv_hsl.py poster.png \
  --space hsv --weights 2.5,1.2,0.5 -o poster_hsv.s
```
![Aggressive weighting produces harsh noise](img/)

---

## Weighting playbook
| Goal | Weights | Notes |
| --- | --- | --- |
| Preserve cel-shaded hue bands | `3,0.8,0.4` | Keeps hue dominant so large flat regions stay in-family even under heavy dithers. |
| Boost neon UI saturation | `1.2,2.5,0.6` | Saturation bias stops conversion toward gray when brightness is normalized. |
| Highlight silhouettes | `0.8,0.9,2.0` | Value/lightness bias ensures rim lighting reads correctly before quantization. |
| Pastel wash | `0.7,0.5,1.8` with `--space hsl` | Lightness emphasis keeps subtle gradients from collapsing to black. |

Tune weights with fractional steps (e.g., increments of `0.1`). The metric squares and sums them, so doubling a channel roughly doubles its influence.

---

## Pipeline ideas
1. **Hue-protected anime frame**
   - `rgb_to_hsv_hsl.py frame.png --space hsv --weights 3,1,0.6`
   - `distance_delta_e.py --metric ciede2000`
   - `floyd-steinberg.py` for fast halftones.
2. **Pastel UI card**
   - `rgb_to_hsv_hsl.py card.png --space hsl --weights 0.6,0.5,1.9`
   - `median_cut.py -c 32`
   - `atkinson.py` for soft diffusion.
3. **Specular metal study**
   - `rgb_to_hsv_hsl.py mech.png --space hsv --weights 1.5,1.5,1.5`
   - `wu_quantizer.py`
   - `sierra.py --variant sierra-lite` to reduce worm artifacts.

Document pipelines in `docs/pipelines.md` as you validate them so others can reproduce the recipe.

---

## Attribution & licensing
- [1] World Wide Web Consortium, *CSS Color Module Level 4*, Editor’s Draft, <https://www.w3.org/TR/css-color-4/> — documents browser-facing `hsl()`/`hwb()` parameterization and is published under the W3C Document License.
---

## Testing & validation checklist
- Run with both `--space hsv` and `--space hsl` on the same asset to ensure parameter parsing works.
- Feed intentionally bad `--weights` (non-numeric, wrong length) to confirm the CLI rejects them before processing.
- Verify generated assembly loads in the ARMLite simulator (`Load → Assemble → Run`) without warnings.
- Spot-check palette matching by logging `best_name` (add a temporary print) for tricky gradients.

---

## Troubleshooting
- **Banding remains after conversion:** Lower the value/lightness weight so hue drives the match, then add diffusion in `dithers/` scripts.
- **Output looks washed out:** Switch to `--space hsv` and increase the saturation weight; HSL tends to desaturate mid-tones.
- **File not found errors:** The script expects paths relative to the repo root; use absolute paths or quote filenames with spaces.
- **Performance concerns:** Crop large source art before conversion; resizing down to 128×96 avoids wasted work and clarifies the final sprite early.

---

## Further reading
- Joblove, G. H., & Greenberg, D. *Color Spaces for Computer Graphics*, SIGGRAPH 1978.
- Smith, A. R. *Color Gamut Transform Pairs*, SIGGRAPH 1978.
- Poynton, C. *Digital Video and HDTV: Algorithms and Interfaces*, Morgan Kaufmann, 2012.
- Bruce Lindbloom, *Color Science and Technology*, https://www.brucelindbloom.com.
- World Wide Web Consortium, *CSS Color Module Level 4*, https://www.w3.org/TR/css-color-4/.

- https://www.w3.org/TR/css-color-4/#rgb-to-hsl
- https://archive.org/details/digital-video-and-hd-algorithms-and-interfaces-2nd-ed.-poynton-2012-02-07/page/215/mode/2up
- https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
- https://www.geeksforgeeks.org/python/three-dimensional-plotting-in-python-using-matplotlib/

Leverage these sources when extending the script—e.g., to add HCL or OKHSL variants—so the documentation keeps pace with the research pedigree.
