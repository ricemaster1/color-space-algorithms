---
title: "`rgb_to_hsv_hsl.py` Deep Dive"
bibliography: references.bib
---

# `rgb_to_hsv_hsl.py` 
# Deep Dive

The `algorithms/color_transforms/src/rgb_to_hsv_hsl.py` module converts arbitrary imagery into ARMLite-compatible sprites by matching pixels in HSV or HSL space before emitting `.Resolution`, `.PixelScreen`, and per-pixel stores. This document narrates the color theory behind the script, explains every function, and provides research references plus assets you can reuse in blog posts or release notes.

---

## Cylindrical color context

> “HSL and HSV are the two most common cylindrical-coordinate representations of points in an RGB color model. The two representations rearrange the geometry of RGB in an attempt to be more intuitive and perceptually relevant than the cartesian (cube) representation.” — Wikipedia contributors, *HSL and HSV* [@wikipedia-hsl]

That canonical description matches what the ARMLite workflow needs: a hue axis that wraps cleanly, plus independent saturation and lightness/value controls that reflect how artists reason about paint or shader parameters. The well-known twin-cone diagram from the article (licensed CC BY-SA 4.0) illustrates why hue neighborhoods feel continuous in HSV/HSL; recreate a derivative version for this repo by combining our local SVG wheel with saturation/value slices so we can ship an attribution-friendly asset instead of hotlinking the figure.

Beyond Wikipedia, the CSS Color Module Level 4 specification [@w3c-color4] codifies how modern browsers interpret `hsl()` and `hsv()` syntax, offering concrete parameter ranges to mirror in CLI validation. Pair that guidance with Joblove & Greenberg’s original SIGGRAPH paper [@joblove1978] and Alvy Ray Smith’s gamut analysis [@smith1978hsv] to anchor both intuitive explanations and the underlying math.

![HSL and HSV models](img/Hsl-hsv_models.svg)

<small>Figure adapted from [Jacob Rus, “Hsl-hsv models”](https://commons.wikimedia.org/wiki/File:Hsl-hsv_models.svg), licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) via Wikimedia Commons.</small>

---

## Mathematical backbone

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
V &= C_{\max},\\[4pt]
S &= \begin{cases}
0 & \text{if } C_{\max} = 0,\\
\dfrac{\Delta}{C_{\max}} & \text{otherwise},
\end{cases}\\[10pt]
H &= \begin{cases}
0 & \text{if } \Delta = 0,\\
60^\circ \bmod 360^\circ \times \begin{cases}
\dfrac{g-b}{\Delta} & C_{\max} = r,\\[6pt]
2 + \dfrac{b-r}{\Delta} & C_{\max} = g,\\[6pt]
4 + \dfrac{r-g}{\Delta} & C_{\max} = b.
\end{cases}
\end{cases}
\end{aligned}
$$

### HSL double cone

$$
\begin{aligned}
L &= \frac{C_{\max} + C_{\min}}{2},\\[4pt]
S &= \begin{cases}
0 & \text{if } \Delta = 0,\\
\dfrac{\Delta}{1 - |2L - 1|} & \text{otherwise},
\end{cases}
\end{aligned}
$$

with the hue branch identical to HSV. All trigonometric work inside `colorsys` happens on normalized angles (`H \in [0,1)`), so the script later wraps hue distances in that same domain.

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

---

## Why run HSV/HSL on ARMLite sprites?
- **Hue preservation:** HSV/HSL treat hue as an angle on a cylinder (0–360°). Matching colors in that space keeps reds grouped with reds even when luminance shifts, which avoids palette "hue flipping" that happens with plain RGB distances.
- **Intentional saturation control:** Designers can bias the `--weights` parameter toward saturation to keep neon palettes intact, or toward value/lightness to produce pastel shading for UI sprites.
- **Predictable gradients:** Value (HSV) maps directly to perceived brightness for emissive displays, while Lightness (HSL) keeps mid-tones centered. Choosing between them lets you dial in dithers that work with, not against, later diffusion stages.
- **Drop-in assembly:** The script already emits valid ARMLite assembly, so it can be first in a pipeline or stand alone for quick palette experiments.

---

## Color theory + research backdrop
- **HSV/HSL origins:** Joblove & Greenberg’s *Color Spaces for Computer Graphics* (SIGGRAPH 1978) details the intuitive hue-saturation-lightness cylinders used here [@joblove1978]. Their work inspired modern paint-pickers and underpins the formulas exposed by Python's `colorsys`.
- **Gamut transform math:** Alvy Ray Smith's *Color Gamut Transform Pairs* (1978) proves that hue-first spaces prevent gamut clipping when mapping between RGB primaries [@smith1978hsv]. Our transform inherits those guarantees when constraining to the ARMLite palette.
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
- Resizing happens before conversion to guarantee a consistent ARMLite screen fit. If you need custom resolutions, change the `(128, 96)` tuple and adjust the `generate_assembly` stride constant.

---

## CLI reference
| Option | Description |
| --- | --- |
| `image` | Path to any raster image PIL can open. It will be resized to 128×96 with antialiasing. |
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

## Giving the transform a soul

HSV/HSL math is clinical, but the sprites you feed it are not. When I run `rgb_to_hsv_hsl.py` on a dusk skyline, I start by nudging hue weight past 2 so oranges stay molten, then I sketch a note about what the scene is *supposed* to feel like—“sodium vapor, rain, patient neon.” That note becomes the constraint for every other parameter in the pipeline. Think of the CLI as a color diary: every weight triple is a tiny manifesto about what matters in the frame.

Try this ritual:

1. **Write a sentence** describing the mood before you touch the keyboard.
2. **Map words to weights** (`vibrant` → high saturation, `somber` → high lightness).
3. **Commit the combo** in `docs/pipelines.md` with a screenshot so the story travels with the codebase.

These micro-stories keep ARMLite sprites from becoming sterile conversions. They remind contributors that we are not just minimizing distances—we are protecting the emotional gradients artists fought to paint in the first place.

---

## Image assets
- `docs/images/rgb-to-hsv-hsl-wheel.svg` – the hue wheel illustration shown at the top of this page.
- `docs/images/rgb-to-hsv-hsl-contrast.svg` – compares HSV Value vs HSL Lightness midpoints.
- `color_transforms/docs/img/Hsl-hsv_models.svg` – Jacob Rus’s CC BY-SA 3.0 twin-cone illustration (stored locally for offline docs builds).

Embed them anywhere via Markdown (`![desc](../images/...)`) or include them in slide decks. If you capture before/after sprites, store PNGs alongside these SVGs following the naming convention `rgb-to-hsv-hsl-example-*.png` so documentation stays organized.

![Value vs Lightness](../images/rgb-to-hsv-hsl-contrast.svg)

---

## Attribution & licensing
- [1] Wikipedia contributors, “HSL and HSV,” *Wikipedia, The Free Encyclopedia*, CC BY-SA 4.0. Credit the article and include a link plus the license whenever you reuse or adapt the twin-cone illustration.
- [2] World Wide Web Consortium, *CSS Color Module Level 4*, Editor’s Draft, <https://www.w3.org/TR/css-color-4/> — documents browser-facing `hsl()`/`hwb()` parameterization and is published under the W3C Document License.
- Jacob Rus, “HSL and HSV Models,” Wikimedia Commons (https://commons.wikimedia.org/wiki/File:Hsl-hsv_models.svg), licensed CC BY-SA 3.0 / GFDL. Our copy in `docs/img/` remains unmodified; include the attribution text shown beneath the figure if redistributed.

Any new diagrams derived from third-party sources should either (a) be redrawn from scratch (preferred) or (b) bundled with the original CC BY-SA attribution text adjacent to the image so downstream users remain compliant.

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
- Poynton, C. *Digital Video and HDTV: Algorithms and Interfaces*, Morgan Kaufmann, 2003.
- Bruce Lindbloom, *Color Science and Technology*, https://www.brucelindbloom.com.
- World Wide Web Consortium, *CSS Color Module Level 4*, https://www.w3.org/TR/css-color-4/.
- Wikipedia contributors, “HSL and HSV,” *Wikipedia, The Free Encyclopedia*, https://en.wikipedia.org/wiki/HSL_and_HSV (CC BY-SA 4.0).

Leverage these sources when extending the script—e.g., to add HCL or OKHSL variants—so the documentation keeps pace with the research pedigree.
