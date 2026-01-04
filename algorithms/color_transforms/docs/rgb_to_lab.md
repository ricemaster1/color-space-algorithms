# RGB → L\*a\*b\* Color Transform

Converts images to ARMlite assembly using the **CIE L\*a\*b\*** color space for perceptually uniform palette matching.

---

## CLI Reference

```bash
python rgb_to_lab.py <image> [output] [options]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `image` | Path to input image (PNG, JPG, etc.) |
| `output` | Output assembly file path (optional) |

### Options

| Flag | Description |
|------|-------------|
| `-o`, `--output-file` | Output path (alternative to positional) |
| `-w`, `--weights L,a,b` | Comma-separated weights for L\*, a\*, b\* channels |
| `-a`, `--auto` | Auto-optimize weights to minimize RGB error |
| `-h`, `--help` | Show help message |

### Examples

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

---

## Weighting Playbook

The L\*a\*b\* color space separates lightness from chromaticity:

| Channel | Range | Meaning | Weight Effect |
|---------|-------|---------|---------------|
| **L\*** | 0–100 | Lightness (black → white) | Higher = preserve brightness/contrast |
| **a\*** | −128 to +127 | Green ↔ Red axis | Higher = preserve green/red tones |
| **b\*** | −128 to +127 | Blue ↔ Yellow axis | Higher = preserve blue/yellow tones |

### Starting Points

| Image Type | Suggested Weights | Rationale |
|------------|-------------------|-----------|
| Portraits | `1.5, 1, 1` | Preserve skin tone luminance |
| Landscapes | `1, 1, 1.2` | Emphasize sky/foliage color |
| High contrast | `2, 1, 1` | Maintain shadow/highlight detail |
| Saturated art | `1, 1.5, 1.5` | Preserve vibrant colors |
| Grayscale-ish | `2, 0.5, 0.5` | Prioritize luminance over hue |

### Auto-Match (`--auto`)

The `--auto` flag runs a grid search to find weights that minimize the average RGB error between original and quantized pixels. This is image-adaptive and often produces the best results.

```bash
python rgb_to_lab.py photo.png --auto
# Output: Auto-matched L*a*b* weights: (1.50, 1.25, 0.75) - Avg RGB error: 28.3
```

---

## Color Theory Background

### Why L\*a\*b\*?

The **CIE L\*a\*b\*** (CIELAB) color space was designed to be **perceptually uniform** — meaning that a given numerical change in color values produces a roughly equal perceived change in color, regardless of where in the color space you are.

This is unlike RGB or HSV, where:
- In RGB, equal numeric steps don't produce equal perceived differences
- In HSV, hue differences near gray are exaggerated

### The Transform Pipeline

```
sRGB → Linear RGB → CIE XYZ → CIE L*a*b*
```

1. **sRGB → Linear RGB**: Remove gamma correction
   ```
   if C ≤ 0.04045:  C_lin = C / 12.92
   else:            C_lin = ((C + 0.055) / 1.055)^2.4
   ```

2. **Linear RGB → XYZ**: Apply 3×3 matrix (D65 illuminant)
   ```
   [X]   [0.4124564  0.3575761  0.1804375] [R_lin]
   [Y] = [0.2126729  0.7151522  0.0721750] [G_lin]
   [Z]   [0.0193339  0.1191920  0.9503041] [B_lin]
   ```

3. **XYZ → L\*a\*b\***: Nonlinear transform with reference white
   ```
   f(t) = t^(1/3)           if t > ε
        = (κt + 16) / 116   otherwise
   
   where ε = 0.008856, κ = 903.3
   
   L* = 116 × f(Y/Yn) − 16
   a* = 500 × (f(X/Xn) − f(Y/Yn))
   b* = 200 × (f(Y/Yn) − f(Z/Zn))
   ```

### D65 Reference White

The script uses the **D65 standard illuminant** (average daylight):
- Xn = 0.95047
- Yn = 1.00000
- Zn = 1.08883

---

## L\*a\*b\* vs HSV/HSL

| Aspect | L\*a\*b\* | HSV/HSL |
|--------|----------|---------|
| **Perceptual uniformity** | ✓ Designed for it | ✗ Not uniform |
| **Hue handling** | Implicit in a\*/b\* | Explicit H channel |
| **Lightness accuracy** | L\* correlates well with human perception | V/L less accurate |
| **Computation cost** | Higher (matrix + nonlinear) | Lower |
| **Best for** | Photo-realistic images | Graphic art, icons |

### When to Use Which

- **L\*a\*b\***: Photographs, images with subtle color gradients, skin tones
- **HSV**: Graphic art, images with distinct hue regions, cartoons

---

## Implementation Details

### Distance Metric

The script uses **weighted Euclidean distance** in L\*a\*b\* space:

```python
distance = sqrt((ΔL* × wL)² + (Δa* × wa)² + (Δb* × wb)²)
```

This is a simplified version of the CIE76 color difference formula. More sophisticated formulas exist (CIE94, CIEDE2000) but add complexity for marginal gains in a 147-color palette context.

### Quantization Process

For each pixel:
1. Convert pixel RGB → L\*a\*b\*
2. Compute weighted distance to all 147 palette colors (pre-converted to L\*a\*b\*)
3. Select the palette color with minimum distance
4. Output the color name to the assembly grid

### Performance Notes

- Palette L\*a\*b\* values are computed once and cached
- Auto-match uses numpy vectorization for ~10× speedup
- Image is resized to 128×96 before processing

---

## Sources

- [CIE 1976 L\*a\*b\* Color Space](https://en.wikipedia.org/wiki/CIELAB_color_space)
- [sRGB to XYZ Conversion](http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html)
- [Color Difference Formulas](https://en.wikipedia.org/wiki/Color_difference)
