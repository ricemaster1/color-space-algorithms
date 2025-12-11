---
title: CIEDE2000 Color-Difference Metric
bibliography: references.bib
---

# CIEDE2000 Color-Difference Metric

The CIEDE2000 formulation refines earlier CIE76/94 metrics by weighting lightness, chroma, and hue differences with empirically tuned scaling functions so that the reported error aligns with human just noticeable differences (JND) across the full color gamut [@luo2001]. In practice it reduces over-penalization of blue hues, balances chroma-heavy samples, and introduces a rotation term that compensates for the purple region instability highlighted by [@sharma2005].

## Mathematical Model

Given two Lab colors $(L_1, a_1, b_1)$ and $(L_2, a_2, b_2)$, their CIEDE2000 distance is

$$
\Delta E_{00} = \sqrt{\left(\frac{\Delta L'}{S_L}\right)^2 + \left(\frac{\Delta C'}{S_C}\right)^2 + \left(\frac{\Delta H'}{S_H}\right)^2 + R_T \frac{\Delta C'}{S_C} \frac{\Delta H'}{S_H}}
$$

with the intermediate terms

$$
\begin{aligned}
\bar{L}' &= \frac{L_1 + L_2}{2}, \\
C_i &= \sqrt{a_i^2 + b_i^2}, \quad \bar{C}' = \frac{C'_1 + C'_2}{2}, \\
G &= \frac{1}{2} \left[1 - \sqrt{\frac{\bar{C}'^7}{\bar{C}'^7 + 25^7}}\right], \quad a'_i = (1+G) a_i, \\
h'_i &= \operatorname{atan2}(b_i, a'_i), \\
\Delta L' &= L_2 - L_1, \quad \Delta C' = C'_2 - C'_1, \\
\Delta h' &= 2\sqrt{C'_1 C'_2} \sin\left(\frac{h'_2 - h'_1}{2}\right), \\
S_L &= 1 + \frac{0.015(\bar{L}'-50)^2}{\sqrt{20 + (\bar{L}'-50)^2}}, \\
S_C &= 1 + 0.045\bar{C}', \quad S_H = 1 + 0.015\bar{C}' T, \\
T &= 1 - 0.17\cos(h'_m - 30^\circ) + 0.24\cos(2h'_m) + 0.32\cos(3h'_m + 6^\circ) - 0.20\cos(4h'_m - 63^\circ), \\
R_T &= -2 \sqrt{\frac{\bar{C}'^{7}}{\bar{C}'^{7} + 25^7}} \sin(2\Delta \theta).
\end{aligned}
$$

Angles use degrees as in the CIE reference text. The rotation term $R_T$ is evaluated with $\Delta \theta = 30^\circ \exp\left[-\left(\frac{h'_m-275^\circ}{25^\circ}\right)^2\right]$ and $h'_m$ the mean hue. All trigonometric functions are evaluated in radians within the implementation even though the standard is defined in degrees.

## Implementation Notes

- Source: `algorithms/distance_metrics/src/distance_ciede2000.py` converts RGB pixels to Lab (via `rgb_to_lab.py`) before applying the metric.
- Numerical stability: the script clamps $C'$ to $10^{-7}$ to avoid division by zero inside $R_T$ and $\Delta h'$.
- Vectorization: NumPy broadcasts the intermediate arrays so the metric remains fast enough for per-pixel palette searches.
- The module exposes a CLI that prints mean/median/percentile errors when supplied with two equal-sized images:

```bash
python algorithms/distance_metrics/src/distance_ciede2000.py assets/reference.png assets/candidate.png \
	--report reports/reference_vs_candidate.json
```

## Pipeline Usage

- **Palette search** – import the `measure` function and pass it into quantizers such as `k_means.py` to bias clustering toward perceptual accuracy.
- **Quality gating** – run the CLI after quantization+dither to confirm that $\Delta E_{00}$ stays below 2 for UI sprites and below 5 for photographic scenes.
- **ARMLite integration** – when calling `algorithms/armlite.py convert`, pass `--algo-extra "--distance distance_metrics/src/distance_ciede2000.py"` so downstream scripts reuse the perceptual metric.

## References

See `references.bib` for the raw BibTeX entries cited above: [@luo2001; @sharma2005].
