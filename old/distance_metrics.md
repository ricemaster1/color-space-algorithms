# Distance Metrics

Distance modules score how different a source pixel is from each ARMLite palette entry. Plug them into quantizers, run them standalone to inspect palette mapping, or use them as building blocks for new pipelines. All scripts output `.s` files, so you can immediately see how a metric behaves across an entire frame.

## Choosing a Metric
- **Euclidean RGB** is simple but overemphasizes midtones.
- **Delta E (CIE76/94/2000)** models human perception by operating in Lab space.
- **Hybrid/Adaptive Delta E** blends formulas based on chroma or luminance thresholds.
- **Mahalanobis** weighs RGB axes based on the source image covariance, great for scenes dominated by one color family.

## Modules

### distance_euclidean.py
```bash
python algorithms/distance_metrics/distance_euclidean.py ref.png -o ref_euclid.s
```
Parameters: `image`, `-o/--output`.

### distance_cie76.py
```bash
python algorithms/distance_metrics/distance_cie76.py ref.png -o ref_76.s
```
Parameters: `image`, `-o/--output`.

### distance_cie94.py
```bash
python algorithms/distance_metrics/distance_cie94.py ref.png -o ref_94.s
```
Parameters: `image`, `-o/--output`.

### distance_ciede2000.py
```bash
python algorithms/distance_metrics/distance_ciede2000.py ref.png -o ref_2000.s
```
Parameters: `image`, `-o/--output`.

### distance_delta_e.py
Switch between Delta E variants without editing code.
```bash
python algorithms/distance_metrics/distance_delta_e.py ref.png -m hybrid --cie94-application textiles -o ref_hybrid.s
```
Parameters: `image`, `-o/--output`, `-m/--metric` (`ciede2000`, `cie94`, `cie76`, `2000`, `94`, `76`, `hybrid`), `--cie94-application` (`graphic_arts`, `textiles`).

### distance_delta_e_neo.py
Adaptive system that blends formulas based on scene characteristics.
```bash
python algorithms/distance_metrics/distance_delta_e_neo.py ref.png -m adaptive --chroma-threshold 8 --luminance-threshold 15 --hybrid-weights 0.5,0.3,0.2 -o ref_neo.s
```
Parameters: `image`, `-o/--output`, `-m/--metric`, `--cie94-application`, `--chroma-threshold`, `--luminance-threshold`, `--hybrid-weights` (comma separated triple parsed by the script).

### distance_mahalanobis.py
```bash
python algorithms/distance_metrics/distance_mahalanobis.py ref.png --epsilon 0.0005 -o ref_maha.s
```
Parameters: `image`, `-o/--output`, `--epsilon` (regularization term, default 1e-3).

Use these metrics to pre-score palettes (generate a `.s` file with a heat map), or import their helper functions into quantizers for tighter integration.
