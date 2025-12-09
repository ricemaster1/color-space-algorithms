# Gallery & Proof of Concept

This page will showcase before/after comparisons along with the exact commands used to generate each sprite. For now, use the template below when contributing new examples.

## Template
1. **Source image**: `assets/mountain.png`
2. **Pipeline**:
   ```bash
   python algorithms/color_transforms/rgb_to_lab.py assets/mountain.png --weights 1.1,1,0.8 -o build/mountain_lab.png
   python algorithms/quantizers/wu_quantizer.py build/mountain_lab.png -o build/mountain_wu.s
   python algorithms/dithers/sierra.py assets/mountain.png --variant sierra3 -o build/mountain_final.s
   ```
3. **Screenshots**: include PNG/GIF preview plus ARMLite screenshot.
4. **Notes**: highlight why this combo works (e.g., “CIEDE2000 preserved subtle blues, Sierra added texture”).

Submit PRs that add sections following this format, and link to them from `pipelines.md` when relevant.
