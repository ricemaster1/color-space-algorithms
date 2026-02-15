ARMLite Algorithm Suite
======================

Documentation for the ARMLite image-to-sprite algorithm collection.
Each algorithm converts source imagery into assembly listings compatible with the
`ARMLite Simulator <https://peterhigginson.co.uk/ARMlite>`_.

.. note::

   This documentation is auto-built from the repository source.
   For environment setup, see the `setup guide <https://github.com/ricemaster1/color-space-algorithms/blob/main/algorithms/setup.md>`_.

----

Quantizers
----------

.. toctree::
   :maxdepth: 2

   quantizers/bsp_partitioning
   quantizers/k_means
   quantizers/kd_tree_palette
   quantizers/median_cut
   quantizers/neuquant
   quantizers/node
   quantizers/nthree
   quantizers/octree
   quantizers/palette_graph_nn
   quantizers/quantizer
   quantizers/som_quantizer
   quantizers/voronoi_palette
   quantizers/wu_quantizer
   quantizers/wu_quantizer-001

Dithers
-------

.. toctree::
   :maxdepth: 2

   dithers/atkinson
   dithers/floyd-steinberg
   dithers/jarvis_judice_ninke
   dithers/sierra
   dithers/stucki

Distance Metrics
----------------

.. toctree::
   :maxdepth: 2

   distance_metrics/distance_cie76
   distance_metrics/distance_cie94
   distance_metrics/distance_ciede2000
   distance_metrics/distance_delta_e
   distance_metrics/distance_delta_e_neo
   distance_metrics/distance_euclidean
   distance_metrics/distance_mahalanobis

Color Transforms
----------------

.. toctree::
   :maxdepth: 2

   color_transforms/rgb_to_hsv_hsl
   color_transforms/rgb_to_lab
   color_transforms/rgb_to_xyz
   color_transforms/rgb_to_ycbcr

