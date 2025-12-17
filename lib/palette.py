"""Shared ARMLite palette constants and color utilities.

This module provides the canonical 147 CSS3 named colors used by ARMLite
along with color distance and matching functions.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import webcolors

# ARMLite color names (CSS3 named colors)
ARMLITE_COLORS = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood',
    'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',
    'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
    'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
    'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue',
    'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
    'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew',
    'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush',
    'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink',
    'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
    'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen',
    'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
    'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose',
    'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
    'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray',
    'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
    'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]


def _to_tuple(rgb) -> Tuple[int, int, int]:
    """Convert webcolors RGB to a plain tuple."""
    try:
        return (rgb.red, rgb.green, rgb.blue)
    except AttributeError:
        return tuple(rgb)


# Map color names to RGB tuples
ARMLITE_RGB: Dict[str, Tuple[int, int, int]] = {
    name: _to_tuple(webcolors.name_to_rgb(name)) for name in ARMLITE_COLORS
}


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
    """Compute squared Euclidean distance between two RGB colors."""
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2


@lru_cache(maxsize=65536)
def closest_color(rgb: Tuple[int, int, int]) -> str:
    """Find the closest ARMLite color name to the given RGB tuple."""
    if not isinstance(rgb, tuple):
        rgb = tuple(rgb)
    best = 'black'
    best_d = 10 ** 9
    for name, c in ARMLITE_RGB.items():
        d = color_distance(rgb, c)
        if d < best_d:
            best_d = d
            best = name
    return best
