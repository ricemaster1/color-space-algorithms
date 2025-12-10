from __future__ import annotations

from PIL import Image
import webcolors
import argparse
import os
import sys
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

from rename_utils import GUI_AVAILABLE as GUI_RENAME_AVAILABLE
from rename_utils import launch_gui as rename_launch_gui
from rename_utils import prompt_cli as rename_prompt_cli
from rename_utils import undo_last as rename_undo_last
from rename_utils import iter_files as rename_iter_files

try:
    import curses
except Exception:  # pragma: no cover
    curses = None  # type: ignore

# ARMLite color names
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

# Map to simple RGB tuples (r, g, b)
def _to_tuple(rgb):
    try:
        return (rgb.red, rgb.green, rgb.blue)
    except AttributeError:
        return tuple(rgb)

ARMLITE_RGB = {name: _to_tuple(webcolors.name_to_rgb(name)) for name in ARMLITE_COLORS}

ANSI_RESET = '\033[0m'
ANSI_RED = '\033[91m'
ANSI_YELLOW = '\033[93m'
ANSI_GREEN = '\033[92m'
ANSI_CYAN = '\033[96m'
ANSI_DIM = '\033[2m'

# Sentinel used to detect whether optional CLI parameters were explicitly provided
ARG_UNSET = object()


class SingleValueAction(argparse.Action):
    """Prevent options that should appear only once from being repeated."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, ARG_UNSET)
        if current is not ARG_UNSET:
            flag = option_string or self.option_strings[-1]
            raise argparse.ArgumentError(self, f"{flag} may be provided at most once.")
        setattr(namespace, self.dest, values)

def color_distance(c1, c2):
    return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2

@lru_cache(maxsize=65536)
def closest_color(rgb):
    if not isinstance(rgb, tuple):
        rgb = tuple(rgb)
    best = None
    best_d = 10**9
    for name, c in ARMLITE_RGB.items():
        d = color_distance(rgb, c)
        if d < best_d:
            best_d = d
            best = name
    return best

def apply_ordered_dither(img):
    bayer2x2 = [[0, 2], [3, 1]]
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            t = bayer2x2[y % 2][x % 2]
            r = min(255, max(0, r + (t - 1) * 8))
            g = min(255, max(0, g + (t - 1) * 8))
            b = min(255, max(0, b + (t - 1) * 8))
            pixels[x, y] = (r, g, b)
    return img

def apply_floyd_steinberg(img):
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            oldp = pixels[x, y]
            cname = closest_color(oldp)
            newp = ARMLITE_RGB[cname]
            pixels[x, y] = newp
            err = (oldp[0]-newp[0], oldp[1]-newp[1], oldp[2]-newp[2])
            def add(ix, iy, f):
                if 0 <= ix < w and 0 <= iy < h:
                    r, g, b = pixels[ix, iy]
                    pixels[ix, iy] = (
                        min(255, max(0, int(r + err[0]*f))),
                        min(255, max(0, int(g + err[1]*f))),
                        min(255, max(0, int(b + err[2]*f))),
                    )
            add(x+1, y, 7/16)
            add(x-1, y+1, 3/16)
            add(x,   y+1, 5/16)
            add(x+1, y+1, 1/16)
    return img

def context_quantize(img, n=3):
    pixels = img.load()
    w, h = img.size
    def top_n(rgb):
        d = [(color_distance(rgb, c), name) for name, c in ARMLITE_RGB.items()]
        d.sort(key=lambda x: x[0])
        return [name for _, name in d[:n]]
    for y in range(h):
        for x in range(w):
            rgb = pixels[x, y]
            cands = top_n(rgb)
            if len(cands) == 1:
                pixels[x, y] = ARMLITE_RGB[cands[0]]
                continue
            # 4-neighborhood smoothness
            neigh = []
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h:
                    neigh.append(pixels[nx, ny])
            best_name = cands[0]
            best_score = 10**12
            for name in cands:
                crgb = ARMLITE_RGB[name]
                score = sum(color_distance(crgb, nrgb) for nrgb in neigh)
                if score < best_score:
                    best_score = score
                    best_name = name
            pixels[x, y] = ARMLITE_RGB[best_name]
    # Return final color-name grid
    return [[closest_color(pixels[x, y]) for x in range(w)] for y in range(h)]

def node_quantize(img, n=3, iterations=3):
    w, h = img.size
    src = list(img.getdata())
    # Precompute top-N per pixel
    def top_n(rgb):
        d = sorted(ARMLITE_RGB.items(), key=lambda it: color_distance(rgb, it[1]))
        return [name for name, _ in d[:n]]
    grid = [[top_n(src[y*w + x]) for x in range(w)] for y in range(h)]
    best = [[cands[0] for cands in row] for row in grid]
    for _ in range(iterations):
        for y in range(h):
            for x in range(w):
                rgb0 = src[y*w + x]
                best_energy = 10**12
                best_name = best[y][x]
                for name in grid[y][x]:
                    crgb = ARMLITE_RGB[name]
                    e = color_distance(crgb, rgb0)
                    for dx in (-1,0,1):
                        for dy in (-1,0,1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < w and 0 <= ny < h:
                                nrgb = ARMLITE_RGB[best[ny][nx]]
                                e += 0.5 * color_distance(crgb, nrgb)
                    if e < best_energy:
                        best_energy = e
                        best_name = name
                best[y][x] = best_name
    return best

def generate_assembly(color_grid, output_path):
    h = len(color_grid)
    w = len(color_grid[0]) if h else 0
    lines = [
        '; === Fullscreen Sprite ===',
        '    MOV R0, #2',
        '    STR R0, .Resolution',
        '    MOV R1, #.PixelScreen',
        '    MOV R6, #512 ; row stride (128 * 4)'
    ]
    for y in range(h):
        for x in range(w):
            offset = ((y * w) + x) * 4
            lines.append(f'    MOV R5, #{offset}')
            lines.append('    ADD R4, R1, R5')
            cname = color_grid[y][x]
            lines.append(f'    MOV R0, #.{cname}')
            lines.append(f'    STR R0, [R4]   ; Pixel ({x},{y})')
    lines.append('    HALT')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Assembly sprite file written to {output_path}")

def build_preview(color_grid, path):
    if not path:
        return
    h = len(color_grid)
    w = len(color_grid[0]) if h else 0
    img = Image.new('RGB', (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ARMLITE_RGB[color_grid[y][x]]
    img.save(path)
    print(f"Preview image saved to {path}")

ALGOS = {
    'nearest': 'Map each pixel to the closest ARMLite color (no dithering).',
    'ordered': 'Apply 2x2 ordered dithering, then map to closest color.',
    'floyd':   'Floyd-Steinberg error-diffusion dithering.',
    'context': 'Top-N neighbor-aware smoothing (no global iterations).',
    'node':    'Iterative node-based optimization with top-N candidates.'
}

def load_mapping(map_file: Path) -> dict:
    if map_file.exists():
        try:
            with map_file.open('r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k).lower(): str(v) for k, v in data.items()}
        except Exception:
            pass
    return {}

def save_mapping(map_file: Path, mapping: dict):
    try:
        map_file.parent.mkdir(parents=True, exist_ok=True)
        with map_file.open('w') as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save mapping file: {e}")

def select_active_mappings(mapping: dict) -> Optional[Set[str]]:
    if not sys.stdin.isatty():
        return set(mapping.keys())
    print('\nAvailable mapping keywords:')
    for idx, (key, dest) in enumerate(sorted(mapping.items()), start=1):
        print(f"  {idx}. {key} -> {dest}")
    print('Enter comma-separated keywords to enable, leave blank for all, or type "none" to disable mapping this run.')
    choice = input('Mappings to use: ').strip()
    if not choice:
        return set(mapping.keys())
    lowered = choice.lower()
    if lowered in {'none', 'no', 'n'}:
        return set()
    selected = {part.strip().lower() for part in choice.split(',') if part.strip()}
    unknown = selected - set(mapping.keys())
    if unknown:
        print(f"Warning: ignored unknown mapping keywords: {', '.join(sorted(unknown))}")
    selected &= set(mapping.keys())
    if not selected:
        print('No valid mapping keywords selected; mapping will be skipped for this run.')
    return selected

def sanitize_filename(name: str) -> str:
    safe = [c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name]
    return ''.join(safe) or 'image'

def next_group_directory(root: Path, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    slug = sanitize_filename(label) if label else ''
    stamp = datetime.now().strftime('%Y-%m-%d at %H.%M.%S')
    dir_name = stamp if not slug else f"{stamp}-{slug}"
    candidate = root / dir_name
    counter = 2
    while candidate.exists():
        suffix = f"{stamp} {counter}"
        dir_name = suffix if not slug else f"{suffix}-{slug}"
        candidate = root / dir_name
        counter += 1
    return candidate

def resolve_output_dir(
    base_out: Path,
    name: str,
    mapping: dict,
    use_mapping: bool,
    active_keys: Optional[Set[str]] = None,
    structure: str = 'flat',
    source_path: Optional[Path] = None,
) -> tuple[Path, Path, bool]:
    if use_mapping and mapping:
        lname = name.lower()
        # choose the longest matching keyword
        matches = [
            (k, mapping[k])
            for k in mapping.keys()
            if k in lname and (active_keys is None or k in active_keys)
        ]
        if matches:
            matches.sort(key=lambda kv: len(kv[0]), reverse=True)
            mapped_base = Path(os.path.expanduser(matches[0][1]))
            if structure == 'nested':
                subdir = mapped_base / sanitize_filename(name)
                return subdir, mapped_base, True
            return mapped_base, mapped_base, True
        if source_path is not None:
            source_resolved = source_path.resolve()
            for key, dest in mapping.items():
                if active_keys is not None and key not in active_keys:
                    continue
                mapped_base = Path(os.path.expanduser(dest)).resolve()
                try:
                    source_resolved.relative_to(mapped_base)
                except ValueError:
                    continue
                if structure == 'nested':
                    subdir = mapped_base / sanitize_filename(name)
                    return subdir, mapped_base, True
                return mapped_base, mapped_base, True
    return base_out, base_out, False


class SelectionCancelled(Exception):
    pass


def _relative_label(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def _fallback_select(labels: list[str]) -> Optional[list[int]]:
    print('Select files to include:')
    for idx, label in enumerate(labels, start=1):
        print(f"  {idx}. {label}")
    choice = input('Enter numbers/ranges (comma separated), leave blank for all, or q to cancel: ').strip()
    if not choice:
        return list(range(len(labels)))
    lowered = choice.lower()
    if lowered in {'q', 'quit'}:
        return None
    selected: set[int] = set()
    parts = [part.strip() for part in choice.replace(' ', '').split(',') if part.strip()]
    for part in parts:
        if '-' in part:
            try:
                start_str, end_str = part.split('-', 1)
                start_idx = int(start_str)
                end_idx = int(end_str)
            except ValueError:
                print(f"  Ignored invalid range: {part}")
                continue
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            for val in range(start_idx, end_idx + 1):
                if 1 <= val <= len(labels):
                    selected.add(val - 1)
        else:
            try:
                val = int(part)
            except ValueError:
                print(f"  Ignored invalid entry: {part}")
                continue
            if 1 <= val <= len(labels):
                selected.add(val - 1)
            else:
                print(f"  Ignored out-of-range index: {val}")
    return sorted(selected) if selected else []


def _select_with_curses(labels: list[str]) -> Optional[list[int]]:
    if curses is None:
        return None

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        selected = set(range(len(labels)))
        cursor = 0
        top = 0
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            visible = max(1, height - 2)
            if cursor < top:
                top = cursor
            elif cursor >= top + visible:
                top = cursor - visible + 1
            end = min(len(labels), top + visible)
            for row, idx in enumerate(range(top, end)):
                mark = '[x]' if idx in selected else '[ ]'
                line = f"{mark} {labels[idx]}"
                if idx == cursor:
                    stdscr.attron(curses.A_REVERSE)
                    stdscr.addnstr(row, 0, line, width - 1)
                    stdscr.attroff(curses.A_REVERSE)
                else:
                    stdscr.addnstr(row, 0, line, width - 1)
            help_text = 'Space toggle · a all · n none · Enter confirm · q cancel'
            stdscr.addnstr(height - 1, 0, help_text, width - 1)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                cursor = (cursor - 1) % len(labels)
            elif key in (curses.KEY_DOWN, ord('j')):
                cursor = (cursor + 1) % len(labels)
            elif key == ord(' '):
                if cursor in selected:
                    selected.remove(cursor)
                else:
                    selected.add(cursor)
            elif key in (ord('a'), ord('A')):
                selected = set(range(len(labels)))
            elif key in (ord('n'), ord('N')):
                selected.clear()
            elif key in (10, 13, curses.KEY_ENTER):
                return sorted(selected)
            elif key in (27, ord('q'), ord('Q')):
                raise SelectionCancelled()

    try:
        return curses.wrapper(_run)
    except SelectionCancelled:
        raise
    except Exception:
        return None


def select_files_subset(items: list[Path], base_dir: Path) -> Optional[list[Path]]:
    if not items:
        return []
    labels = [_relative_label(base_dir, p) for p in items]
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            selected_indices = _select_with_curses(labels)
        except SelectionCancelled:
            return None
        if selected_indices is None:
            selected_indices = _fallback_select(labels)
            if selected_indices is None:
                return None
    else:
        print('Non-interactive session detected; selecting all files.')
        selected_indices = list(range(len(items)))

    if selected_indices is None:
        return None
    if not selected_indices:
        return []
    return [items[idx] for idx in selected_indices if 0 <= idx < len(items)]

def iter_input_images(input_path: Path):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    if input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    elif input_path.is_file():
        if input_path.suffix.lower() in exts:
            yield input_path
    else:
        return

def format_output_name(rename_mode: str, pattern: str, base_name: str, index: int, algo: str, n: int) -> str:
    if rename_mode == 'pattern' and pattern:
        # Provide format fields: name, index, algo, n
        try:
            return pattern.format(name=base_name, index=index, algo=algo, n=n)
        except Exception:
            pass
    elif rename_mode == 'prompt':
        try:
            newn = input(f"Output name for '{base_name}' (empty to keep): ").strip()
            if newn:
                return newn
        except EOFError:
            pass
    # default
    return base_name

def run_convert(args) -> None:
    if args.algo.lower() in ('help', 'list', '?'):
        print('Available algorithms:')
        for key, desc in ALGOS.items():
            print(f'  - {key}: {desc}')
        return

    pattern_provided = args.pattern is not ARG_UNSET
    if args.rename_mode == 'pattern':
        pattern_template = args.pattern if pattern_provided else '{name}'
    else:
        if pattern_provided:
            print('Error: --pattern requires --rename-mode pattern.')
            raise SystemExit(2)
        pattern_template = ''

    script_dir = Path(__file__).resolve().parent
    input_path = Path(os.path.expanduser(args.input))
    if not input_path.exists():
        alt = script_dir / args.input
        if alt.exists():
            input_path = alt
        elif args.input == 'images':
            alt.mkdir(parents=True, exist_ok=True)
            input_path = alt
        else:
            print(f"Input path not found: {input_path}")
            print('Hint: provide an existing image file or directory, or create an "images" folder next to armlite.py.')
            raise SystemExit(1)

    base_out = Path(os.path.expanduser(args.output_dir))
    preview_dir = Path(os.path.expanduser(args.preview)) if args.preview is not None else None

    map_file = Path(os.path.expanduser(args.map_file))
    mapping = load_mapping(map_file)
    updated = False
    for spec in args.map_add:
        if '=' not in spec:
            print(f"Warning: ignored mapping '{spec}' (expected KEY=PATH format).")
            continue
        key, path_value = spec.split('=', 1)
        key = key.strip().lower()
        expanded = os.path.expanduser(path_value.strip())
        target = Path(expanded)
        if target.exists():
            if not target.is_dir():
                print(f"Warning: mapping '{key}' skipped because '{target}' is not a directory.")
                continue
        else:
            print(f"Warning: mapping '{key}' skipped because '{target}' does not exist. Create it first or supply a valid directory.")
            continue
        mapping[key] = str(target.resolve())
        updated = True
    if updated:
        save_mapping(map_file, mapping)
        if not args.map_use:
            print(f"Mapping updated in {map_file}.")

    group_mode = args.map_group
    group_enabled = False
    group_root: Optional[Path] = None
    group_dir: Optional[Path] = None
    group_preview_dir: Optional[Path] = None
    if args.map_group_label:
        group_label_raw = args.map_group_label
    else:
        if input_path.is_dir():
            group_label_raw = input_path.name or 'batch'
        else:
            group_label_raw = input_path.stem or input_path.name or 'batch'
    group_label_slug = sanitize_filename(group_label_raw) or 'batch'

    if args.map_use and mapping:
        if group_mode == 'on':
            group_enabled = True
        elif group_mode == 'ask' and sys.stdin.isatty():
            answer = input('Group mapped outputs into a batch folder? [y/N]: ').strip().lower()
            group_enabled = answer in {'y', 'yes'}
    if group_enabled:
        if args.map_group_root:
            group_root = Path(os.path.expanduser(args.map_group_root))
        elif input_path.is_dir():
            group_root = input_path / 'map-out'
        else:
            group_root = base_out.parent / 'map-out'
        group_root.mkdir(parents=True, exist_ok=True)

    items = list(iter_input_images(input_path))
    if not items:
        if args.map_add and not args.map_use:
            print('Mapping updated. Add images later or rerun with a valid input path when ready to convert.')
            return
        print(f'No input images found in {input_path}.')
        if args.input == 'images':
            print('Hint: drop image files into the newly created "images" folder and rerun, or pass a different path.')
        raise SystemExit(1)

    if args.select_files:
        base_dir_for_labels = input_path if input_path.is_dir() else input_path.parent
        selection = select_files_subset(items, base_dir_for_labels)
        if selection is None:
            print('Selection cancelled. Exiting.')
            return
        if not selection:
            print('No files selected. Exiting.')
            return
        items = selection

    active_keys: Optional[Set[str]] = None
    if args.map_use and mapping:
        active_keys = select_active_mappings(mapping)
        if active_keys is not None and not active_keys:
            print('Mapping disabled for this run; outputs will go to the base directory.')

    algo = args.algo.lower()
    base_out.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    total_items = len(items)
    progress_mode = args.progress
    if progress_mode == 'on':
        show_progress = True
    elif progress_mode == 'off':
        show_progress = False
    else:
        show_progress = sys.stdout.isatty()

    color_enabled = show_progress and sys.stdout.isatty() and os.getenv('NO_COLOR') is None

    def render_progress(done: int) -> None:
        if not show_progress or total_items == 0:
            return
        width = 30
        ratio = max(0, min(1, done / total_items))
        filled = int(width * ratio)
        empty = width - filled
        fill_segment = '#' * filled
        remainder_segment = '-' * empty
        if color_enabled and filled > 0:
            if ratio < 0.34:
                color = ANSI_RED
            elif ratio < 0.67:
                color = ANSI_YELLOW
            else:
                color = ANSI_GREEN
            fill_segment = f"{color}{fill_segment}{ANSI_RESET}"
            prefix = f"{ANSI_CYAN}[{ANSI_RESET}"
            suffix = f"{ANSI_CYAN}]{ANSI_RESET} {ANSI_DIM}{done}/{total_items}{ANSI_RESET}"
        elif color_enabled:
            prefix = f"{ANSI_CYAN}[{ANSI_RESET}"
            suffix = f"{ANSI_CYAN}]{ANSI_RESET} {ANSI_DIM}{done}/{total_items}{ANSI_RESET}"
        else:
            prefix = '['
            suffix = f'] {done}/{total_items}'
        bar = f"{fill_segment}{remainder_segment}"
        print(f"\r{prefix}{bar}{suffix}", end='', flush=True)

    if show_progress:
        render_progress(0)

    map_structure = args.map_structure

    for idx, img_path in enumerate(items, start=1):
        name = img_path.stem
        dest_dir, mapped_base, mapped = resolve_output_dir(
            base_out,
            name,
            mapping,
            args.map_use,
            active_keys,
            structure=map_structure,
            source_path=img_path,
        )
        target_dir = dest_dir
        preview_target: Optional[Path] = preview_dir

        if mapped:
            if group_enabled and group_root is not None:
                if group_dir is None:
                    group_dir = next_group_directory(group_root, group_label_slug)
                target_dir = group_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                if preview_dir is not None:
                    if group_preview_dir is None:
                        if group_label_slug:
                            preview_label_dir = preview_dir / group_label_slug
                            preview_label_dir.mkdir(parents=True, exist_ok=True)
                            group_preview_dir = preview_label_dir / group_dir.name
                        else:
                            group_preview_dir = preview_dir / group_dir.name
                    preview_target = group_preview_dir
                    preview_target.mkdir(parents=True, exist_ok=True)
            elif map_structure == 'nested':
                target_dir = dest_dir
                if preview_dir is not None:
                    preview_target = preview_dir / dest_dir.name
            else:
                target_dir = mapped_base
                if preview_dir is not None:
                    preview_target = preview_dir / mapped_base.name
        else:
            target_dir = base_out
            if preview_dir is not None:
                preview_target = preview_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        out_name = format_output_name(args.rename_mode, pattern_template, name, idx, algo, args.n)
        asm_path = target_dir / f"{out_name}.s"
        prev_path = None
        if preview_target is not None:
            preview_target.mkdir(parents=True, exist_ok=True)
            prev_path = preview_target / f"{out_name}.png"

        if args.dry_run:
            print(f"[DRY] {img_path} -> {asm_path}{' | preview: ' + str(prev_path) if prev_path else ''}")
            render_progress(idx)
            continue

        img = Image.open(str(img_path)).convert('RGB').resize((128, 96))

        if algo == 'nearest':
            grid = [[closest_color(img.getpixel((x, y))) for x in range(128)] for y in range(96)]
        elif algo == 'ordered':
            img = apply_ordered_dither(img)
            grid = [[closest_color(img.getpixel((x, y))) for x in range(128)] for y in range(96)]
        elif algo == 'floyd':
            img = apply_floyd_steinberg(img)
            grid = [[closest_color(img.getpixel((x, y))) for x in range(128)] for y in range(96)]
        elif algo == 'context':
            grid = context_quantize(img, n=max(1, args.n))
        elif algo == 'node':
            grid = node_quantize(img, n=max(1, args.n))
        else:
            print(f"Unknown algorithm '{args.algo}'. Use -a help to list options.")
            raise SystemExit(2)

        if prev_path is not None:
            build_preview(grid, str(prev_path))

        generate_assembly(grid, str(asm_path))
        render_progress(idx)

    if show_progress:
        print()


def run_rename(args) -> None:
    directory = Path(os.path.expanduser(args.directory)).resolve()
    if not directory.exists():
        print(f'Directory not found: {directory}')
        raise SystemExit(1)
    if not directory.is_dir():
        print(f'Provided path is not a directory: {directory}')
        raise SystemExit(1)

    pattern_provided = args.pattern is not ARG_UNSET
    pattern_value: Optional[str] = args.pattern if pattern_provided else None
    if args.mode != 'pattern' and pattern_provided:
        print('Error: --pattern requires --mode pattern.')
        raise SystemExit(2)

    if args.undo:
        undone, warnings = rename_undo_last(directory, dry_run=args.dry_run)
        if undone:
            for line in undone:
                print(f'UNDO: {line}')
        if warnings:
            for msg in warnings:
                print(f'Warning: {msg}')
        if args.dry_run and undone:
            print('(dry run – no changes made)')
        return

    if args.mode == 'gui':
        if not GUI_RENAME_AVAILABLE:
            print('GUI renaming is unavailable on this system.')
            raise SystemExit(1)
        rename_launch_gui(directory)
        return

    files = rename_iter_files(directory)
    if not files:
        print('No files found to rename.')
        return

    selected_files = files
    if args.select_files:
        selection = select_files_subset(files, directory)
        if selection is None:
            print('Selection cancelled. Exiting.')
            return
        if not selection:
            print('No files selected. Exiting.')
            return
        selected_files = selection

    if args.mode == 'pattern':
        if not pattern_value:
            print('Pattern mode requires --pattern.')
            raise SystemExit(2)
        rename_prompt_cli(directory, pattern_value, args.start, args.dry_run, files=selected_files)
    else:
        rename_prompt_cli(directory, None, args.start, args.dry_run, files=selected_files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='armlite.py', description='ARMLite toolkit')
    subparsers = parser.add_subparsers(dest='command', required=True)

    convert = subparsers.add_parser('convert', help='Convert images to ARMLite assembly code')
    convert.set_defaults(func=run_convert)
    convert.add_argument('input', nargs='?', default='images', help='Path to input image or directory (default: images)')
    convert.add_argument('-O', '--output-dir', default='output', help='Directory to write .s files (default: output)')
    convert.add_argument('-a', '--algo', default='nearest', help="Algorithm: nearest|ordered|floyd|context|node or 'help' to list")
    convert.add_argument('-n', type=int, default=3, help='Top-N for context/node algorithms (default: 3)')
    convert.add_argument('--preview', nargs='?', const='preview', help='Save preview PNGs. If value is a directory, files go there; with no value, uses ./preview')
    convert.add_argument('--rename-mode', choices=['default', 'prompt', 'pattern'], default='default', help='Control output filenames (default: base name)')
    convert.add_argument('--pattern', default=ARG_UNSET, action=SingleValueAction, help='Filename pattern with fields {name}, {index}, {algo}, {n} (requires --rename-mode pattern; default: {name})')
    convert.add_argument('--map-use', action='store_true', help='Enable keyword→folder mapping to auto-route outputs (interactive selection when run in a terminal)')
    convert.add_argument('--map-add', action='append', default=[], help='Add keyword→folder mapping KEY=PATH (repeatable). Run by itself to update the map file.')
    convert.add_argument('--map-file', default=str(Path.home() / '.armlite_output_map.json'), help='Path to mapping JSON (default: ~/.armlite_output_map.json)')
    convert.add_argument('--map-structure', choices=['flat', 'nested'], default='flat', help='How to store outputs inside mapped folders (default: flat)')
    convert.add_argument('--map-group', choices=['ask', 'off', 'on'], default='ask', help='Group mapped outputs into a per-run directory (ask when interactive by default)')
    convert.add_argument('--map-group-root', help='Root directory for grouped mapped outputs (default: map-out alongside the input directory)')
    convert.add_argument('--map-group-label', help='Label used when creating grouped directories (default: input folder name)')
    convert.add_argument('--dry-run', action='store_true', help='Print planned actions without writing files')
    convert.add_argument('--select-files', action='store_true', help='Interactively choose which input files to process before mapping')
    convert.add_argument('--progress', choices=['auto', 'on', 'off'], default='auto', help='Display a simple progress bar while converting (auto when running in a terminal)')

    rename = subparsers.add_parser('rename', help='Rename source files with CLI, pattern, or GUI workflow')
    rename.set_defaults(func=run_rename)
    rename.add_argument('directory', nargs='?', default='images', help='Directory containing files to rename (default: images)')
    rename.add_argument('--mode', choices=['cli', 'pattern', 'gui'], default='cli', help='Rename via per-file prompt, pattern, or GUI (default: cli)')
    rename.add_argument('--pattern', default=ARG_UNSET, action=SingleValueAction, help='Pattern for pattern mode (requires --mode pattern; use {stem}, {index}, {ext})')
    rename.add_argument('--start', type=int, default=1, help='Start index for pattern rename (default: 1)')
    rename.add_argument('--dry-run', action='store_true', help='Preview planned renames without applying changes')
    rename.add_argument('--undo', action='store_true', help='Undo the most recent rename batch')
    rename.add_argument('--select-files', action='store_true', help='Interactively choose which files to rename (cli/pattern modes)')

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
