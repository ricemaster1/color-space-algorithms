#!/usr/bin/env python3
"""Interactive batch file renamer for image conversion workflows."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from rename_utils import GUI_AVAILABLE, launch_gui, prompt_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive batch file renamer")
    parser.add_argument("directory", nargs="?", help="Directory containing files to rename")
    parser.add_argument("--pattern", help="Pattern rename in CLI mode (use {stem}, {index}, {ext})")
    parser.add_argument("--start", type=int, default=1, help="Start index for pattern renames (default: 1)")
    parser.add_argument("--cli", action="store_true", help="Run in terminal-only mode (no GUI)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without renaming")
    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve() if args.directory else None
    if directory and not directory.exists():
        parser.error(f"Directory does not exist: {directory}")

    if args.cli or not GUI_AVAILABLE:
        if directory is None:
            parser.error("CLI mode requires a directory argument.")
        prompt_cli(directory, args.pattern, args.start, args.dry_run)
        return

    launch_gui(directory)


if __name__ == "__main__":
    main()
