"""Shared helpers for batch file renaming (CLI + optional GUI)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import json

try:  # GUI components are optional
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    filedialog = messagebox = ttk = None  # type: ignore

GUI_AVAILABLE = tk is not None

HISTORY_FILE = '.armlite_rename_history.json'


@dataclass
class RenamePlan:
    original: Path
    target: Path


def iter_files(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and not p.name.startswith('.')
    )


def format_pattern(pattern: str, file: Path, index: int) -> str:
    stem = file.stem
    ext = file.suffix
    parent = file.parent.name
    return pattern.format(index=index, stem=stem, ext=ext, parent=parent)


def build_plans(files: Sequence[Path], pattern: str, start: int) -> List[RenamePlan]:
    plans: List[RenamePlan] = []
    for offset, file in enumerate(files):
        new_name = format_pattern(pattern, file, start + offset)
        target = file.with_name(new_name)
        plans.append(RenamePlan(file, target))
    return plans


def apply_plans(plans: Iterable[RenamePlan], dry_run: bool = False) -> List[str]:
    logs: List[str] = []
    for plan in plans:
        if plan.target.exists():
            raise FileExistsError(f"Target already exists: {plan.target}")
        logs.append(f"{plan.original.name} -> {plan.target.name}")
        if not dry_run:
            plan.original.rename(plan.target)
    return logs


def _history_path(directory: Path) -> Path:
    return directory / HISTORY_FILE


def _load_history(directory: Path) -> List[dict]:
    path = _history_path(directory)
    if not path.exists():
        return []
    try:
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []


def _write_history(directory: Path, history: List[dict]) -> None:
    path = _history_path(directory)
    try:
        with path.open('w', encoding='utf-8') as handle:
            json.dump(history, handle, indent=2)
    except OSError as exc:
        print(f"Warning: failed to update rename history in {path}: {exc}")


def _rel_or_name(directory: Path, path: Path) -> str:
    try:
        return path.relative_to(directory).as_posix()
    except ValueError:
        return path.name


def record_history(directory: Path, plans: Sequence[RenamePlan]) -> None:
    if not plans:
        return
    batch = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'entries': [
            {'from': _rel_or_name(directory, plan.original), 'to': _rel_or_name(directory, plan.target)}
            for plan in plans
        ],
    }
    history = _load_history(directory)
    history.append(batch)
    _write_history(directory, history)


def undo_last(directory: Path, dry_run: bool = False) -> Tuple[List[str], List[str]]:
    directory = directory.resolve()
    history = _load_history(directory)
    if not history:
        return [], ["No rename history to undo."]
    batch = history[-1]
    entries = batch.get('entries') or []
    successes: List[str] = []
    failures: List[str] = []
    ops = list(entries)
    failed_indices = set()
    for idx in reversed(range(len(ops))):
        entry = ops[idx]
        src = directory / entry['to']
        dest = directory / entry['from']
        if not src.exists():
            failures.append(f"Missing file: {entry['to']}")
            failed_indices.add(idx)
            continue
        if dest.exists():
            failures.append(f"Target already exists: {entry['from']}")
            failed_indices.add(idx)
            continue
        successes.append(f"{entry['to']} -> {entry['from']}")
        if not dry_run:
            src.rename(dest)
    if not dry_run:
        if failed_indices:
            history[-1]['entries'] = [ops[i] for i in sorted(failed_indices)]
        else:
            history.pop()
        _write_history(directory, history)
    return successes[::-1], failures


def prompt_cli(
    directory: Path,
    pattern: Optional[str],
    start: int,
    dry_run: bool,
    files: Optional[Sequence[Path]] = None,
) -> None:
    files = list(files) if files is not None else iter_files(directory)
    if not files:
        print("No files found to rename.")
        return

    if pattern:
        plans = build_plans(files, pattern, start)
        logs = apply_plans(plans, dry_run)
        for line in logs:
            print(line)
        if dry_run:
            print("(dry run – no changes made)")
        else:
            record_history(directory, plans)
        return

    manual_plans: List[RenamePlan] = []
    for file in files:
        while True:
            prompt = f"Rename '{file.name}' to (blank to keep): "
            try:
                new_name = input(prompt)
            except EOFError:
                print()  # newline for clean exit
                return
            new_name = new_name.strip()
            if not new_name:
                break
            target = file.with_name(new_name)
            if target.exists():
                print("  Target already exists. Try again.")
                continue
            print(f"  {file.name} -> {target.name}")
            if not dry_run:
                file.rename(target)
                manual_plans.append(RenamePlan(file, target))
            break
    if dry_run:
        print("(dry run – no changes made)")
    elif manual_plans:
        record_history(directory, manual_plans)


class RenameApp(tk.Tk):  # type: ignore[misc]
    def __init__(self, directory: Path):
        super().__init__()
        self.title("ARMLite Image Batch Renamer")
        self.resizable(False, False)
        self.directory = directory
        self.files: List[Path] = []
        self.pattern_var = tk.StringVar(value="{stem}_{index:02d}{ext}")
        self.start_var = tk.StringVar(value="1")
        self.selected_name = tk.StringVar()
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 5}

        path_frame = ttk.Frame(self)
        path_frame.grid(row=0, column=0, sticky="ew", **padding)
        ttk.Label(path_frame, text=f"Directory: {self.directory}").grid(row=0, column=0, sticky="w")
        ttk.Button(path_frame, text="Change…", command=self._choose_directory).grid(row=0, column=1, padx=(10, 0))

        self.listbox = tk.Listbox(self, width=60, height=12, exportselection=False)
        self.listbox.grid(row=1, column=0, sticky="nsew", **padding)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        entry_frame = ttk.Frame(self)
        entry_frame.grid(row=2, column=0, sticky="ew", **padding)
        ttk.Label(entry_frame, text="Rename selected to:").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(entry_frame, textvariable=self.selected_name, width=40)
        entry.grid(row=0, column=1, padx=(5, 0))
        ttk.Button(entry_frame, text="Rename", command=self._rename_selected).grid(row=0, column=2, padx=(5, 0))

        pattern_frame = ttk.LabelFrame(self, text="Pattern rename")
        pattern_frame.grid(row=3, column=0, sticky="ew", **padding)
        ttk.Label(pattern_frame, text="Pattern (use {stem}, {index}, {ext}):").grid(row=0, column=0, sticky="w")
        ttk.Entry(pattern_frame, textvariable=self.pattern_var, width=45).grid(row=0, column=1, padx=(5, 0))
        ttk.Label(pattern_frame, text="Start index:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(pattern_frame, textvariable=self.start_var, width=6).grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(5, 0))
        ttk.Button(pattern_frame, text="Apply to all", command=self._apply_pattern).grid(row=0, column=2, rowspan=2, padx=(10, 0))

        ttk.Button(self, text="Refresh", command=self._refresh).grid(row=4, column=0, pady=(0, 10))

    def _choose_directory(self) -> None:
        new_dir = filedialog.askdirectory(initialdir=str(self.directory))
        if new_dir:
            self.directory = Path(new_dir)
            self._refresh()

    def _refresh(self) -> None:
        self.files = iter_files(self.directory)
        self.listbox.delete(0, tk.END)
        for file in self.files:
            self.listbox.insert(tk.END, file.name)
        self.selected_name.set("")

    def _current_selection(self) -> Optional[Path]:
        selection = self.listbox.curselection()
        if not selection:
            return None
        return self.files[selection[0]]

    def _on_select(self, _event) -> None:
        file = self._current_selection()
        if file:
            self.selected_name.set(file.name)

    def _rename_selected(self) -> None:
        file = self._current_selection()
        if not file:
            messagebox.showinfo("Rename", "Select a file to rename.")
            return
        new_name = self.selected_name.get().strip()
        if not new_name:
            messagebox.showerror("Rename", "New name cannot be empty.")
            return
        target = file.with_name(new_name)
        if target.exists():
            messagebox.showerror("Rename", "Target name already exists.")
            return
        try:
            file.rename(target)
        except OSError as exc:
            messagebox.showerror("Rename", f"Failed to rename file: {exc}")
            return
        record_history(self.directory, [RenamePlan(file, target)])
        self._refresh()
        messagebox.showinfo("Rename", f"Renamed to {new_name}")

    def _apply_pattern(self) -> None:
        pattern = self.pattern_var.get().strip()
        if not pattern:
            messagebox.showerror("Pattern", "Pattern cannot be empty.")
            return
        try:
            start = int(self.start_var.get() or 1)
        except ValueError:
            messagebox.showerror("Pattern", "Start index must be an integer.")
            return
        plans = build_plans(self.files, pattern, start)
        try:
            logs = apply_plans(plans, dry_run=False)
        except Exception as exc:
            messagebox.showerror("Pattern", str(exc))
            self._refresh()
            return
        record_history(self.directory, plans)
        self._refresh()
        messagebox.showinfo("Pattern", "\n".join(logs))


def launch_gui(directory: Optional[Path]) -> None:
    if tk is None:
        print("Tkinter is not available on this system. Use CLI renaming options instead.")
        raise SystemExit(1)
    if directory is None:
        initial = filedialog.askdirectory(title="Select directory to rename")
        if not initial:
            print("No directory selected.")
            return
        directory = Path(initial)
    app = RenameApp(directory)
    app.mainloop()


__all__ = [
    "RenamePlan",
    "apply_plans",
    "build_plans",
    "format_pattern",
    "iter_files",
    "launch_gui",
    "prompt_cli",
    "undo_last",
    "GUI_AVAILABLE",
]
