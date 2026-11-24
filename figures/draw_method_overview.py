#!/usr/bin/env python3
"""Rebuild the method figure used by the paper and README."""

from pathlib import Path
from runpy import run_path
from shutil import copy2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "figures" / "draw_method_main.py"


def main() -> None:
    run_path(str(SOURCE), run_name="__main__")
    for ext in ("png", "svg"):
        source = ROOT / "paper" / "figures" / f"fig_method_main.{ext}"
        target = ROOT / "assets" / f"quotientopt_method_overview.{ext}"
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)
        print(target)


if __name__ == "__main__":
    main()
