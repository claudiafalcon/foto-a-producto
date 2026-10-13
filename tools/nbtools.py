"""Tiny helper to write .ipynb files from a list of cells (no nbformat dependency)."""
import json
from pathlib import Path

CELLS = []


def reset():
    CELLS.clear()


def md(source):
    CELLS.append(("markdown", source.strip("\n")))


def code(source):
    CELLS.append(("code", source.strip("\n")))


def build(path, gpu=True):
    metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        metadata.update(accelerator="GPU")
        metadata["colab"]["gpuType"] = "T4"
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": metadata, "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", path, len(CELLS), "cells")
