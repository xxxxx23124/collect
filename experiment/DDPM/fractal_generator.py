import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback for environments without numba
    njit = None


Array = np.ndarray

IMAGE_NAME_RE = re.compile(r"^fractal_(\d+)\.png$")
FAMILY = "ifs_leaf"

LEAF_STYLES = {
    "classic_fern": {
        "category": "fern",
        "affines": [[0.0, 0.0, 0.0, 0.16, 0.0, 0.0], [0.85, 0.04, -0.04, 0.85, 0.0, 1.60], [0.20, -0.26, 0.23, 0.22, 0.0, 1.60], [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44]],
        "probs": [0.01, 0.78, 0.105, 0.105],
        "bounds": [-3.0, 3.0, 0.0, 10.5],
        "noise": [0.020, 0.028, 0.028, 0.020, 0.055, 0.090],
        "x_stretch": [0.86, 1.18],
        "y_step": [0.92, 1.08],
    },
    "soft_fern": {
        "category": "fern",
        "affines": [[0.0, 0.0, 0.0, 0.17, 0.0, 0.0], [0.84, 0.02, -0.02, 0.86, 0.0, 1.50], [0.18, -0.22, 0.20, 0.23, 0.0, 1.45], [-0.14, 0.24, 0.22, 0.22, 0.0, 0.55]],
        "probs": [0.012, 0.80, 0.095, 0.093],
        "bounds": [-2.8, 2.8, 0.0, 10.6],
        "noise": [0.018, 0.024, 0.024, 0.020, 0.050, 0.080],
        "x_stretch": [0.90, 1.12],
        "y_step": [0.94, 1.08],
    },
    "dense_fern": {
        "category": "fern",
        "affines": [[0.0, 0.0, 0.0, 0.13, 0.0, 0.0], [0.80, 0.05, -0.05, 0.84, 0.0, 1.30], [0.25, -0.34, 0.28, 0.22, 0.0, 1.35], [-0.22, 0.32, 0.28, 0.24, 0.0, 0.48]],
        "probs": [0.010, 0.70, 0.145, 0.145],
        "bounds": [-3.8, 3.8, 0.0, 9.8],
        "noise": [0.024, 0.032, 0.032, 0.024, 0.065, 0.090],
        "x_stretch": [0.94, 1.24],
        "y_step": [0.90, 1.05],
    },
    "sparse_fern": {
        "category": "fern",
        "affines": [[0.0, 0.0, 0.0, 0.18, 0.0, 0.0], [0.88, 0.03, -0.03, 0.87, 0.0, 1.70], [0.16, -0.30, 0.20, 0.20, 0.0, 1.75], [-0.12, 0.30, 0.20, 0.22, 0.0, 0.60]],
        "probs": [0.014, 0.84, 0.074, 0.072],
        "bounds": [-2.9, 2.9, 0.0, 12.0],
        "noise": [0.018, 0.030, 0.030, 0.020, 0.055, 0.105],
        "x_stretch": [0.84, 1.08],
        "y_step": [0.96, 1.12],
    },
    "wide_frond": {
        "category": "fern",
        "affines": [[0.0, 0.0, 0.0, 0.14, 0.0, 0.0], [0.80, 0.06, -0.04, 0.82, 0.0, 1.30], [0.28, -0.36, 0.32, 0.24, 0.0, 1.45], [-0.24, 0.34, 0.30, 0.26, 0.0, 0.55]],
        "probs": [0.012, 0.73, 0.13, 0.128],
        "bounds": [-4.0, 4.0, 0.0, 9.5],
        "noise": [0.026, 0.034, 0.034, 0.026, 0.075, 0.100],
        "x_stretch": [1.00, 1.30],
        "y_step": [0.88, 1.04],
    },
    "slender_leaf": {
        "category": "single_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.20, 0.0, 0.0], [0.90, 0.02, -0.02, 0.88, 0.0, 1.35], [0.12, -0.18, 0.18, 0.18, 0.0, 1.20], [-0.10, 0.18, 0.16, 0.20, 0.0, 0.70]],
        "probs": [0.015, 0.82, 0.085, 0.080],
        "bounds": [-2.2, 2.2, 0.0, 12.0],
        "noise": [0.014, 0.020, 0.020, 0.018, 0.045, 0.075],
        "x_stretch": [0.70, 0.98],
        "y_step": [0.98, 1.16],
    },
    "lanceolate_leaf": {
        "category": "single_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.19, 0.0, 0.0], [0.88, 0.01, -0.01, 0.89, 0.0, 1.22], [0.16, -0.16, 0.18, 0.20, 0.0, 1.10], [-0.14, 0.16, 0.18, 0.20, 0.0, 0.82]],
        "probs": [0.012, 0.81, 0.090, 0.088],
        "bounds": [-2.4, 2.4, 0.0, 11.6],
        "noise": [0.014, 0.018, 0.018, 0.018, 0.040, 0.070],
        "x_stretch": [0.78, 1.05],
        "y_step": [0.96, 1.12],
    },
    "oval_leaf": {
        "category": "single_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.16, 0.0, 0.0], [0.78, 0.00, 0.00, 0.78, 0.0, 1.18], [0.25, -0.18, 0.24, 0.24, 0.0, 1.08], [-0.25, 0.18, 0.24, 0.24, 0.0, 1.08]],
        "probs": [0.012, 0.70, 0.144, 0.144],
        "bounds": [-3.6, 3.6, 0.0, 8.8],
        "noise": [0.018, 0.018, 0.018, 0.020, 0.060, 0.070],
        "x_stretch": [1.04, 1.36],
        "y_step": [0.86, 1.00],
    },
    "round_leaf": {
        "category": "single_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.14, 0.0, 0.0], [0.74, 0.00, 0.00, 0.74, 0.0, 1.08], [0.28, -0.16, 0.25, 0.23, 0.0, 0.98], [-0.28, 0.16, 0.25, 0.23, 0.0, 0.98]],
        "probs": [0.012, 0.66, 0.164, 0.164],
        "bounds": [-3.9, 3.9, 0.0, 8.0],
        "noise": [0.018, 0.016, 0.016, 0.020, 0.070, 0.060],
        "x_stretch": [1.16, 1.48],
        "y_step": [0.78, 0.94],
    },
    "heart_leaf": {
        "category": "single_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.15, 0.0, 0.0], [0.76, -0.03, 0.03, 0.76, 0.0, 1.10], [0.30, -0.20, 0.26, 0.22, 0.0, 1.05], [-0.30, 0.20, 0.26, 0.22, 0.0, 1.05]],
        "probs": [0.012, 0.64, 0.174, 0.174],
        "bounds": [-4.0, 4.0, 0.0, 8.2],
        "noise": [0.018, 0.020, 0.020, 0.020, 0.075, 0.070],
        "x_stretch": [1.18, 1.52],
        "y_step": [0.78, 0.96],
    },
    "split_leaf": {
        "category": "split_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.15, 0.0, 0.0], [0.83, 0.00, 0.00, 0.84, 0.0, 1.42], [0.24, -0.31, 0.30, 0.20, 0.0, 1.34], [-0.24, 0.31, 0.30, 0.20, 0.0, 1.34]],
        "probs": [0.014, 0.74, 0.123, 0.123],
        "bounds": [-3.7, 3.7, 0.0, 10.2],
        "noise": [0.022, 0.026, 0.026, 0.020, 0.070, 0.085],
        "x_stretch": [0.96, 1.24],
        "y_step": [0.90, 1.08],
    },
    "maple_like": {
        "category": "split_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.13, 0.0, 0.0], [0.74, 0.00, 0.00, 0.78, 0.0, 1.18], [0.34, -0.42, 0.36, 0.20, 0.0, 1.20], [-0.34, 0.42, 0.36, 0.20, 0.0, 1.20]],
        "probs": [0.012, 0.56, 0.214, 0.214],
        "bounds": [-4.6, 4.6, 0.0, 8.8],
        "noise": [0.020, 0.034, 0.034, 0.024, 0.090, 0.075],
        "x_stretch": [1.15, 1.55],
        "y_step": [0.82, 1.00],
    },
    "palmate_leaf": {
        "category": "split_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.14, 0.0, 0.0], [0.72, 0.02, -0.02, 0.76, 0.0, 1.05], [0.36, -0.36, 0.32, 0.22, 0.0, 1.10], [-0.36, 0.36, 0.32, 0.22, 0.0, 1.10]],
        "probs": [0.012, 0.58, 0.204, 0.204],
        "bounds": [-4.7, 4.7, 0.0, 8.3],
        "noise": [0.020, 0.032, 0.032, 0.024, 0.090, 0.070],
        "x_stretch": [1.18, 1.56],
        "y_step": [0.78, 0.98],
    },
    "lobed_leaf": {
        "category": "split_leaf",
        "affines": [[0.0, 0.0, 0.0, 0.15, 0.0, 0.0], [0.78, 0.04, -0.03, 0.79, 0.0, 1.20], [0.30, -0.28, 0.27, 0.23, 0.0, 1.08], [-0.30, 0.28, 0.27, 0.23, 0.0, 0.82]],
        "probs": [0.012, 0.64, 0.176, 0.172],
        "bounds": [-4.2, 4.2, 0.0, 8.8],
        "noise": [0.020, 0.030, 0.030, 0.024, 0.085, 0.075],
        "x_stretch": [1.08, 1.42],
        "y_step": [0.82, 1.02],
    },
    "grass_blade": {
        "category": "grass",
        "affines": [[0.0, 0.0, 0.0, 0.22, 0.0, 0.0], [0.93, 0.03, -0.05, 0.91, 0.0, 1.10], [0.07, -0.10, 0.12, 0.16, 0.0, 0.95], [-0.06, 0.10, 0.10, 0.17, 0.0, 0.58]],
        "probs": [0.018, 0.88, 0.052, 0.050],
        "bounds": [-1.5, 1.5, 0.0, 13.2],
        "noise": [0.012, 0.018, 0.020, 0.018, 0.030, 0.065],
        "x_stretch": [0.42, 0.72],
        "y_step": [1.02, 1.18],
    },
    "reed_leaf": {
        "category": "grass",
        "affines": [[0.0, 0.0, 0.0, 0.21, 0.0, 0.0], [0.91, 0.06, -0.03, 0.90, 0.0, 1.22], [0.10, -0.12, 0.13, 0.17, 0.0, 1.00], [-0.08, 0.12, 0.12, 0.18, 0.0, 0.70]],
        "probs": [0.016, 0.86, 0.064, 0.060],
        "bounds": [-1.8, 1.8, 0.0, 13.0],
        "noise": [0.014, 0.022, 0.020, 0.018, 0.036, 0.070],
        "x_stretch": [0.50, 0.82],
        "y_step": [1.00, 1.16],
    },
    "curved_blade": {
        "category": "grass",
        "affines": [[0.0, 0.0, 0.0, 0.20, 0.0, 0.0], [0.90, 0.11, -0.08, 0.90, 0.0, 1.18], [0.09, -0.14, 0.13, 0.16, 0.0, 1.00], [-0.07, 0.13, 0.11, 0.17, 0.0, 0.64]],
        "probs": [0.016, 0.86, 0.064, 0.060],
        "bounds": [-2.1, 2.1, 0.0, 13.0],
        "noise": [0.014, 0.026, 0.026, 0.018, 0.042, 0.070],
        "x_stretch": [0.52, 0.86],
        "y_step": [1.00, 1.15],
    },
    "compound_leaf": {
        "category": "compound",
        "affines": [[0.0, 0.0, 0.0, 0.16, 0.0, 0.0], [0.84, 0.02, -0.02, 0.84, 0.0, 1.36], [0.18, -0.28, 0.24, 0.20, 0.0, 1.22], [-0.18, 0.28, 0.24, 0.20, 0.0, 0.86]],
        "probs": [0.012, 0.76, 0.114, 0.114],
        "bounds": [-3.3, 3.3, 0.0, 10.6],
        "noise": [0.020, 0.030, 0.030, 0.022, 0.065, 0.080],
        "x_stretch": [0.92, 1.20],
        "y_step": [0.92, 1.10],
    },
    "branching_leaf": {
        "category": "compound",
        "affines": [[0.0, 0.0, 0.0, 0.16, 0.0, 0.0], [0.82, 0.08, -0.06, 0.83, 0.0, 1.36], [0.20, -0.32, 0.26, 0.22, 0.0, 1.28], [-0.14, 0.34, 0.24, 0.22, 0.0, 0.70]],
        "probs": [0.012, 0.74, 0.132, 0.116],
        "bounds": [-3.7, 3.7, 0.0, 10.4],
        "noise": [0.022, 0.034, 0.034, 0.022, 0.075, 0.085],
        "x_stretch": [0.94, 1.28],
        "y_step": [0.90, 1.08],
    },
    "asymmetric_sprig": {
        "category": "compound",
        "affines": [[0.0, 0.0, 0.0, 0.17, 0.0, 0.0], [0.84, 0.07, -0.04, 0.84, 0.0, 1.42], [0.22, -0.30, 0.26, 0.20, 0.0, 1.30], [-0.10, 0.24, 0.20, 0.21, 0.0, 0.56]],
        "probs": [0.012, 0.78, 0.142, 0.066],
        "bounds": [-3.4, 3.4, 0.0, 10.8],
        "noise": [0.020, 0.034, 0.030, 0.022, 0.075, 0.085],
        "x_stretch": [0.92, 1.26],
        "y_step": [0.92, 1.10],
    },
    "ginkgo_like": {
        "category": "fan",
        "affines": [[0.0, 0.0, 0.0, 0.13, 0.0, 0.0], [0.68, 0.00, 0.00, 0.72, 0.0, 0.98], [0.38, -0.20, 0.24, 0.20, 0.0, 0.92], [-0.38, 0.20, 0.24, 0.20, 0.0, 0.92]],
        "probs": [0.012, 0.52, 0.234, 0.234],
        "bounds": [-5.0, 5.0, 0.0, 7.4],
        "noise": [0.018, 0.024, 0.024, 0.022, 0.095, 0.060],
        "x_stretch": [1.30, 1.75],
        "y_step": [0.72, 0.90],
    },
    "fan_leaf": {
        "category": "fan",
        "affines": [[0.0, 0.0, 0.0, 0.12, 0.0, 0.0], [0.66, 0.02, -0.02, 0.70, 0.0, 0.92], [0.40, -0.24, 0.25, 0.18, 0.0, 0.88], [-0.40, 0.24, 0.25, 0.18, 0.0, 0.88]],
        "probs": [0.012, 0.50, 0.244, 0.244],
        "bounds": [-5.2, 5.2, 0.0, 7.0],
        "noise": [0.018, 0.026, 0.026, 0.022, 0.100, 0.058],
        "x_stretch": [1.34, 1.82],
        "y_step": [0.70, 0.90],
    },
    "drooping_leaf": {
        "category": "special",
        "affines": [[0.0, 0.0, 0.0, 0.18, 0.0, 0.0], [0.86, -0.08, 0.06, 0.86, 0.0, 1.34], [0.16, -0.22, 0.20, 0.20, 0.0, 1.12], [-0.14, 0.20, 0.18, 0.21, 0.0, 0.68]],
        "probs": [0.014, 0.82, 0.086, 0.080],
        "bounds": [-2.8, 2.8, 0.0, 11.5],
        "noise": [0.018, 0.026, 0.026, 0.020, 0.060, 0.080],
        "x_stretch": [0.78, 1.08],
        "y_step": [0.96, 1.14],
    },
    "twisted_leaf": {
        "category": "special",
        "affines": [[0.0, 0.0, 0.0, 0.17, 0.0, 0.0], [0.84, 0.12, -0.10, 0.84, 0.0, 1.40], [0.18, -0.28, 0.26, 0.22, 0.0, 1.22], [-0.16, 0.30, 0.24, 0.22, 0.0, 0.66]],
        "probs": [0.012, 0.78, 0.108, 0.100],
        "bounds": [-3.3, 3.3, 0.0, 10.8],
        "noise": [0.020, 0.036, 0.036, 0.022, 0.070, 0.085],
        "x_stretch": [0.88, 1.20],
        "y_step": [0.92, 1.10],
    },
    "curved_stem": {
        "category": "special",
        "affines": [[0.0, 0.0, 0.0, 0.17, 0.0, 0.0], [0.82, 0.10, -0.08, 0.84, 0.0, 1.48], [0.18, -0.30, 0.24, 0.23, 0.0, 1.46], [-0.13, 0.32, 0.25, 0.24, 0.0, 0.52]],
        "probs": [0.012, 0.77, 0.11, 0.108],
        "bounds": [-3.4, 3.4, 0.0, 10.8],
        "noise": [0.020, 0.032, 0.032, 0.022, 0.060, 0.095],
        "x_stretch": [0.90, 1.22],
        "y_step": [0.90, 1.10],
    },
}


@dataclass(frozen=True)
class RenderTask:
    index: int
    function_id: int
    seed: int
    size: int
    params: Dict[str, object]


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_image_index(path: Path) -> Optional[int]:
    match = IMAGE_NAME_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _max_existing_image_index(image_dir: Path) -> int:
    indices = [_parse_image_index(path) for path in image_dir.glob("fractal_*.png")]
    return max((index for index in indices if index is not None), default=-1)


def _max_existing_function_id(metadata_path: Path) -> int:
    if not metadata_path.exists():
        return -1

    max_function_id = -1
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            try:
                value = json.loads(line).get("function_id")
            except json.JSONDecodeError:
                continue
            if isinstance(value, int):
                max_function_id = max(max_function_id, value)
    return max_function_id


def prepare_output(args: argparse.Namespace) -> Tuple[Path, Path, int, int]:
    image_dir = args.out / "images"
    metadata_path = args.out / "metadata.jsonl"

    if args.overwrite:
        for path in image_dir.glob("fractal_*.png"):
            path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    image_dir.mkdir(parents=True, exist_ok=True)
    start_index = _max_existing_image_index(image_dir) + 1
    start_function_id = _max_existing_function_id(metadata_path) + 1
    return image_dir, metadata_path, start_index, start_function_id


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.samples_per_function <= 0:
        raise ValueError("--samples-per-function must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.leaf_styles:
        raise ValueError("--leaf-styles must include at least one style")
    unknown_styles = sorted(set(args.leaf_styles) - set(LEAF_STYLES))
    if unknown_styles:
        raise ValueError(f"Unknown --leaf-styles: {', '.join(unknown_styles)}")


def warn_if_slow_path() -> None:
    if njit is None:
        print("Warning: numba is not available; leaf generation will be much slower.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate leaf-like IFS fractal PNG images for DDPM training."
    )
    parser.add_argument("--count", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--size", type=int, default=512, help="Square image size in pixels.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_fractals",
        help="Output dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument(
        "--samples-per-function",
        type=int,
        default=3,
        help="How many random crops/colors to sample from one generated leaf function.",
    )
    parser.add_argument(
        "--leaf-styles",
        nargs="*",
        choices=sorted(LEAF_STYLES),
        default=sorted(LEAF_STYLES),
        help="Leaf style presets to sample. Defaults to all styles.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. Use 1 for easiest debugging.",
    )
    parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write metadata.jsonl next to the image folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing generated images and metadata before writing new samples.",
    )
    return parser.parse_args()


def make_tasks(
    args: argparse.Namespace,
    start_index: int = 0,
    start_function_id: int = 0,
) -> List[RenderTask]:
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    tasks: List[RenderTask] = []

    samples_per_function = max(1, args.samples_per_function)
    function_count = math.ceil(args.count / samples_per_function)
    leaf_styles = list(args.leaf_styles)
    for function_id in range(function_count):
        leaf_params = random_ifs_leaf_params(rng, leaf_styles)
        for _ in range(samples_per_function):
            if len(tasks) >= args.count:
                break
            params = dict(leaf_params)
            params["palette"] = random_leaf_palette(rng)
            tasks.append(
                RenderTask(
                    index=start_index + len(tasks),
                    function_id=start_function_id + function_id,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    size=args.size,
                    params=params,
                )
            )
    return tasks


def random_ifs_leaf_params(rng: np.random.Generator, leaf_styles: List[str]) -> Dict[str, object]:
    style = str(rng.choice(leaf_styles))
    config = LEAF_STYLES[style]
    affines = np.asarray(config["affines"], dtype=np.float64)
    noise = np.asarray(config["noise"], dtype=np.float64)
    affines = affines + rng.normal(0.0, noise, size=affines.shape)

    # Keep the stem transform stable while allowing fronds to vary more freely.
    affines[0, [0, 1, 2, 4]] = 0.0
    affines[0, 3] = np.clip(affines[0, 3], 0.10, 0.24)

    x_stretch = float(rng.uniform(*config.get("x_stretch", [0.86, 1.18])))
    y_step = float(rng.uniform(*config.get("y_step", [0.92, 1.08])))
    affines[:, [0, 1, 4]] *= x_stretch
    affines[:, 5] *= y_step

    if rng.random() < 0.5:
        affines[:, [0, 1, 4]] *= -1.0

    affines = limit_affine_contraction(affines)

    probs = np.asarray(config["probs"], dtype=np.float64)
    probs = np.clip(probs + rng.normal(0.0, 0.018, size=probs.shape), 0.004, None)
    probs = probs / probs.sum()

    bounds = np.asarray(config["bounds"], dtype=np.float64)
    bounds[[0, 1]] *= max(1.0, x_stretch) * 1.08
    bounds[3] *= max(1.0, y_step) * 1.04

    return {
        "leaf_style": style,
        "leaf_category": config["category"],
        "affines": affines.tolist(),
        "probs": probs.tolist(),
        "bounds": bounds.tolist(),
        "points": int(rng.integers(520_000, 1_050_000)),
        "burn_in": 50,
        "blur_passes": int(rng.integers(1, 3)),
    }


def limit_affine_contraction(affines: Array, max_radius: float = 0.98) -> Array:
    affines = affines.copy()
    for index in range(affines.shape[0]):
        matrix = affines[index, :4].reshape((2, 2))
        radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
        if radius > max_radius:
            affines[index, :4] *= max_radius / radius
    return affines


def random_leaf_palette(rng: np.random.Generator) -> Dict[str, object]:
    palettes = {
        "deep_green": {
            "background": [0.015, 0.020, 0.014],
            "shadow": [0.025, 0.105, 0.045],
            "mid": [0.120, 0.430, 0.160],
            "highlight": [0.720, 0.900, 0.340],
            "accent": [0.950, 0.720, 0.280],
        },
        "moss_mint": {
            "background": [0.025, 0.040, 0.035],
            "shadow": [0.045, 0.160, 0.120],
            "mid": [0.210, 0.580, 0.390],
            "highlight": [0.760, 0.970, 0.700],
            "accent": [0.520, 0.900, 0.820],
        },
        "blue_green": {
            "background": [0.018, 0.024, 0.040],
            "shadow": [0.025, 0.120, 0.160],
            "mid": [0.080, 0.430, 0.460],
            "highlight": [0.500, 0.900, 0.760],
            "accent": [0.340, 0.700, 0.980],
        },
        "autumn_leaf": {
            "background": [0.035, 0.024, 0.018],
            "shadow": [0.160, 0.065, 0.025],
            "mid": [0.660, 0.260, 0.070],
            "highlight": [0.980, 0.770, 0.230],
            "accent": [0.900, 0.260, 0.180],
        },
        "soft_illustration": {
            "background": [0.840, 0.820, 0.760],
            "shadow": [0.170, 0.270, 0.140],
            "mid": [0.360, 0.610, 0.250],
            "highlight": [0.900, 0.890, 0.520],
            "accent": [0.700, 0.380, 0.580],
        },
    }

    name = str(rng.choice(list(palettes)))
    palette = dict(palettes[name])
    for key in ("background", "shadow", "mid", "highlight", "accent"):
        color = np.asarray(palette[key], dtype=np.float64)
        color += rng.normal(0.0, 0.025, size=3)
        palette[key] = np.clip(color, 0.0, 1.0).tolist()

    palette.update(
        {
            "name": name,
            "contrast": float(rng.uniform(0.90, 1.30)),
            "gamma": float(rng.uniform(0.72, 1.18)),
            "vein_strength": float(rng.uniform(0.18, 0.48)),
            "grain": float(rng.uniform(0.0, 0.018)),
        }
    )
    return palette


def random_leaf_viewport(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "center_x": float(rng.uniform(-0.08, 0.08)),
        "center_y": float(rng.uniform(-0.08, 0.10)),
        "scale": float(10.0 ** rng.uniform(-0.10, 0.10)),
        "rotation": float(rng.uniform(0.0, 2.0 * math.pi)),
    }


def render_task(task: RenderTask) -> Tuple[int, Array, Dict[str, object]]:
    rng = np.random.default_rng(task.seed)
    viewport = random_leaf_viewport(rng)
    features = render_ifs_leaf(task.size, task.params, viewport, task.seed)
    image = colorize_leaf(features, dict(task.params["palette"]), rng)
    metadata = {
        "index": task.index,
        "filename": f"fractal_{task.index:06d}.png",
        "seed": task.seed,
        "function_id": task.function_id,
        "family": FAMILY,
        "leaf_style": task.params["leaf_style"],
        "leaf_category": task.params["leaf_category"],
        "palette": task.params["palette"],
        "viewport": viewport,
        "params": {key: value for key, value in task.params.items() if key != "palette"},
    }
    return task.index, image, metadata


def render_ifs_leaf(
    size: int,
    params: Dict[str, object],
    viewport: Dict[str, float],
    seed: int,
) -> Dict[str, Array]:
    affines = np.asarray(params["affines"], dtype=np.float64)
    probs = np.asarray(params["probs"], dtype=np.float64)
    probs = probs / probs.sum()
    cumulative = np.cumsum(probs)
    cumulative[-1] = 1.0
    bounds = np.asarray(params["bounds"], dtype=np.float64)

    hist, angle_sin, angle_cos = _ifs_histogram(
        size,
        affines,
        cumulative,
        int(params["points"]),
        int(params["burn_in"]),
        seed,
        float(viewport["center_x"]),
        float(viewport["center_y"]),
        float(viewport["scale"]),
        float(viewport["rotation"]),
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
    )

    hist = hist.astype(np.float64)
    angle_sin = angle_sin.astype(np.float64)
    angle_cos = angle_cos.astype(np.float64)
    for _ in range(int(params.get("blur_passes", 1))):
        hist = cheap_blur(hist)
        angle_sin = cheap_blur(angle_sin)
        angle_cos = cheap_blur(angle_cos)

    value = normalize01(np.log1p(hist))
    density = normalize01(hist)
    angle = np.where(hist > 0, np.arctan2(angle_sin, angle_cos), 0.0)
    detail = normalize01(edge_detail(value) + value * 0.36)
    return {"value": value, "density": density, "angle": angle, "detail": detail}


if njit is not None:

    @njit
    def _lcg_next(state):
        state = (state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407))
        value = ((state >> np.uint64(11)) & np.uint64(9007199254740991)) / 9007199254740992.0
        return state, value


    @njit
    def _choose(cumulative, r):
        for i in range(cumulative.shape[0]):
            if r <= cumulative[i]:
                return i
        return cumulative.shape[0] - 1


    @njit
    def _ifs_histogram(
        size,
        affines,
        cumulative,
        points,
        burn_in,
        seed,
        center_x,
        center_y,
        scale,
        rotation,
        min_x,
        max_x,
        min_y,
        max_y,
    ):
        hist = np.zeros((size, size), dtype=np.float64)
        angle_sin = np.zeros((size, size), dtype=np.float64)
        angle_cos = np.zeros((size, size), dtype=np.float64)
        state = np.uint64(seed + 1)
        x = 0.0
        y = 0.0
        mid_x = (min_x + max_x) * 0.5
        half_x = (max_x - min_x) * 0.5
        mid_y = (min_y + max_y) * 0.5
        half_y = (max_y - min_y) * 0.5
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        for i in range(points + burn_in):
            state, r = _lcg_next(state)
            k = _choose(cumulative, r)
            a, b, c, d, e, f = affines[k]
            xn = a * x + b * y + e
            yn = c * x + d * y + f
            x = xn
            y = yn
            if i >= burn_in:
                wx = (x - mid_x) / half_x
                wy = (y - mid_y) / half_y
                dx = (wx - center_x) / scale
                dy = (wy - center_y) / scale
                vx = dx * cos_t + dy * sin_t
                vy = -dx * sin_t + dy * cos_t
                px = int((vx + 1.0) * 0.5 * (size - 1))
                py = int((1.0 - vy) * 0.5 * (size - 1))
                if 0 <= px < size and 0 <= py < size:
                    hist[py, px] += 1.0
                    theta = math.atan2(vy, vx + 1e-12)
                    angle_sin[py, px] += math.sin(theta)
                    angle_cos[py, px] += math.cos(theta)
        return hist, angle_sin, angle_cos

else:

    def _ifs_histogram(
        size,
        affines,
        cumulative,
        points,
        burn_in,
        seed,
        center_x,
        center_y,
        scale,
        rotation,
        min_x,
        max_x,
        min_y,
        max_y,
    ):
        rng = np.random.default_rng(seed)
        hist = np.zeros((size, size), dtype=np.float64)
        angle_sin = np.zeros((size, size), dtype=np.float64)
        angle_cos = np.zeros((size, size), dtype=np.float64)
        x = 0.0
        y = 0.0
        mid_x = (min_x + max_x) * 0.5
        half_x = (max_x - min_x) * 0.5
        mid_y = (min_y + max_y) * 0.5
        half_y = (max_y - min_y) * 0.5
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        for i in range(points + burn_in):
            k = int(np.searchsorted(cumulative, rng.random()))
            k = min(k, cumulative.shape[0] - 1)
            a, b, c, d, e, f = affines[k]
            x, y = a * x + b * y + e, c * x + d * y + f
            if i >= burn_in:
                wx = (x - mid_x) / half_x
                wy = (y - mid_y) / half_y
                dx = (wx - center_x) / scale
                dy = (wy - center_y) / scale
                vx = dx * cos_t + dy * sin_t
                vy = -dx * sin_t + dy * cos_t
                px = int((vx + 1.0) * 0.5 * (size - 1))
                py = int((1.0 - vy) * 0.5 * (size - 1))
                if 0 <= px < size and 0 <= py < size:
                    hist[py, px] += 1.0
                    theta = math.atan2(vy, vx + 1e-12)
                    angle_sin[py, px] += math.sin(theta)
                    angle_cos[py, px] += math.cos(theta)
        return hist, angle_sin, angle_cos


def normalize01(values: Array) -> Array:
    values = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(values, 1.0)
    hi = np.percentile(values, 99.4)
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def edge_detail(values: Array) -> Array:
    gx = np.zeros_like(values, dtype=np.float64)
    gy = np.zeros_like(values, dtype=np.float64)
    gx[:, 1:-1] = values[:, 2:] - values[:, :-2]
    gy[1:-1, :] = values[2:, :] - values[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def cheap_blur(values: Array) -> Array:
    padded = np.pad(values, 1, mode="edge")
    return (
        padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 5.0


def colorize_leaf(features: Dict[str, Array], palette: Dict[str, object], rng: np.random.Generator) -> Array:
    value = normalize01(features["value"])
    density = normalize01(features["density"])
    detail = normalize01(features["detail"])
    angle = np.mod((features["angle"] + math.pi) / (2.0 * math.pi), 1.0)

    contrast = float(palette["contrast"])
    gamma = float(palette["gamma"])
    tone = np.clip((value - 0.5) * contrast + 0.5, 0.0, 1.0)
    tone = np.power(tone, gamma)
    tone = np.clip(tone * (0.78 + 0.30 * density), 0.0, 1.0)

    background = color_array(palette["background"])
    shadow = color_array(palette["shadow"])
    mid = color_array(palette["mid"])
    highlight = color_array(palette["highlight"])
    accent = color_array(palette["accent"])

    low_mix = smoothstep(tone * 2.0)
    high_mix = smoothstep(tone * 2.0 - 1.0)
    body = lerp(shadow, mid, low_mix)
    body = lerp(body, highlight, high_mix)

    vein = np.clip(detail * density * float(palette["vein_strength"]), 0.0, 1.0)
    angle_wash = 0.10 * np.sin(2.0 * math.pi * (angle + 0.15)) * density
    body = np.clip(body + accent * vein[..., None] + angle_wash[..., None], 0.0, 1.0)

    alpha = smoothstep(np.clip(density * 1.35 + value * 0.25, 0.0, 1.0))
    rgb = lerp(background, body, alpha)

    grain = float(palette["grain"])
    if grain > 0.0:
        rgb = rgb + rng.normal(0.0, grain, size=rgb.shape)

    rgb = subtle_vignette(np.clip(rgb, 0.0, 1.0))
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def color_array(color: object) -> Array:
    return np.asarray(color, dtype=np.float64).reshape((1, 1, 3))


def lerp(a: Array, b: Array, t: Array) -> Array:
    return a * (1.0 - t[..., None]) + b * t[..., None]


def smoothstep(x: Array) -> Array:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def subtle_vignette(rgb: Array) -> Array:
    size = rgb.shape[0]
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x * x + y * y)
    vignette = np.clip(1.07 - 0.22 * r * r, 0.76, 1.07)
    return rgb * vignette[..., None]


def save_result(result: Tuple[int, Array, Dict[str, object]], image_dir: Path) -> Dict[str, object]:
    _, image, metadata = result
    path = image_dir / metadata["filename"]
    Image.fromarray(image).save(path, format="PNG", optimize=True)
    return metadata


def iter_results(tasks: List[RenderTask], workers: int) -> Iterable[Tuple[int, Array, Dict[str, object]]]:
    if workers <= 1:
        for task in tasks:
            yield render_task(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(render_task, tasks)


def main() -> None:
    args = parse_args()
    validate_args(args)
    warn_if_slow_path()

    image_dir, metadata_path, start_index, start_function_id = prepare_output(args)
    tasks = make_tasks(args, start_index=start_index, start_function_id=start_function_id)

    metadata_file = metadata_path.open("a", encoding="utf-8") if args.metadata else None
    try:
        progress = tqdm(iter_results(tasks, args.workers), total=len(tasks), desc="Generating leaf fractals")
        for result in progress:
            metadata = save_result(result, image_dir)
            if metadata_file is not None:
                metadata_file.write(json.dumps(metadata, ensure_ascii=False, default=_json_default) + "\n")
                metadata_file.flush()
    finally:
        if metadata_file is not None:
            metadata_file.close()

    print(f"Generated {len(tasks)} leaf images in: {image_dir}")
    if args.metadata:
        print(f"Metadata appended to: {metadata_path}")


if __name__ == "__main__":
    # Keeps Windows multiprocessing safe when --workers > 1.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache"))
    main()