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


FAMILIES = (
    "mandelbrot",
    "burning_ship",
    "tricorn",
    "julia",
    "newton",
    "ifs_leaf",
    "flame",
)


@dataclass(frozen=True)
class RenderTask:
    index: int
    function_id: int
    seed: int
    size: int
    family: str
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
    if not args.families:
        raise ValueError("--families must include at least one family")


def warn_if_slow_path(args: argparse.Namespace) -> None:
    point_families = {"ifs_leaf", "flame"}
    if njit is None and point_families.intersection(args.families):
        print("Warning: numba is not available; ifs_leaf/flame generation will be much slower.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diverse 512x512 fractal PNG images for DDPM training."
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
        help="How many random viewports to sample from one generated function.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="Fractal families to use. Defaults to all families.",
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
    for function_id in range(function_count):
        family = str(rng.choice(args.families))
        params = random_family_params(rng, family)
        for _ in range(samples_per_function):
            if len(tasks) >= args.count:
                break
            tasks.append(
                RenderTask(
                    index=start_index + len(tasks),
                    function_id=start_function_id + function_id,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    size=args.size,
                    family=family,
                    params=params,
                )
            )
    return tasks


def random_family_params(rng: np.random.Generator, family: str) -> Dict[str, object]:
    palette = {
        "hue": float(rng.random()),
        "hue_scale": float(rng.uniform(0.35, 1.6)),
        "angle_mix": float(rng.uniform(-0.35, 0.35)),
        "sat": float(rng.uniform(0.58, 0.95)),
        "val": float(rng.uniform(0.75, 1.08)),
        "contrast": float(rng.uniform(0.85, 1.35)),
        "gamma": float(rng.uniform(0.75, 1.35)),
        "background": float(rng.uniform(0.0, 0.08)),
    }

    params: Dict[str, object] = {"palette": palette}
    if family in {"mandelbrot", "burning_ship", "tricorn"}:
        params.update(
            {
                "max_iter": int(rng.integers(120, 420)),
                "power": int(rng.choice([2, 2, 2, 3, 4])),
                "bailout": float(rng.uniform(6.0, 24.0)),
                "jitter": float(rng.uniform(0.0, 0.035)),
                "trap": [float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8))],
            }
        )
    elif family == "julia":
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(0.45, 0.86)
        params.update(
            {
                "c": [float(radius * math.cos(angle)), float(radius * math.sin(angle))],
                "max_iter": int(rng.integers(120, 440)),
                "power": int(rng.choice([2, 2, 2, 3, 4])),
                "bailout": float(rng.uniform(6.0, 24.0)),
                "trap": [float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.6, 0.6))],
            }
        )
    elif family == "newton":
        degree = int(rng.integers(3, 7))
        angles = np.linspace(0.0, 2.0 * math.pi, degree, endpoint=False)
        angles += rng.uniform(-0.35, 0.35, size=degree)
        radii = rng.uniform(0.65, 1.3, size=degree)
        roots = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        params.update(
            {
                "roots": roots.tolist(),
                "max_iter": int(rng.integers(36, 90)),
                "relaxation": float(rng.uniform(0.78, 1.18)),
            }
        )
    elif family == "ifs_leaf":
        params.update(random_ifs_leaf_params(rng))
    elif family == "flame":
        params.update(random_flame_params(rng))
    else:
        raise ValueError(f"Unknown family: {family}")
    return params


def random_ifs_leaf_params(rng: np.random.Generator) -> Dict[str, object]:
    # Barnsley-like presets are perturbed so the generator keeps a leaf/plant bias
    # while still producing many silhouettes.
    base = np.array(
        [
            [0.0, 0.0, 0.0, 0.16, 0.0, 0.0],
            [0.85, 0.04, -0.04, 0.85, 0.0, 1.6],
            [0.2, -0.26, 0.23, 0.22, 0.0, 1.6],
            [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44],
        ],
        dtype=np.float64,
    )
    noise = rng.normal(0.0, [0.025, 0.03, 0.03, 0.025, 0.08, 0.12], size=base.shape)
    affines = base + noise
    probs = np.array([0.01, 0.78, 0.105, 0.105], dtype=np.float64)
    probs = probs + rng.normal(0.0, 0.025, size=4)
    probs = np.clip(probs, 0.005, None)
    probs = probs / probs.sum()
    return {
        "affines": affines.tolist(),
        "probs": probs.tolist(),
        "points": int(rng.integers(420_000, 900_000)),
        "burn_in": 40,
        "blur_passes": int(rng.integers(1, 3)),
    }


def random_flame_params(rng: np.random.Generator) -> Dict[str, object]:
    transform_count = int(rng.integers(3, 7))
    affines = []
    for _ in range(transform_count):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        scale = rng.uniform(0.25, 0.82)
        shear = rng.uniform(-0.28, 0.28)
        cos_a = math.cos(angle) * scale
        sin_a = math.sin(angle) * scale
        affines.append(
            [
                cos_a + shear,
                -sin_a,
                sin_a,
                cos_a - shear,
                rng.uniform(-0.8, 0.8),
                rng.uniform(-0.8, 0.8),
            ]
        )
    variation_weights = rng.dirichlet(np.ones(7) * rng.uniform(0.25, 1.5))
    probs = rng.dirichlet(np.ones(transform_count))
    return {
        "affines": affines,
        "probs": probs.tolist(),
        "variation_weights": variation_weights.tolist(),
        "points": int(rng.integers(650_000, 1_350_000)),
        "burn_in": 80,
        "blur_passes": int(rng.integers(1, 3)),
    }


def random_viewport(rng: np.random.Generator, family: str) -> Dict[str, float]:
    if family == "burning_ship":
        center_x = rng.uniform(-1.85, -1.35)
        center_y = rng.uniform(-0.18, 0.12)
        scale = 10.0 ** rng.uniform(-0.45, 0.25)
    elif family == "mandelbrot":
        center_x = rng.uniform(-0.9, 0.35)
        center_y = rng.uniform(-0.65, 0.65)
        scale = 10.0 ** rng.uniform(-0.62, 0.23)
    elif family == "tricorn":
        center_x = rng.uniform(-0.45, 0.45)
        center_y = rng.uniform(-0.75, 0.75)
        scale = 10.0 ** rng.uniform(-0.55, 0.22)
    elif family == "newton":
        center_x = rng.uniform(-0.2, 0.2)
        center_y = rng.uniform(-0.2, 0.2)
        scale = 10.0 ** rng.uniform(-0.18, 0.22)
    else:
        center_x = rng.uniform(-0.35, 0.35)
        center_y = rng.uniform(-0.35, 0.35)
        scale = 10.0 ** rng.uniform(-0.25, 0.25)
    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "scale": float(scale),
        "rotation": float(rng.uniform(0.0, 2.0 * math.pi)),
    }


def render_task(task: RenderTask) -> Tuple[int, Array, Dict[str, object]]:
    rng = np.random.default_rng(task.seed)
    viewport = random_viewport(rng, task.family)
    palette = dict(task.params["palette"])

    if task.family in {"mandelbrot", "burning_ship", "tricorn", "julia"}:
        features = render_escape_time(task.family, task.size, task.params, viewport, rng)
    elif task.family == "newton":
        features = render_newton(task.size, task.params, viewport, rng)
    elif task.family == "ifs_leaf":
        features = render_point_fractal(task.size, task.params, viewport, task.seed, leaf_mode=True)
    elif task.family == "flame":
        features = render_point_fractal(task.size, task.params, viewport, task.seed, leaf_mode=False)
    else:
        raise ValueError(f"Unknown family: {task.family}")

    image = unified_color(features, palette, rng)
    metadata = {
        "index": task.index,
        "filename": f"fractal_{task.index:06d}.png",
        "seed": task.seed,
        "function_id": task.function_id,
        "family": task.family,
        "viewport": viewport,
        "params": task.params,
    }
    return task.index, image, metadata


def complex_grid(size: int, viewport: Dict[str, float]) -> Array:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    scale = float(viewport["scale"])
    theta = float(viewport["rotation"])
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xr = (x * cos_t - y * sin_t) * scale + float(viewport["center_x"])
    yr = (x * sin_t + y * cos_t) * scale + float(viewport["center_y"])
    return xr + 1j * yr


def render_escape_time(
    family: str,
    size: int,
    params: Dict[str, object],
    viewport: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, Array]:
    grid = complex_grid(size, viewport)
    max_iter = int(params["max_iter"])
    power = int(params["power"])
    bailout = float(params["bailout"])
    trap_c = complex(*params["trap"])

    if family == "julia":
        z = grid.copy()
        c = complex(*params["c"])
        c_grid = np.full_like(grid, c)
    else:
        z = np.zeros_like(grid)
        c_grid = grid

    escape_iter = np.full(grid.shape, max_iter, dtype=np.float64)
    trap = np.full(grid.shape, 1e9, dtype=np.float64)
    final_abs = np.zeros(grid.shape, dtype=np.float64)

    for i in range(max_iter):
        active = escape_iter == max_iter
        if not np.any(active):
            break
        za = z[active]
        ca = c_grid[active]
        if family == "burning_ship":
            za = np.abs(za.real) + 1j * np.abs(za.imag)
            zn = za**power + ca
        elif family == "tricorn":
            zn = np.conj(za) ** power + ca
        else:
            zn = za**power + ca
        z[active] = zn
        trap[active] = np.minimum(trap[active], np.abs(zn - trap_c))
        escaped_now = active & (np.abs(z) > bailout)
        escape_iter[escaped_now] = i + 1
        final_abs[escaped_now] = np.abs(z[escaped_now])

    active = escape_iter == max_iter
    final_abs[active] = np.abs(z[active])
    with np.errstate(divide="ignore", invalid="ignore"):
        smooth = escape_iter + 1.0 - np.log(np.log(np.maximum(final_abs, 1.000001))) / math.log(max(power, 2))
    escaped = escape_iter < max_iter
    value = np.where(escaped, smooth / max_iter, 0.0)
    value = normalize01(value)
    trap_value = np.exp(-3.0 * np.clip(trap, 0.0, 4.0))
    detail = normalize01(trap_value + edge_detail(value) * 0.65)
    density = normalize01(np.where(escaped, 1.0 - escape_iter / max_iter, trap_value))
    angle = np.angle(z)

    if float(params.get("jitter", 0.0)) > 0.0:
        detail = normalize01(detail + rng.normal(0.0, float(params["jitter"]), size=detail.shape))

    return {"value": value, "density": density, "angle": angle, "detail": detail}


def render_newton(
    size: int,
    params: Dict[str, object],
    viewport: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, Array]:
    z = complex_grid(size, viewport)
    roots_xy = np.asarray(params["roots"], dtype=np.float64)
    roots = roots_xy[:, 0] + 1j * roots_xy[:, 1]
    max_iter = int(params["max_iter"])
    relaxation = float(params["relaxation"])
    conv_iter = np.full(z.shape, max_iter, dtype=np.float64)

    for i in range(max_iter):
        active = conv_iter == max_iter
        if not np.any(active):
            break
        za = z[active]
        p = np.ones_like(za)
        for root in roots:
            p *= za - root
        dp = np.zeros_like(za)
        for skip in range(len(roots)):
            term = np.ones_like(za)
            for j, root in enumerate(roots):
                if j != skip:
                    term *= za - root
            dp += term
        step = np.divide(p, dp, out=np.zeros_like(p), where=np.abs(dp) > 1e-12)
        za = za - relaxation * step
        z[active] = za
        min_dist = np.min(np.abs(za[:, None] - roots[None, :]), axis=1)
        conv_iter[active] = np.where(min_dist < 1e-5, i + 1, conv_iter[active])

    distances = np.stack([np.abs(z - root) for root in roots], axis=0)
    root_id = np.argmin(distances, axis=0)
    min_dist = np.min(distances, axis=0)
    value = normalize01(root_id / max(1, len(roots) - 1) + 0.35 * (conv_iter / max_iter))
    density = normalize01(1.0 - conv_iter / max_iter + np.exp(-15.0 * min_dist))
    angle = np.angle(z)
    detail = normalize01(edge_detail(root_id.astype(np.float64)) + edge_detail(conv_iter))
    detail = normalize01(detail + rng.normal(0.0, 0.015, size=detail.shape))
    return {"value": value, "density": density, "angle": angle, "detail": detail}


def render_point_fractal(
    size: int,
    params: Dict[str, object],
    viewport: Dict[str, float],
    seed: int,
    leaf_mode: bool,
) -> Dict[str, Array]:
    affines = np.asarray(params["affines"], dtype=np.float64)
    probs = np.asarray(params["probs"], dtype=np.float64)
    probs = probs / probs.sum()
    cumulative = np.cumsum(probs)
    points = int(params["points"])
    burn_in = int(params["burn_in"])
    center_x = float(viewport["center_x"])
    center_y = float(viewport["center_y"])
    scale = float(viewport["scale"])
    rotation = float(viewport["rotation"])

    if leaf_mode:
        hist, angle_acc = _ifs_histogram(size, affines, cumulative, points, burn_in, seed, center_x, center_y, scale, rotation)
    else:
        weights = np.asarray(params["variation_weights"], dtype=np.float64)
        hist, angle_acc = _flame_histogram(
            size, affines, cumulative, weights, points, burn_in, seed, center_x, center_y, scale, rotation
        )

    hist = hist.astype(np.float64)
    for _ in range(int(params.get("blur_passes", 1))):
        hist = cheap_blur(hist)
        angle_acc = cheap_blur(angle_acc)

    value = normalize01(np.log1p(hist))
    density = normalize01(hist)
    angle = np.where(hist > 0, angle_acc / np.maximum(hist, 1.0), 0.0)
    detail = normalize01(edge_detail(value) + value * 0.45)

    return {"value": value, "density": density, "angle": angle, "detail": detail}


if njit is not None:

    @njit(cache=True)
    def _lcg_next(state):
        state = (state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407))
        value = ((state >> np.uint64(11)) & np.uint64(9007199254740991)) / 9007199254740992.0
        return state, value


    @njit(cache=True)
    def _choose(cumulative, r):
        for i in range(cumulative.shape[0]):
            if r <= cumulative[i]:
                return i
        return cumulative.shape[0] - 1


    @njit(cache=True)
    def _ifs_histogram(size, affines, cumulative, points, burn_in, seed, center_x, center_y, scale, rotation):
        hist = np.zeros((size, size), dtype=np.float64)
        angle_acc = np.zeros((size, size), dtype=np.float64)
        state = np.uint64(seed + 1)
        x = 0.0
        y = 0.0
        min_x = -3.0
        max_x = 3.0
        min_y = 0.0
        max_y = 10.5
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
                    angle_acc[py, px] += math.atan2(vy, vx + 1e-12)
        return hist, angle_acc


    @njit(cache=True)
    def _flame_histogram(size, affines, cumulative, weights, points, burn_in, seed, center_x, center_y, scale, rotation):
        hist = np.zeros((size, size), dtype=np.float64)
        angle_acc = np.zeros((size, size), dtype=np.float64)
        state = np.uint64(seed + 11)
        x = 0.0
        y = 0.0
        min_v = -2.6
        max_v = 2.6
        mid_v = (min_v + max_v) * 0.5
        half_v = (max_v - min_v) * 0.5
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        for i in range(points + burn_in):
            state, r = _lcg_next(state)
            k = _choose(cumulative, r)
            a, b, c, d, e, f = affines[k]
            u = a * x + b * y + e
            v = c * x + d * y + f
            r2 = u * u + v * v + 1e-9
            radius = math.sqrt(r2)
            theta = math.atan2(v, u)

            nx = weights[0] * u
            ny = weights[0] * v
            nx += weights[1] * math.sin(u)
            ny += weights[1] * math.sin(v)
            nx += weights[2] * (u * math.sin(r2) - v * math.cos(r2))
            ny += weights[2] * (u * math.cos(r2) + v * math.sin(r2))
            nx += weights[3] * (u / r2)
            ny += weights[3] * (v / r2)
            nx += weights[4] * ((u - v) * (u + v) / radius)
            ny += weights[4] * (2.0 * u * v / radius)
            nx += weights[5] * (theta / math.pi)
            ny += weights[5] * (radius - 1.0)
            nx += weights[6] * (radius * math.sin(theta + radius))
            ny += weights[6] * (radius * math.cos(theta - radius))
            x = nx
            y = ny
            if i >= burn_in:
                wx = (x - mid_v) / half_v
                wy = (y - mid_v) / half_v
                dx = (wx - center_x) / scale
                dy = (wy - center_y) / scale
                vx = dx * cos_t + dy * sin_t
                vy = -dx * sin_t + dy * cos_t
                px = int((vx + 1.0) * 0.5 * (size - 1))
                py = int((1.0 - vy) * 0.5 * (size - 1))
                if 0 <= px < size and 0 <= py < size:
                    hist[py, px] += 1.0
                    angle_acc[py, px] += math.atan2(vy, vx + 1e-12)
        return hist, angle_acc

else:

    def _ifs_histogram(size, affines, cumulative, points, burn_in, seed, center_x, center_y, scale, rotation):
        rng = np.random.default_rng(seed)
        hist = np.zeros((size, size), dtype=np.float64)
        angle_acc = np.zeros((size, size), dtype=np.float64)
        x = 0.0
        y = 0.0
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        for i in range(points + burn_in):
            k = int(np.searchsorted(cumulative, rng.random()))
            a, b, c, d, e, f = affines[k]
            x, y = a * x + b * y + e, c * x + d * y + f
            if i >= burn_in:
                wx = x / 3.0
                wy = (y - 5.25) / 5.25
                dx = (wx - center_x) / scale
                dy = (wy - center_y) / scale
                vx = dx * cos_t + dy * sin_t
                vy = -dx * sin_t + dy * cos_t
                px = int((vx + 1.0) * 0.5 * (size - 1))
                py = int((1.0 - vy) * 0.5 * (size - 1))
                if 0 <= px < size and 0 <= py < size:
                    hist[py, px] += 1.0
                    angle_acc[py, px] += math.atan2(vy, vx + 1e-12)
        return hist, angle_acc


    def _flame_histogram(size, affines, cumulative, weights, points, burn_in, seed, center_x, center_y, scale, rotation):
        rng = np.random.default_rng(seed)
        hist = np.zeros((size, size), dtype=np.float64)
        angle_acc = np.zeros((size, size), dtype=np.float64)
        x = 0.0
        y = 0.0
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        for i in range(points + burn_in):
            k = int(np.searchsorted(cumulative, rng.random()))
            a, b, c, d, e, f = affines[k]
            u, v = a * x + b * y + e, c * x + d * y + f
            r2 = u * u + v * v + 1e-9
            radius = math.sqrt(r2)
            theta = math.atan2(v, u)
            x = weights[0] * u + weights[1] * math.sin(u) + weights[2] * (u * math.sin(r2) - v * math.cos(r2))
            y = weights[0] * v + weights[1] * math.sin(v) + weights[2] * (u * math.cos(r2) + v * math.sin(r2))
            x += weights[3] * (u / r2) + weights[4] * ((u - v) * (u + v) / radius) + weights[5] * (theta / math.pi)
            y += weights[3] * (v / r2) + weights[4] * (2.0 * u * v / radius) + weights[5] * (radius - 1.0)
            x += weights[6] * (radius * math.sin(theta + radius))
            y += weights[6] * (radius * math.cos(theta - radius))
            if i >= burn_in:
                wx = x / 2.6
                wy = y / 2.6
                dx = (wx - center_x) / scale
                dy = (wy - center_y) / scale
                vx = dx * cos_t + dy * sin_t
                vy = -dx * sin_t + dy * cos_t
                px = int((vx + 1.0) * 0.5 * (size - 1))
                py = int((1.0 - vy) * 0.5 * (size - 1))
                if 0 <= px < size and 0 <= py < size:
                    hist[py, px] += 1.0
                    angle_acc[py, px] += math.atan2(vy, vx + 1e-12)
        return hist, angle_acc


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


def unified_color(features: Dict[str, Array], palette: Dict[str, float], rng: np.random.Generator) -> Array:
    value = normalize01(features["value"])
    density = normalize01(features["density"])
    detail = normalize01(features["detail"])
    angle = (features["angle"] + math.pi) / (2.0 * math.pi)
    angle = np.mod(angle, 1.0)

    contrast = float(palette["contrast"])
    gamma = float(palette["gamma"])
    value = np.clip((value - 0.5) * contrast + 0.5, 0.0, 1.0)
    value = np.power(value, gamma)

    hue = np.mod(
        float(palette["hue"])
        + float(palette["hue_scale"]) * value
        + float(palette["angle_mix"]) * angle
        + 0.10 * detail,
        1.0,
    )
    sat = np.clip(float(palette["sat"]) * (0.55 + 0.55 * detail + 0.15 * density), 0.0, 1.0)
    val = np.clip(float(palette["val"]) * (0.18 + 0.82 * value) * (0.72 + 0.38 * density), 0.0, 1.0)

    rgb = hsv_to_rgb(hue, sat, val)
    bg = float(palette["background"])
    rgb = np.clip(rgb + bg * rng.random(3), 0.0, 1.0)
    rgb = subtle_vignette(rgb)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def hsv_to_rgb(h: Array, s: Array, v: Array) -> Array:
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    shape = h.shape + (3,)
    rgb = np.empty(shape, dtype=np.float64)
    masks = [i == k for k in range(6)]
    rgb[masks[0]] = np.stack([v, t, p], axis=-1)[masks[0]]
    rgb[masks[1]] = np.stack([q, v, p], axis=-1)[masks[1]]
    rgb[masks[2]] = np.stack([p, v, t], axis=-1)[masks[2]]
    rgb[masks[3]] = np.stack([p, q, v], axis=-1)[masks[3]]
    rgb[masks[4]] = np.stack([t, p, v], axis=-1)[masks[4]]
    rgb[masks[5]] = np.stack([v, p, q], axis=-1)[masks[5]]
    return rgb


def subtle_vignette(rgb: Array) -> Array:
    size = rgb.shape[0]
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x * x + y * y)
    vignette = np.clip(1.08 - 0.23 * r * r, 0.75, 1.08)
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
    warn_if_slow_path(args)

    image_dir, metadata_path, start_index, start_function_id = prepare_output(args)
    tasks = make_tasks(args, start_index=start_index, start_function_id=start_function_id)

    metadata_file = metadata_path.open("a", encoding="utf-8") if args.metadata else None
    try:
        progress = tqdm(iter_results(tasks, args.workers), total=len(tasks), desc="Generating fractals")
        for result in progress:
            metadata = save_result(result, image_dir)
            if metadata_file is not None:
                metadata_file.write(json.dumps(metadata, ensure_ascii=False, default=_json_default) + "\n")
    finally:
        if metadata_file is not None:
            metadata_file.close()

    print(f"Generated {len(tasks)} images in: {image_dir}")
    if args.metadata:
        print(f"Metadata appended to: {metadata_path}")


if __name__ == "__main__":
    # Keeps Windows multiprocessing safe when --workers > 1.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache"))
    main()