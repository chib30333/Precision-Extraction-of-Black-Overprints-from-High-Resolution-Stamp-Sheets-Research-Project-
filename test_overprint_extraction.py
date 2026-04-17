#!/usr/bin/env python3
r"""
test_overprint_extraction.py

Client-facing single-image test extractor for validating fidelity before the
full multi-PDF project is run.

This script:
- accepts one sample image
- preserves original resolution
- removes red annotations conservatively
- extracts only the black/purple overprint using the same research-safe logic
- writes a true binary black-on-transparent PNG as the primary output
- optionally writes a tight binary crop of the "1931" anchor block
- optionally writes a white-background preview
- writes a side-by-side comparison image with the original crop

Example:
    python test_overprint_extraction.py ^
      --input .\sample.png ^
      --output-dir .\test_output ^
      --transparent-bg ^
      --white-preview
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from extract_overprints import (
    ExtractionConfig,
    apply_optional_morphology,
    build_black_likelihood_mask,
    build_red_mask,
    clamp01,
    configure_logging,
    cv2_imwrite,
    render_rgba_on_bg,
    smoothstep,
    utc_now_iso,
)


LOGGER = logging.getLogger("test_overprint_extraction")


TEST_CONFIG_TEMPLATE: Dict[str, Any] = {
    "extraction": {
        "red_h1_max": 20,
        "red_h2_min": 165,
        "red_s_min": 35,
        "red_v_min": 25,
        "red_lab_a_min": 140,
        "darkness_low": 0.08,
        "darkness_high": 0.94,
        "neutrality_low": 0.05,
        "neutrality_high": 0.90,
        "red_penalty_strength": 0.90,
        "binary_threshold": 0.18,
        "morphology": {
            "enabled": True,
            "operation": "close",
            "kernel_size": 2,
            "iterations": 1,
        },
    }
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image conservative black/purple overprint extraction test."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input sample image")
    parser.add_argument("--output-dir", type=Path, default=Path("test_output"), help="Output folder")
    parser.add_argument("--config", type=Path, help="Optional JSON config for extraction tuning")
    parser.add_argument("--transparent-bg", action="store_true", help="Write transparent PNG output")
    parser.add_argument("--white-preview", action="store_true", help="Write white-background visibility preview")
    parser.add_argument("--save-debug", action="store_true", help="Save debug masks and previews")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pil_save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def make_binary_rgba(mask_u8: np.ndarray) -> np.ndarray:
    rgba = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 3] = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    return rgba


def make_white_preview(mask_u8: np.ndarray) -> np.ndarray:
    preview = np.full((mask_u8.shape[0], mask_u8.shape[1], 3), 255, dtype=np.uint8)
    preview[mask_u8 > 0] = 0
    return preview


def load_config(config_path: Optional[Path]) -> ExtractionConfig:
    raw = TEST_CONFIG_TEMPLATE
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            user = json.load(f)
        raw = deep_merge(TEST_CONFIG_TEMPLATE, user)
    extraction = raw.get("extraction", {})
    morphology = extraction.get("morphology", {})
    return ExtractionConfig(
        red_h1_max=int(extraction.get("red_h1_max", 20)),
        red_h2_min=int(extraction.get("red_h2_min", 165)),
        red_s_min=int(extraction.get("red_s_min", 35)),
        red_v_min=int(extraction.get("red_v_min", 25)),
        red_lab_a_min=int(extraction.get("red_lab_a_min", 140)),
        darkness_low=float(extraction.get("darkness_low", 0.08)),
        darkness_high=float(extraction.get("darkness_high", 0.94)),
        neutrality_low=float(extraction.get("neutrality_low", 0.05)),
        neutrality_high=float(extraction.get("neutrality_high", 0.90)),
        red_penalty_strength=float(extraction.get("red_penalty_strength", 0.90)),
        alpha_gamma=1.0,
        alpha_clip_min=0.0,
        binary_threshold=float(extraction.get("binary_threshold", 0.18)),
        morphology=ExtractionConfig().morphology.__class__(
            enabled=bool(morphology.get("enabled", True)),
            operation=str(morphology.get("operation", "close")),
            kernel_size=int(morphology.get("kernel_size", 2)),
            iterations=int(morphology.get("iterations", 1)),
        ),
    )


def read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read input image: {path}")
    return image


def build_purple_support_mask(image_bgr: np.ndarray, cfg: ExtractionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    b = image_bgr[:, :, 0].astype(np.float32) / 255.0
    g = image_bgr[:, :, 1].astype(np.float32) / 255.0
    r = image_bgr[:, :, 2].astype(np.float32) / 255.0
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Purple overprint can be dark but chromatic, so we allow a conservative
    # auxiliary support path instead of insisting on near-neutral black only.
    purple_h = np.logical_and(h >= 118, h <= 165).astype(np.float32)
    purple_sat = smoothstep(0.10, 0.95, s)
    purple_dark = smoothstep(cfg.darkness_low, cfg.darkness_high, 1.0 - v)
    rb_balance = 1.0 - np.abs(r - b)
    rb_support = smoothstep(0.20, 0.95, rb_balance)
    green_suppression = smoothstep(0.0, 0.70, 1.0 - g)
    return clamp01(purple_h * purple_sat * purple_dark * rb_support * green_suppression)


def extract_black_purple_overprint(image_bgr: np.ndarray, cfg: ExtractionConfig) -> Dict[str, np.ndarray]:
    red_mask = build_red_mask(image_bgr, cfg)
    black_likelihood = build_black_likelihood_mask(image_bgr, red_mask, cfg)
    purple_support = build_purple_support_mask(image_bgr, cfg)

    combined = np.maximum(black_likelihood, purple_support * 0.92)
    combined *= (1.0 - red_mask)
    combined = clamp01(combined)
    combined = apply_optional_morphology(combined, cfg.morphology)

    binary_mask = (combined >= cfg.binary_threshold).astype(np.uint8) * 255
    rgba = make_binary_rgba(binary_mask)
    return {
        "rgba": rgba,
        "binary_mask": binary_mask,
        "red_mask": np.clip(red_mask * 255.0, 0, 255).astype(np.uint8),
        "black_likelihood": np.clip(black_likelihood * 255.0, 0, 255).astype(np.uint8),
        "purple_support": np.clip(purple_support * 255.0, 0, 255).astype(np.uint8),
        "combined": np.clip(combined * 255.0, 0, 255).astype(np.uint8),
    }


def rotate_image(image: np.ndarray, angle_degrees: float, border_value: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(math.ceil((h * sin) + (w * cos)))
    new_h = int(math.ceil((h * cos) + (w * sin)))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def crop_to_foreground(mask_u8: np.ndarray, padding: int = 0) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return mask_u8.copy(), None
    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + 1 + padding, mask_u8.shape[1])
    y2 = min(int(ys.max()) + 1 + padding, mask_u8.shape[0])
    return mask_u8[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _component_candidates(mask_u8: np.ndarray) -> List[Dict[str, float]]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    components: List[Dict[str, float]] = []
    if num_labels <= 1:
        return components

    area_floor = max(5, int(round(mask_u8.shape[0] * mask_u8.shape[1] * 0.00015)))
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < area_floor or w <= 0 or h <= 0:
            continue
        cx, cy = centroids[label]
        components.append(
            {
                "label": float(label),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "area": float(area),
                "cx": float(cx),
                "cy": float(cy),
            }
        )
    return components


def find_anchor_block_bbox(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    components = _component_candidates(mask_u8)
    if not components:
        return None

    h, w = mask_u8.shape[:2]
    median_h = float(np.median([comp["h"] for comp in components]))
    max_gap = max(6.0, median_h * 1.4)
    max_vdiff = max(6.0, median_h * 0.9)
    lower_limit = h * 0.30

    lower_components = [comp for comp in components if comp["cy"] >= lower_limit]
    if len(lower_components) < 2:
        lower_components = components
    lower_components.sort(key=lambda comp: comp["x"])

    best_score = float("-inf")
    best_group_len = 0
    best_bbox: Optional[Tuple[int, int, int, int]] = None

    for start in range(len(lower_components)):
        group = [lower_components[start]]
        x_left = lower_components[start]["x"]
        x_right = lower_components[start]["x"] + lower_components[start]["w"]
        y_top = lower_components[start]["y"]
        y_bottom = lower_components[start]["y"] + lower_components[start]["h"]
        area_sum = lower_components[start]["area"]

        for end in range(start + 1, len(lower_components)):
            candidate = lower_components[end]
            gap = candidate["x"] - x_right
            center_y = (y_top + y_bottom) / 2.0
            if gap > max_gap or abs(candidate["cy"] - center_y) > max_vdiff:
                break
            group.append(candidate)
            x_right = max(x_right, candidate["x"] + candidate["w"])
            y_top = min(y_top, candidate["y"])
            y_bottom = max(y_bottom, candidate["y"] + candidate["h"])
            area_sum += candidate["area"]

            width = x_right - x_left
            height = y_bottom - y_top
            if len(group) < 2 or height <= 0:
                continue

            aspect = width / max(height, 1.0)
            bottom_bias = y_bottom / max(float(h), 1.0)
            count_score = 1.0 - min(abs(len(group) - 4), 3) / 3.0
            aspect_score = max(0.0, 1.0 - abs(aspect - 3.2) / 3.0)
            fill_score = min(area_sum / max(width * height, 1.0), 1.0)
            score = (bottom_bias * 2.0) + (count_score * 2.2) + (aspect_score * 1.5) + fill_score

            if score > best_score:
                pad = max(2, int(round(height * 0.12)))
                best_score = score
                best_group_len = len(group)
                best_bbox = (
                    max(int(x_left) - pad, 0),
                    max(int(y_top) - pad, 0),
                    min(int(x_right) + pad, w),
                    min(int(y_bottom) + pad, h),
                )

    if best_bbox is not None and best_score >= 4.6 and best_group_len >= 3:
        return best_bbox
    return None


def estimate_foreground_angle(mask_u8: np.ndarray) -> float:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 2:
        return 0.0
    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    angle = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def build_anchor_outputs(mask_u8: np.ndarray) -> Dict[str, Any]:
    bbox = find_anchor_block_bbox(mask_u8)
    if bbox is None:
        return {"bbox": None, "raw_mask": None, "aligned_mask": None, "angle_degrees": 0.0}

    x1, y1, x2, y2 = bbox
    anchor_mask = mask_u8[y1:y2, x1:x2].copy()
    anchor_mask, _ = crop_to_foreground(anchor_mask, padding=1)

    angle = estimate_foreground_angle(anchor_mask)
    aligned_mask = None
    if abs(angle) >= 0.5:
        rotated = rotate_image(anchor_mask, -angle, border_value=0)
        aligned_mask, _ = crop_to_foreground(rotated, padding=1)

    return {"bbox": bbox, "raw_mask": anchor_mask, "aligned_mask": aligned_mask, "angle_degrees": angle}


def make_comparison_sheet(original_bgr: np.ndarray, extracted_rgba: np.ndarray) -> np.ndarray:
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    extracted_on_white = render_rgba_on_bg(extracted_rgba, 255)
    extracted_on_white = cv2.cvtColor(extracted_on_white, cv2.COLOR_BGR2RGB)

    h = max(original_rgb.shape[0], extracted_on_white.shape[0])
    gap = 24
    label_band = 50
    w = original_rgb.shape[1] + extracted_on_white.shape[1] + gap
    canvas = np.full((h + label_band, w, 3), 255, dtype=np.uint8)

    canvas[label_band : label_band + original_rgb.shape[0], 0 : original_rgb.shape[1]] = original_rgb
    right_x = original_rgb.shape[1] + gap
    canvas[
        label_band : label_band + extracted_on_white.shape[0],
        right_x : right_x + extracted_on_white.shape[1],
    ] = extracted_on_white

    cv2.putText(canvas, "Original Sample", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Extracted Overprint",
        (right_x, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        cfg = load_config(args.config)
        image = read_image(args.input)
        ensure_dir(args.output_dir)

        stem = args.input.stem
        transparent_path = args.output_dir / f"{stem}_overprint_transparent.png"
        white_path = args.output_dir / f"{stem}_overprint_white.png"
        anchor_path = args.output_dir / f"{stem}_1931_anchor_transparent.png"
        anchor_aligned_path = args.output_dir / f"{stem}_1931_anchor_aligned_transparent.png"
        comparison_path = args.output_dir / f"{stem}_comparison.png"
        original_copy_path = args.output_dir / f"{stem}_original.png"
        manifest_path = args.output_dir / f"{stem}_test_manifest.json"
        debug_dir = args.output_dir / "debug"

        result = extract_black_purple_overprint(image, cfg)
        pil_save_rgba(transparent_path, result["rgba"])
        cv2_imwrite(original_copy_path, image)
        anchor = build_anchor_outputs(result["binary_mask"])
        if anchor["raw_mask"] is not None:
            pil_save_rgba(anchor_path, make_binary_rgba(anchor["raw_mask"]))
        if anchor["aligned_mask"] is not None:
            pil_save_rgba(anchor_aligned_path, make_binary_rgba(anchor["aligned_mask"]))

        if args.white_preview or not args.transparent_bg:
            cv2_imwrite(white_path, make_white_preview(result["binary_mask"]))

        comparison = make_comparison_sheet(image, result["rgba"])
        cv2_imwrite(comparison_path, comparison)

        if args.save_debug:
            ensure_dir(debug_dir)
            cv2_imwrite(debug_dir / f"{stem}_red_mask.png", result["red_mask"])
            cv2_imwrite(debug_dir / f"{stem}_black_likelihood.png", result["black_likelihood"])
            cv2_imwrite(debug_dir / f"{stem}_purple_support.png", result["purple_support"])
            cv2_imwrite(debug_dir / f"{stem}_combined_mask.png", result["combined"])
            cv2_imwrite(debug_dir / f"{stem}_strict_binary.png", result["binary_mask"])
            cv2_imwrite(debug_dir / f"{stem}_preview_black.png", render_rgba_on_bg(result["rgba"], 0))

        write_manifest(
            manifest_path,
            {
                "created_utc": utc_now_iso(),
                "input": str(args.input),
                "output_dir": str(args.output_dir),
                "outputs": {
                    "transparent_png": str(transparent_path),
                    "white_preview": str(white_path) if (args.white_preview or not args.transparent_bg) else None,
                    "anchor_1931_transparent_png": str(anchor_path) if anchor["raw_mask"] is not None else None,
                    "anchor_1931_aligned_transparent_png": str(anchor_aligned_path) if anchor["aligned_mask"] is not None else None,
                    "comparison": str(comparison_path),
                    "original_copy": str(original_copy_path),
                },
                "config": {
                    "extraction": {
                        "red_h1_max": cfg.red_h1_max,
                        "red_h2_min": cfg.red_h2_min,
                        "red_s_min": cfg.red_s_min,
                        "red_v_min": cfg.red_v_min,
                        "red_lab_a_min": cfg.red_lab_a_min,
                        "darkness_low": cfg.darkness_low,
                        "darkness_high": cfg.darkness_high,
                        "neutrality_low": cfg.neutrality_low,
                        "neutrality_high": cfg.neutrality_high,
                        "red_penalty_strength": cfg.red_penalty_strength,
                        "binary_threshold": cfg.binary_threshold,
                        "morphology": {
                            "enabled": cfg.morphology.enabled,
                            "operation": cfg.morphology.operation,
                            "kernel_size": cfg.morphology.kernel_size,
                            "iterations": cfg.morphology.iterations,
                        },
                    }
                },
                "analysis_ready_output": {
                    "foreground": "pure_black",
                    "background": "transparent",
                    "anti_aliasing": "disabled",
                    "semi_transparent_pixels": False,
                },
                "anchor_detection": {
                    "bbox_xyxy": anchor["bbox"],
                    "aligned_angle_degrees": round(float(anchor["angle_degrees"]), 4),
                },
            },
        )

        print(f"Input processed: {args.input}")
        print(f"Transparent PNG: {transparent_path}")
        if args.white_preview or not args.transparent_bg:
            print(f"White preview: {white_path}")
        if anchor["raw_mask"] is not None:
            print(f"1931 anchor PNG: {anchor_path}")
        if anchor["aligned_mask"] is not None:
            print(f"1931 aligned anchor PNG: {anchor_aligned_path}")
        print(f"Comparison image: {comparison_path}")
        print(f"Original copy: {original_copy_path}")
        if args.save_debug:
            print(f"Debug directory: {debug_dir}")
        print(f"Manifest: {manifest_path}")
        return 0
    except Exception as exc:
        LOGGER.exception("Test extraction failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
