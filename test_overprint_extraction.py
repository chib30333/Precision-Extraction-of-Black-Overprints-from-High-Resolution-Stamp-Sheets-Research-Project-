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
- writes a transparent PNG as the primary output
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
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from extract_overprints import (
    ExtractionConfig,
    build_black_likelihood_mask,
    build_red_mask,
    clamp01,
    compose_rgba_foreground,
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
        "alpha_gamma": 1.0,
        "alpha_clip_min": 0.02,
        "binary_threshold": 0.18,
        "morphology": {
            "enabled": False,
            "operation": "none",
            "kernel_size": 1,
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
        alpha_gamma=float(extraction.get("alpha_gamma", 1.0)),
        alpha_clip_min=float(extraction.get("alpha_clip_min", 0.02)),
        binary_threshold=float(extraction.get("binary_threshold", 0.18)),
        morphology=ExtractionConfig().morphology.__class__(
            enabled=bool(morphology.get("enabled", False)),
            operation=str(morphology.get("operation", "none")),
            kernel_size=int(morphology.get("kernel_size", 1)),
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

    alpha = combined.copy()
    if cfg.alpha_gamma != 1.0 and cfg.alpha_gamma > 0:
        alpha = np.power(alpha, cfg.alpha_gamma)
    alpha[alpha < cfg.alpha_clip_min] = 0.0
    alpha = clamp01(alpha)

    rgba = compose_rgba_foreground(image_bgr, alpha, "transparent")
    return {
        "rgba": rgba,
        "alpha": np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
        "red_mask": np.clip(red_mask * 255.0, 0, 255).astype(np.uint8),
        "black_likelihood": np.clip(black_likelihood * 255.0, 0, 255).astype(np.uint8),
        "purple_support": np.clip(purple_support * 255.0, 0, 255).astype(np.uint8),
    }


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
        comparison_path = args.output_dir / f"{stem}_comparison.png"
        original_copy_path = args.output_dir / f"{stem}_original.png"
        manifest_path = args.output_dir / f"{stem}_test_manifest.json"
        debug_dir = args.output_dir / "debug"

        result = extract_black_purple_overprint(image, cfg)
        pil_save_rgba(transparent_path, result["rgba"])
        cv2_imwrite(original_copy_path, image)

        if args.white_preview or not args.transparent_bg:
            cv2_imwrite(white_path, render_rgba_on_bg(result["rgba"], 255))

        comparison = make_comparison_sheet(image, result["rgba"])
        cv2_imwrite(comparison_path, comparison)

        if args.save_debug:
            ensure_dir(debug_dir)
            cv2_imwrite(debug_dir / f"{stem}_red_mask.png", result["red_mask"])
            cv2_imwrite(debug_dir / f"{stem}_black_likelihood.png", result["black_likelihood"])
            cv2_imwrite(debug_dir / f"{stem}_purple_support.png", result["purple_support"])
            cv2_imwrite(debug_dir / f"{stem}_alpha.png", result["alpha"])
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
                        "alpha_gamma": cfg.alpha_gamma,
                        "alpha_clip_min": cfg.alpha_clip_min,
                        "binary_threshold": cfg.binary_threshold,
                    }
                },
            },
        )

        print(f"Input processed: {args.input}")
        print(f"Transparent PNG: {transparent_path}")
        if args.white_preview or not args.transparent_bg:
            print(f"White preview: {white_path}")
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
