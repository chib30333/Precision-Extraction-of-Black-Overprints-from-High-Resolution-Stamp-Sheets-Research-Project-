#!/usr/bin/env python3
"""
extract_overprint_test.py

Single-image test extractor for client review.

Purpose
-------
This script is intentionally narrower than the full sheet-processing pipeline.
It is built for a short validation task on one supplied sample image so the
client can verify fidelity before the larger project proceeds.

Primary goals
-------------
- Extract only the black or purple overprint from a single sample image
- Remove red arrows / red markings conservatively
- Preserve faint, broken, irregular, or incomplete ink exactly as scanned
- Avoid enhancement, cleanup, denoising, filling, sharpening, smoothing, or repair
- Preserve original pixel dimensions with lossless PNG outputs

Outputs
-------
- Transparent PNG of the extracted overprint
- Optional white-background visibility preview
- Side-by-side comparison image containing the original crop and extracted result

Typical usage
-------------
python extract_overprint_test.py ^
  --input .\\sample.png ^
  --output-dir .\\test_output ^
  --transparent-bg ^
  --save-white-preview

If the sample image includes surrounding context, an explicit crop may be used:

python extract_overprint_test.py ^
  --input .\\sample.png ^
  --output-dir .\\test_output ^
  --crop-xyxy 120 80 980 760 ^
  --transparent-bg ^
  --save-white-preview
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from extract_overprints import (
    ExtractionConfig,
    build_red_mask,
    clamp01,
    configure_logging,
    render_rgba_on_bg,
    smoothstep,
)


LOGGER = logging.getLogger("extract_overprint_test")


@dataclass
class TestExtractionConfig:
    red_h1_max: int = 20
    red_h2_min: int = 165
    red_s_min: int = 35
    red_v_min: int = 25
    red_lab_a_min: int = 140
    darkness_low: float = 0.10
    darkness_high: float = 0.92
    neutrality_low: float = 0.15
    neutrality_high: float = 0.92
    red_penalty_strength: float = 1.0
    alpha_gamma: float = 1.0
    alpha_clip_min: float = 0.03
    binary_threshold: float = 0.22
    purple_h_low: int = 118
    purple_h_high: int = 168
    purple_s_low: float = 0.08
    purple_s_high: float = 0.95
    purple_darkness_low: float = 0.08
    purple_darkness_high: float = 0.90
    purple_rb_delta_max: float = 0.22


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pil_save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def cv2_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-image black/purple overprint extraction test.")
    parser.add_argument("--input", type=Path, required=True, help="Input sample image path")
    parser.add_argument("--output-dir", type=Path, default=Path("test_output"), help="Output directory")
    parser.add_argument("--crop-xyxy", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Optional crop rectangle in absolute pixels")
    parser.add_argument("--crop-xywh", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Optional crop rectangle in absolute pixels")
    parser.add_argument("--transparent-bg", action="store_true", help="Write primary result with transparent background")
    parser.add_argument("--white-bg", action="store_true", help="Write primary result composited on white")
    parser.add_argument("--black-bg", action="store_true", help="Write primary result composited on black")
    parser.add_argument("--save-white-preview", action="store_true", help="Also save a white-background preview")
    parser.add_argument("--save-masks", action="store_true", help="Save debug masks for inspection")
    parser.add_argument("--config-json", type=Path, help="Optional JSON file overriding test extraction parameters")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def load_test_config(config_json: Optional[Path]) -> TestExtractionConfig:
    if config_json is None:
        return TestExtractionConfig()
    if not config_json.exists():
        raise FileNotFoundError(f"Config override not found: {config_json}")
    with config_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    merged = asdict(TestExtractionConfig())
    merged.update(payload)
    return TestExtractionConfig(**merged)


def resolve_background_mode(args: argparse.Namespace) -> str:
    if args.white_bg:
        return "white"
    if args.black_bg:
        return "black"
    return "transparent"


def load_image_bgr(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def parse_crop(args: argparse.Namespace, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if args.crop_xyxy and args.crop_xywh:
        raise ValueError("Use only one of --crop-xyxy or --crop-xywh")
    if args.crop_xyxy:
        x1, y1, x2, y2 = [int(v) for v in args.crop_xyxy]
    elif args.crop_xywh:
        x, y, w, h = [int(v) for v in args.crop_xywh]
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        return None
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def crop_image(image: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop is None:
        return image.copy()
    x1, y1, x2, y2 = crop
    result = image[y1:y2, x1:x2].copy()
    if result.size == 0:
        raise ValueError("Crop produced an empty image")
    return result


def band_pass_smooth(x: np.ndarray, low: float, high: float, edge: float) -> np.ndarray:
    left = smoothstep(low - edge, low + edge, x)
    right = 1.0 - smoothstep(high - edge, high + edge, x)
    return clamp01(left * right)


def build_black_or_purple_likelihood(crop_bgr: np.ndarray, red_mask: np.ndarray, cfg: TestExtractionConfig) -> np.ndarray:
    b = crop_bgr[:, :, 0].astype(np.float32) / 255.0
    g = crop_bgr[:, :, 1].astype(np.float32) / 255.0
    r = crop_bgr[:, :, 2].astype(np.float32) / 255.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    chroma = maxc - minc
    darkness = 1.0 - maxc
    neutrality = 1.0 - chroma
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark_gray = 1.0 - gray

    # Conservative black score: favors dark, neutral pixels while keeping weak gray ink.
    black_dark_score = np.maximum(
        smoothstep(cfg.darkness_low, cfg.darkness_high, darkness),
        smoothstep(cfg.darkness_low, cfg.darkness_high, dark_gray),
    )
    neutral_score = smoothstep(cfg.neutrality_low, cfg.neutrality_high, neutrality)
    black_score = black_dark_score * neutral_score

    # Purple support for dark magenta-violet overprint without admitting bright red annotations.
    hue_score = band_pass_smooth(h, float(cfg.purple_h_low), float(cfg.purple_h_high), edge=6.0)
    sat_score = band_pass_smooth(s, cfg.purple_s_low, cfg.purple_s_high, edge=0.08)
    purple_dark_score = smoothstep(cfg.purple_darkness_low, cfg.purple_darkness_high, darkness)

    # Purple ink tends to keep red and blue closer to each other than red arrows do.
    rb_delta = np.abs(r - b)
    rb_similarity = 1.0 - smoothstep(cfg.purple_rb_delta_max * 0.5, cfg.purple_rb_delta_max, rb_delta)

    # Penalize green-heavy pixels so stamp artwork is less likely to leak in.
    purple_balance = clamp01(((r + b) * 0.5) - g + 0.25)
    purple_score = hue_score * sat_score * purple_dark_score * rb_similarity * purple_balance

    likelihood = np.maximum(black_score, purple_score)
    likelihood *= (1.0 - clamp01(red_mask * cfg.red_penalty_strength))
    return clamp01(likelihood.astype(np.float32))


def compose_rgba(crop_bgr: np.ndarray, alpha: np.ndarray, background_mode: str) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    if background_mode == "transparent":
        return rgba

    alpha_f = rgba[:, :, 3].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    if background_mode == "white":
        bg = np.full_like(fg, 255.0)
    elif background_mode == "black":
        bg = np.zeros_like(fg)
    else:
        raise ValueError(f"Unsupported background mode: {background_mode}")
    comp = fg * alpha_f[:, :, None] + bg * (1.0 - alpha_f[:, :, None])
    return np.dstack([comp.astype(np.uint8), np.full_like(rgba[:, :, 3], 255, dtype=np.uint8)])


def extract_black_purple_overprint(
    crop_bgr: np.ndarray,
    cfg: TestExtractionConfig,
    background_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_cfg = ExtractionConfig(
        red_h1_max=cfg.red_h1_max,
        red_h2_min=cfg.red_h2_min,
        red_s_min=cfg.red_s_min,
        red_v_min=cfg.red_v_min,
        red_lab_a_min=cfg.red_lab_a_min,
        darkness_low=cfg.darkness_low,
        darkness_high=cfg.darkness_high,
        neutrality_low=cfg.neutrality_low,
        neutrality_high=cfg.neutrality_high,
        red_penalty_strength=cfg.red_penalty_strength,
        alpha_gamma=cfg.alpha_gamma,
        alpha_clip_min=cfg.alpha_clip_min,
        binary_threshold=cfg.binary_threshold,
    )
    red_mask = build_red_mask(crop_bgr, base_cfg)
    likelihood = build_black_or_purple_likelihood(crop_bgr, red_mask, cfg)

    alpha = likelihood.copy()
    if cfg.alpha_gamma != 1.0 and cfg.alpha_gamma > 0:
        alpha = np.power(alpha, cfg.alpha_gamma)
    alpha[alpha < cfg.alpha_clip_min] = 0.0
    alpha = clamp01(alpha)

    rgba = compose_rgba(crop_bgr, alpha, background_mode)
    return (
        rgba,
        np.clip(red_mask * 255.0, 0, 255).astype(np.uint8),
        np.clip(likelihood * 255.0, 0, 255).astype(np.uint8),
        np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
    )


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    label_h = 44
    out = np.full((image.shape[0] + label_h, image.shape[1], 3), 255, dtype=np.uint8)
    out[label_h:, :, :] = image
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def make_side_by_side(original_bgr: np.ndarray, extracted_rgba: np.ndarray) -> np.ndarray:
    extracted_white = render_rgba_on_bg(extracted_rgba, 255)
    left = add_label(original_bgr, "Original crop")
    right = add_label(extracted_white, "Extracted overprint")
    gap = 24
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + gap + right.shape[1]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right
    return canvas


def write_manifest(output_dir: Path, payload: Dict[str, Any]) -> None:
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        cfg = load_test_config(args.config_json)
        background_mode = resolve_background_mode(args)
        ensure_dir(args.output_dir)

        source = load_image_bgr(args.input)
        crop_rect = parse_crop(args, source.shape[1], source.shape[0])
        crop = crop_image(source, crop_rect)

        rgba, red_mask, likelihood, alpha = extract_black_purple_overprint(crop, cfg, background_mode)

        transparent_rgba, _, _, _ = extract_black_purple_overprint(crop, cfg, "transparent")
        transparent_path = args.output_dir / "extracted_overprint_transparent.png"
        pil_save_rgba(transparent_path, transparent_rgba)

        primary_name = {
            "transparent": "extracted_overprint_primary.png",
            "white": "extracted_overprint_primary_white.png",
            "black": "extracted_overprint_primary_black.png",
        }[background_mode]
        pil_save_rgba(args.output_dir / primary_name, rgba)

        if args.save_white_preview:
            white_preview = render_rgba_on_bg(transparent_rgba, 255)
            cv2_imwrite(args.output_dir / "extracted_overprint_white_preview.png", white_preview)

        cv2_imwrite(args.output_dir / "original_crop.png", crop)
        comparison = make_side_by_side(crop, transparent_rgba)
        cv2_imwrite(args.output_dir / "comparison_original_vs_extracted.png", comparison)

        if args.save_masks:
            cv2_imwrite(args.output_dir / "debug_red_mask.png", red_mask)
            cv2_imwrite(args.output_dir / "debug_black_purple_likelihood.png", likelihood)
            cv2_imwrite(args.output_dir / "debug_alpha_mask.png", alpha)

        manifest = {
            "created_utc": utc_now_iso(),
            "input": str(args.input),
            "output_dir": str(args.output_dir),
            "crop": {
                "mode": "full-image" if crop_rect is None else "manual",
                "x1": None if crop_rect is None else crop_rect[0],
                "y1": None if crop_rect is None else crop_rect[1],
                "x2": None if crop_rect is None else crop_rect[2],
                "y2": None if crop_rect is None else crop_rect[3],
            },
            "source_size": {"width": int(source.shape[1]), "height": int(source.shape[0])},
            "crop_size": {"width": int(crop.shape[1]), "height": int(crop.shape[0])},
            "background_mode": background_mode,
            "save_white_preview": bool(args.save_white_preview),
            "save_masks": bool(args.save_masks),
            "config": asdict(cfg),
            "outputs": {
                "transparent_png": str(transparent_path),
                "primary_output": str(args.output_dir / primary_name),
                "original_crop": str(args.output_dir / "original_crop.png"),
                "comparison": str(args.output_dir / "comparison_original_vs_extracted.png"),
            },
        }
        write_manifest(args.output_dir, manifest)

        print(f"Input image: {args.input}")
        print(f"Crop used: {'full image' if crop_rect is None else crop_rect}")
        print(f"Transparent PNG: {transparent_path}")
        print(f"Comparison image: {args.output_dir / 'comparison_original_vs_extracted.png'}")
        if args.save_white_preview:
            print(f"White preview: {args.output_dir / 'extracted_overprint_white_preview.png'}")
        if args.save_masks:
            print(f"Debug masks saved in: {args.output_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("Test extraction failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
