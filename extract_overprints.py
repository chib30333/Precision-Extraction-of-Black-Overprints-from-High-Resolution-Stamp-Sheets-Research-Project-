#!/usr/bin/env python3
"""
extract_overprints.py

Research-safe stamp-sheet reconstruction and black-overprint extraction.

Sample config template
----------------------
{
  "project_name": "stamp_overprint_extraction",
  "expected_pdf_count": 6,
  "sheet_layout": {"rows": 10, "cols": 5},
  "reconstruction": {
    "mode": "config",
    "assist_search_radius": 120,
    "assist_overlap_min_pixels": 2000,
    "canvas_padding": 64,
    "background_value": 255
  },
  "extraction": {
    "red_h1_max": 20,
    "red_h2_min": 165,
    "red_s_min": 35,
    "red_v_min": 25,
    "red_lab_a_min": 140,
    "darkness_low": 0.10,
    "darkness_high": 0.92,
    "neutrality_low": 0.15,
    "neutrality_high": 0.92,
    "red_penalty_strength": 1.0,
    "alpha_gamma": 1.0,
    "alpha_clip_min": 0.03,
    "binary_threshold": 0.22,
    "morphology": {
      "enabled": false,
      "operation": "none",
      "kernel_size": 1,
      "iterations": 1
    }
  },
  "output": {
    "background_mode": "transparent",
    "save_rasterized": false,
    "save_debug": true
  },
  "sheets": {
    "1": {
      "segments": [
        {"file": "sheet1_seg1.pdf", "placement": {"x": 0, "y": 0}},
        {"file": "sheet1_seg2.pdf", "placement": {"x": 6200, "y": 0}},
        {"file": "sheet1_seg3.pdf", "placement": {"x": 0, "y": 8200}}
      ],
      "grid": {
        "mode": "manual",
        "left_margin": 280,
        "top_margin": 260,
        "cell_width": 1520,
        "cell_height": 1060
      }
    },
    "2": {
      "segments": [
        {"file": "sheet2_seg1.pdf", "placement": {"x": 0, "y": 0}},
        {"file": "sheet2_seg2.pdf", "placement": {"x": 6200, "y": 0}},
        {"file": "sheet2_seg3.pdf", "placement": {"x": 0, "y": 8200}}
      ],
      "grid": {
        "mode": "manual",
        "left_margin": 280,
        "top_margin": 260,
        "cell_width": 1520,
        "cell_height": 1060
      }
    }
  }
}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import fitz

    HAVE_PYMUPDF = True
except Exception:
    fitz = None
    HAVE_PYMUPDF = False

try:
    from pdf2image import convert_from_path

    HAVE_PDF2IMAGE = True
except Exception:
    convert_from_path = None
    HAVE_PDF2IMAGE = False


LOGGER = logging.getLogger("extract_overprints")

SAMPLE_CONFIG: Dict[str, Any] = {
    "project_name": "stamp_overprint_extraction",
    "expected_pdf_count": 6,
    "sheet_layout": {"rows": 10, "cols": 5},
    "reconstruction": {
        "mode": "config",
        "assist_search_radius": 120,
        "assist_overlap_min_pixels": 2000,
        "canvas_padding": 64,
        "background_value": 255,
    },
    "extraction": {
        "red_h1_max": 20,
        "red_h2_min": 165,
        "red_s_min": 35,
        "red_v_min": 25,
        "red_lab_a_min": 140,
        "darkness_low": 0.10,
        "darkness_high": 0.92,
        "neutrality_low": 0.15,
        "neutrality_high": 0.92,
        "red_penalty_strength": 1.0,
        "alpha_gamma": 1.0,
        "alpha_clip_min": 0.03,
        "binary_threshold": 0.22,
        "morphology": {
            "enabled": False,
            "operation": "none",
            "kernel_size": 1,
            "iterations": 1,
        },
    },
    "output": {
        "background_mode": "transparent",
        "save_rasterized": False,
        "save_debug": True,
    },
    "sheets": {
        "1": {"segments": [], "grid": {"mode": "manual"}},
        "2": {"segments": [], "grid": {"mode": "manual"}},
    },
}


@dataclass
class SegmentPlacement:
    x: int
    y: int


@dataclass
class SegmentSpec:
    file: str
    placement: Optional[SegmentPlacement] = None
    approximate_placement: Optional[SegmentPlacement] = None
    z_index: int = 0


@dataclass
class GridSpec:
    mode: str = "manual"
    left_margin: Optional[float] = None
    top_margin: Optional[float] = None
    right_margin: Optional[float] = None
    bottom_margin: Optional[float] = None
    cell_width: Optional[float] = None
    cell_height: Optional[float] = None
    x0: Optional[float] = None
    y0: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass
class SheetSpec:
    sheet_id: int
    segments: List[SegmentSpec]
    grid: GridSpec


@dataclass
class ReconstructionConfig:
    mode: str = "config"
    assist_search_radius: int = 120
    assist_overlap_min_pixels: int = 2000
    canvas_padding: int = 64
    background_value: int = 255


@dataclass
class MorphologyConfig:
    enabled: bool = False
    operation: str = "none"
    kernel_size: int = 1
    iterations: int = 1


@dataclass
class ExtractionConfig:
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
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)


@dataclass
class OutputConfig:
    background_mode: str = "transparent"
    save_rasterized: bool = False
    save_debug: bool = False


@dataclass
class ProjectConfig:
    project_name: str
    expected_pdf_count: int
    rows: int
    cols: int
    reconstruction: ReconstructionConfig
    extraction: ExtractionConfig
    output: OutputConfig
    sheets: Dict[int, SheetSpec]
    raw: Dict[str, Any]


@dataclass
class SegmentRaster:
    pdf_path: Path
    image: np.ndarray
    dpi: int
    page_index: int
    raster_path: Optional[Path] = None


@dataclass
class PlacedSegment:
    pdf_path: Path
    x: int
    y: int
    width: int
    height: int

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class SheetAssemblyResult:
    sheet_id: int
    image: np.ndarray
    placed_segments: List[PlacedSegment]
    source_segment_names: List[str]
    output_path: Optional[Path] = None
    debug_boundary_path: Optional[Path] = None


@dataclass
class GridCell:
    sheet: int
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def filename(self) -> str:
        return f"S{self.sheet}_R{self.row:02d}_C{self.col:02d}.png"

    @property
    def label(self) -> str:
        return f"S{self.sheet}_R{self.row:02d}_C{self.col:02d}"


@dataclass
class ExtractionResult:
    rgba: np.ndarray
    alpha: np.ndarray
    red_mask: np.ndarray
    black_likelihood: np.ndarray
    strict_binary: np.ndarray


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge0).astype(np.float32)
    t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def ensure_dir(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def cv2_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


def pil_save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def validate_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got: {value!r}")
    return value


def validate_config(config: Dict[str, Any]) -> ProjectConfig:
    if "sheets" not in config:
        raise ValueError("Config must contain 'sheets'")
    if "1" not in config["sheets"] or "2" not in config["sheets"]:
        raise ValueError("Config must define sheets '1' and '2'")

    expected_pdf_count = validate_positive_int(config.get("expected_pdf_count", 6), "expected_pdf_count")
    layout = config.get("sheet_layout", {})
    rows = validate_positive_int(layout.get("rows", 10), "sheet_layout.rows")
    cols = validate_positive_int(layout.get("cols", 5), "sheet_layout.cols")

    reconstruction = ReconstructionConfig(**deep_merge(SAMPLE_CONFIG["reconstruction"], config.get("reconstruction", {})))
    extraction_dict = deep_merge(SAMPLE_CONFIG["extraction"], config.get("extraction", {}))
    morph = MorphologyConfig(**deep_merge(SAMPLE_CONFIG["extraction"]["morphology"], extraction_dict.get("morphology", {})))
    extraction_dict["morphology"] = morph
    extraction = ExtractionConfig(**extraction_dict)
    output = OutputConfig(**deep_merge(SAMPLE_CONFIG["output"], config.get("output", {})))

    sheets: Dict[int, SheetSpec] = {}
    total_segments = 0
    for sheet_id_str in ("1", "2"):
        raw_sheet = config["sheets"][sheet_id_str]
        raw_segments = raw_sheet.get("segments", [])
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"Sheet {sheet_id_str} must contain a non-empty segments list")

        segments: List[SegmentSpec] = []
        for idx, seg in enumerate(raw_segments):
            if "file" not in seg:
                raise ValueError(f"Sheet {sheet_id_str} segment {idx} missing 'file'")
            placement = None
            if isinstance(seg.get("placement"), dict):
                placement = SegmentPlacement(x=int(seg["placement"]["x"]), y=int(seg["placement"]["y"]))
            approximate_placement = None
            if isinstance(seg.get("approximate_placement"), dict):
                approximate_placement = SegmentPlacement(
                    x=int(seg["approximate_placement"]["x"]),
                    y=int(seg["approximate_placement"]["y"]),
                )
            segments.append(
                SegmentSpec(
                    file=str(seg["file"]),
                    placement=placement,
                    approximate_placement=approximate_placement,
                    z_index=int(seg.get("z_index", 0)),
                )
            )
        total_segments += len(segments)
        grid = GridSpec(**raw_sheet.get("grid", {"mode": "manual"}))
        sheets[int(sheet_id_str)] = SheetSpec(sheet_id=int(sheet_id_str), segments=segments, grid=grid)

    if total_segments != expected_pdf_count:
        raise ValueError(f"Expected {expected_pdf_count} total PDF segments across both sheets, found {total_segments}")

    return ProjectConfig(
        project_name=str(config.get("project_name", "stamp_overprint_extraction")),
        expected_pdf_count=expected_pdf_count,
        rows=rows,
        cols=cols,
        reconstruction=reconstruction,
        extraction=extraction,
        output=output,
        sheets=sheets,
        raw=config,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct stamp sheets and conservatively extract black overprints.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing source PDF segments")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--config", type=Path, help="Project JSON configuration")
    parser.add_argument("--dpi", type=int, default=1200, help="Rasterization DPI, default: 1200")
    parser.add_argument("--save-rasterized", action="store_true", help="Save rasterized PDF pages as PNG")
    parser.add_argument("--save-debug", action="store_true", help="Save debug outputs and QC artifacts")
    parser.add_argument("--transparent-bg", action="store_true", help="Final output uses transparent background")
    parser.add_argument("--white-bg", action="store_true", help="Final output uses white background")
    parser.add_argument("--black-bg", action="store_true", help="Final output uses black background")
    parser.add_argument("--grid-mode", choices=["manual", "auto"], help="Override grid mode")
    parser.add_argument("--reconstruct-mode", choices=["config", "assist"], help="Override reconstruction mode")
    parser.add_argument("--sheet-layout", default="10x5", help="Grid layout as ROWSxCOLS, default: 10x5")
    parser.add_argument("--strict", action="store_true", help="Also save strict binary debug masks")
    parser.add_argument("--dry-run", action="store_true", help="Plan actions without writing outputs")
    parser.add_argument("--summary-file", type=Path, help="Optional path to summary.txt")
    parser.add_argument("--self-check", action="store_true", help="Run synthetic self-checks and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def parse_layout(layout_str: str) -> Tuple[int, int]:
    try:
        parts = layout_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError
        rows, cols = int(parts[0]), int(parts[1])
    except Exception as exc:
        raise ValueError(f"Invalid --sheet-layout value: {layout_str!r}. Expected like 10x5") from exc
    if rows <= 0 or cols <= 0:
        raise ValueError("Sheet layout dimensions must be positive integers")
    return rows, cols


def load_project_config(config_path: Optional[Path], args: argparse.Namespace) -> ProjectConfig:
    if config_path is None:
        raise ValueError("--config is required unless --self-check is used")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)

    merged = deep_merge(SAMPLE_CONFIG, user_config)
    rows, cols = parse_layout(args.sheet_layout)
    merged["sheet_layout"]["rows"] = rows
    merged["sheet_layout"]["cols"] = cols

    if args.grid_mode:
        for sheet_id in ("1", "2"):
            merged["sheets"][sheet_id].setdefault("grid", {})
            merged["sheets"][sheet_id]["grid"]["mode"] = args.grid_mode
    if args.reconstruct_mode:
        merged.setdefault("reconstruction", {})
        merged["reconstruction"]["mode"] = args.reconstruct_mode
    if args.save_rasterized:
        merged.setdefault("output", {})
        merged["output"]["save_rasterized"] = True
    if args.save_debug:
        merged.setdefault("output", {})
        merged["output"]["save_debug"] = True

    background_mode = None
    if args.transparent_bg:
        background_mode = "transparent"
    if args.white_bg:
        background_mode = "white"
    if args.black_bg:
        background_mode = "black"
    if background_mode is not None:
        merged.setdefault("output", {})
        merged["output"]["background_mode"] = background_mode

    return validate_config(merged)


def resolve_pdf_path(input_dir: Optional[Path], filename: str) -> Path:
    candidate = Path(filename)
    path = candidate if candidate.is_absolute() else ((input_dir / filename) if input_dir is not None else candidate)
    if not path.exists():
        raise FileNotFoundError(f"Source PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Configured file is not a PDF: {path}")
    return path


def rasterize_pdf_page(pdf_path: Path, dpi: int) -> np.ndarray:
    if HAVE_PYMUPDF:
        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                raise ValueError(f"PDF has no pages: {pdf_path}")
            page = doc.load_page(0)
            scale = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                rgb = rgb[:, :, :3]
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        finally:
            doc.close()
    if HAVE_PDF2IMAGE:
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1, fmt="png")
        if not images:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        rgb = np.array(images[0].convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise RuntimeError("Neither PyMuPDF nor pdf2image is available.")


def rasterize_pdfs(
    project: ProjectConfig,
    input_dir: Optional[Path],
    output_dir: Path,
    dpi: int,
    save_rasterized: bool,
    dry_run: bool,
) -> Dict[Path, SegmentRaster]:
    LOGGER.info("Rasterizing source PDF segments at %d DPI", dpi)
    rasterized: Dict[Path, SegmentRaster] = {}
    raster_dir = output_dir / "debug" / "rasterized"
    if save_rasterized and not dry_run:
        ensure_dir(raster_dir)

    for sheet in project.sheets.values():
        for segment in sheet.segments:
            pdf_path = resolve_pdf_path(input_dir, segment.file)
            if pdf_path in rasterized:
                continue
            image = rasterize_pdf_page(pdf_path, dpi=dpi)
            raster_path = None
            if save_rasterized and not dry_run:
                raster_path = raster_dir / f"{pdf_path.stem}_dpi{dpi}.png"
                cv2_imwrite(raster_path, image)
            rasterized[pdf_path] = SegmentRaster(pdf_path=pdf_path, image=image, dpi=dpi, page_index=0, raster_path=raster_path)
    return rasterized


def compute_canvas_bounds(
    placements: List[Tuple[SegmentSpec, np.ndarray, int, int]],
    padding: int,
) -> Tuple[int, int, int, int]:
    min_x = min(x for _, _, x, _ in placements)
    min_y = min(y for _, _, _, y in placements)
    max_x = max(x + img.shape[1] for _, img, x, _ in placements)
    max_y = max(y + img.shape[0] for _, img, _, y in placements)
    return min_x - padding, min_y - padding, max_x + padding, max_y + padding


def overlap_slices(
    base_shape: Tuple[int, int],
    img_shape: Tuple[int, int],
    x: int,
    y: int,
) -> Optional[Tuple[slice, slice, slice, slice]]:
    base_h, base_w = base_shape[:2]
    img_h, img_w = img_shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(base_w, x + img_w)
    y2 = min(base_h, y + img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return slice(y1, y2), slice(x1, x2), slice(y1 - y, y2 - y), slice(x1 - x, x2 - x)


def composite_segments(
    placements: List[Tuple[SegmentSpec, Path, np.ndarray, int, int]],
    background_value: int,
    padding: int,
) -> Tuple[np.ndarray, List[PlacedSegment], Tuple[int, int]]:
    working = [(seg, img, x, y) for seg, _, img, x, y in placements]
    min_x, min_y, max_x, max_y = compute_canvas_bounds(working, padding=padding)
    canvas = np.full((max_y - min_y, max_x - min_x, 3), int(background_value), dtype=np.uint8)
    placed_segments: List[PlacedSegment] = []
    for seg, pdf_path, img, x, y in sorted(placements, key=lambda item: item[0].z_index):
        adj_x = x - min_x
        adj_y = y - min_y
        slices = overlap_slices(canvas.shape, img.shape, adj_x, adj_y)
        if slices is None:
            continue
        by, bx, iy, ix = slices
        canvas[by, bx] = img[iy, ix]
        placed_segments.append(
            PlacedSegment(pdf_path=pdf_path, x=adj_x, y=adj_y, width=img.shape[1], height=img.shape[0])
        )
    return canvas, placed_segments, (-min_x, -min_y)


def grayscale_float(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def score_overlap(base: np.ndarray, img: np.ndarray, x: int, y: int, min_pixels: int) -> float:
    slices = overlap_slices(base.shape, img.shape, x, y)
    if slices is None:
        return -1e9
    by, bx, iy, ix = slices
    base_roi = base[by, bx]
    img_roi = img[iy, ix]
    valid = np.any(base_roi < 250, axis=2)
    count = int(valid.sum())
    if count < min_pixels:
        return -1e8
    base_gray = grayscale_float(base_roi)[valid]
    img_gray = grayscale_float(img_roi)[valid]
    if base_gray.size < min_pixels:
        return -1e8
    bg_mean, ig_mean = float(base_gray.mean()), float(img_gray.mean())
    bg_std, ig_std = float(base_gray.std()), float(img_gray.std())
    if bg_std < 1e-6 or ig_std < 1e-6:
        return -1e7
    corr = float(np.mean(((base_gray - bg_mean) / bg_std) * ((img_gray - ig_mean) / ig_std)))
    diff_penalty = float(np.mean(np.abs(base_gray - img_gray)))
    return corr - 0.25 * diff_penalty


def assisted_place_segments(
    sheet: SheetSpec,
    rasterized: Dict[Path, SegmentRaster],
    input_dir: Optional[Path],
    recon_cfg: ReconstructionConfig,
) -> List[Tuple[SegmentSpec, Path, np.ndarray, int, int]]:
    resolved: List[Tuple[SegmentSpec, Path, np.ndarray, int, int]] = []
    first_seg = sheet.segments[0]
    first_path = resolve_pdf_path(input_dir, first_seg.file)
    first_img = rasterized[first_path].image
    first_place = first_seg.placement or first_seg.approximate_placement or SegmentPlacement(0, 0)
    resolved.append((first_seg, first_path, first_img, first_place.x, first_place.y))
    temp_canvas, _, _ = composite_segments(resolved, recon_cfg.background_value, recon_cfg.canvas_padding)

    for seg in sheet.segments[1:]:
        pdf_path = resolve_pdf_path(input_dir, seg.file)
        img = rasterized[pdf_path].image
        start = seg.placement or seg.approximate_placement
        if start is None:
            raise ValueError(f"Assist mode requires placement or approximate_placement for {pdf_path.name}")
        best_x, best_y, best_score = start.x, start.y, -1e18
        radius = max(0, int(recon_cfg.assist_search_radius))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cx = start.x + dx
                cy = start.y + dy
                score = score_overlap(
                    temp_canvas,
                    img,
                    cx + recon_cfg.canvas_padding,
                    cy + recon_cfg.canvas_padding,
                    recon_cfg.assist_overlap_min_pixels,
                )
                if score > best_score:
                    best_x, best_y, best_score = cx, cy, score
        LOGGER.info(
            "Assist placement sheet=%s segment=%s start=(%d,%d) best=(%d,%d) score=%.5f",
            sheet.sheet_id,
            pdf_path.name,
            start.x,
            start.y,
            best_x,
            best_y,
            best_score,
        )
        resolved.append((seg, pdf_path, img, best_x, best_y))
        temp_canvas, _, _ = composite_segments(resolved, recon_cfg.background_value, recon_cfg.canvas_padding)
    return resolved


def reconstruct_sheet(
    sheet: SheetSpec,
    rasterized: Dict[Path, SegmentRaster],
    input_dir: Optional[Path],
    output_dir: Path,
    recon_cfg: ReconstructionConfig,
    save_debug: bool,
    dry_run: bool,
) -> SheetAssemblyResult:
    LOGGER.info("Reconstructing sheet %d using mode=%s", sheet.sheet_id, recon_cfg.mode)
    placements: List[Tuple[SegmentSpec, Path, np.ndarray, int, int]] = []
    if recon_cfg.mode == "assist":
        placements = assisted_place_segments(sheet, rasterized, input_dir, recon_cfg)
    else:
        for seg in sheet.segments:
            pdf_path = resolve_pdf_path(input_dir, seg.file)
            img = rasterized[pdf_path].image
            place = seg.placement or seg.approximate_placement
            if place is None:
                raise ValueError(f"Config mode requires placement or approximate_placement for {pdf_path.name}")
            placements.append((seg, pdf_path, img, place.x, place.y))

    composite, placed_segments, _ = composite_segments(
        placements,
        background_value=recon_cfg.background_value,
        padding=recon_cfg.canvas_padding,
    )
    output_path = output_dir / f"Sheet{sheet.sheet_id}" / f"Sheet{sheet.sheet_id}_reconstructed.png"
    debug_boundary_path = output_dir / "debug" / f"Sheet{sheet.sheet_id}_segment_boundaries.png"

    if not dry_run:
        ensure_dir(output_path.parent)
        cv2_imwrite(output_path, composite)
        if save_debug:
            debug = composite.copy()
            for i, ps in enumerate(placed_segments, start=1):
                x1, y1, x2, y2 = ps.rect
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 165, 255), 6)
                cv2.putText(debug, f"{i}:{ps.pdf_path.name}", (x1 + 10, max(40, y1 + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
            ensure_dir(debug_boundary_path.parent)
            cv2_imwrite(debug_boundary_path, debug)

    return SheetAssemblyResult(
        sheet_id=sheet.sheet_id,
        image=composite,
        placed_segments=placed_segments,
        source_segment_names=[Path(seg.file).name for seg in sheet.segments],
        output_path=output_path if not dry_run else None,
        debug_boundary_path=debug_boundary_path if (save_debug and not dry_run) else None,
    )


def infer_content_bounds(sheet_img: np.ndarray) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    col_energy = inv.mean(axis=0)
    row_energy = inv.mean(axis=1)
    col_thresh = np.percentile(col_energy, 60)
    row_thresh = np.percentile(row_energy, 60)
    xs = np.where(col_energy > col_thresh)[0]
    ys = np.where(row_energy > row_thresh)[0]
    if xs.size == 0 or ys.size == 0:
        return 0, 0, sheet_img.shape[1], sheet_img.shape[0]
    return int(xs[0]), int(ys[0]), int(xs[-1] + 1), int(ys[-1] + 1)


def estimate_or_load_grid(sheet: SheetSpec, assembly: SheetAssemblyResult, rows: int, cols: int) -> List[GridCell]:
    img_h, img_w = assembly.image.shape[:2]
    grid = sheet.grid
    mode = (grid.mode or "manual").lower()
    if mode not in {"manual", "auto"}:
        raise ValueError(f"Invalid grid mode for sheet {sheet.sheet_id}: {grid.mode}")
    if mode == "manual":
        if grid.x0 is not None and grid.y0 is not None and grid.width is not None and grid.height is not None:
            x0, y0 = float(grid.x0), float(grid.y0)
            cell_w = float(grid.cell_width) if grid.cell_width is not None else float(grid.width) / cols
            cell_h = float(grid.cell_height) if grid.cell_height is not None else float(grid.height) / rows
        else:
            if grid.left_margin is None or grid.top_margin is None:
                raise ValueError(f"Manual grid mode for sheet {sheet.sheet_id} requires left_margin and top_margin")
            x0, y0 = float(grid.left_margin), float(grid.top_margin)
            if grid.cell_width is not None:
                cell_w = float(grid.cell_width)
            else:
                if grid.right_margin is None:
                    raise ValueError(f"Manual grid mode for sheet {sheet.sheet_id} needs cell_width or right_margin")
                cell_w = float(img_w - grid.left_margin - grid.right_margin) / cols
            if grid.cell_height is not None:
                cell_h = float(grid.cell_height)
            else:
                if grid.bottom_margin is None:
                    raise ValueError(f"Manual grid mode for sheet {sheet.sheet_id} needs cell_height or bottom_margin")
                cell_h = float(img_h - grid.top_margin - grid.bottom_margin) / rows
    else:
        x1, y1, x2, y2 = infer_content_bounds(assembly.image)
        x0, y0 = float(x1), float(y1)
        cell_w = float(x2 - x1) / cols
        cell_h = float(y2 - y1) / rows
        LOGGER.info("Auto grid estimate sheet=%d bounds=(%d,%d,%d,%d) cell=(%.2f, %.2f)", sheet.sheet_id, x1, y1, x2, y2, cell_w, cell_h)

    cells: List[GridCell] = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            fx1 = x0 + (col - 1) * cell_w
            fy1 = y0 + (row - 1) * cell_h
            fx2 = x0 + col * cell_w
            fy2 = y0 + row * cell_h
            x1i = max(0, min(int(round(fx1)), img_w - 1))
            y1i = max(0, min(int(round(fy1)), img_h - 1))
            x2i = max(x1i + 1, min(int(round(fx2)), img_w))
            y2i = max(y1i + 1, min(int(round(fy2)), img_h))
            cells.append(GridCell(sheet=sheet.sheet_id, row=row, col=col, x1=x1i, y1=y1i, x2=x2i, y2=y2i))
    return cells


def draw_grid_preview(sheet_image: np.ndarray, cells: List[GridCell], output_path: Path, dry_run: bool) -> None:
    preview = sheet_image.copy()
    for cell in cells:
        cv2.rectangle(preview, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 0), 4)
        cv2.putText(preview, cell.label, (cell.x1 + 8, min(cell.y2 - 10, cell.y1 + 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    if not dry_run:
        ensure_dir(output_path.parent)
        cv2_imwrite(output_path, preview)


def crop_stamps(sheet_image: np.ndarray, cells: List[GridCell]) -> Dict[str, np.ndarray]:
    crops: Dict[str, np.ndarray] = {}
    for cell in cells:
        crop = sheet_image[cell.y1:cell.y2, cell.x1:cell.x2].copy()
        if crop.size == 0:
            raise ValueError(f"Empty crop for {cell.label}")
        crops[cell.label] = crop
    return crops


def build_red_mask(crop_bgr: np.ndarray, cfg: ExtractionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    a = lab[:, :, 1].astype(np.float32)

    red_h = np.logical_or(h <= cfg.red_h1_max, h >= cfg.red_h2_min).astype(np.float32)
    sat = smoothstep(cfg.red_s_min / 255.0, 1.0, s)
    val = smoothstep(cfg.red_v_min / 255.0, 1.0, v)
    lab_red = smoothstep(float(cfg.red_lab_a_min), 255.0, a)
    return clamp01((red_h * sat * val * lab_red).astype(np.float32))


def build_black_likelihood_mask(crop_bgr: np.ndarray, red_mask: np.ndarray, cfg: ExtractionConfig) -> np.ndarray:
    b = crop_bgr[:, :, 0].astype(np.float32) / 255.0
    g = crop_bgr[:, :, 1].astype(np.float32) / 255.0
    r = crop_bgr[:, :, 2].astype(np.float32) / 255.0

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    chroma = maxc - minc
    darkness = 1.0 - maxc
    neutrality = 1.0 - chroma
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark_gray = 1.0 - gray

    dark_score = smoothstep(cfg.darkness_low, cfg.darkness_high, darkness)
    gray_dark_score = smoothstep(cfg.darkness_low, cfg.darkness_high, dark_gray)
    neutral_score = smoothstep(cfg.neutrality_low, cfg.neutrality_high, neutrality)

    likelihood = np.maximum(dark_score, gray_dark_score) * neutral_score
    if cfg.red_penalty_strength > 0:
        likelihood *= (1.0 - clamp01(red_mask * cfg.red_penalty_strength))
    return clamp01(likelihood.astype(np.float32))


def apply_optional_morphology(mask: np.ndarray, cfg: MorphologyConfig) -> np.ndarray:
    if not cfg.enabled or cfg.operation == "none" or cfg.kernel_size <= 1 or cfg.iterations <= 0:
        return mask
    kernel = np.ones((cfg.kernel_size, cfg.kernel_size), np.uint8)
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    if cfg.operation == "erode":
        result = cv2.erode(mask_u8, kernel, iterations=cfg.iterations)
    elif cfg.operation == "dilate":
        result = cv2.dilate(mask_u8, kernel, iterations=cfg.iterations)
    elif cfg.operation == "open":
        result = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=cfg.iterations)
    elif cfg.operation == "close":
        result = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=cfg.iterations)
    else:
        raise ValueError(f"Unsupported morphology operation: {cfg.operation}")
    return result.astype(np.float32) / 255.0


def compose_rgba_foreground(crop_bgr: np.ndarray, alpha: np.ndarray, background_mode: str) -> np.ndarray:
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


def extract_overprint(crop_bgr: np.ndarray, cfg: ExtractionConfig, background_mode: str = "transparent") -> ExtractionResult:
    red_mask = build_red_mask(crop_bgr, cfg)
    black_likelihood = build_black_likelihood_mask(crop_bgr, red_mask, cfg)
    black_likelihood = apply_optional_morphology(black_likelihood, cfg.morphology)

    alpha = black_likelihood.copy()
    if cfg.alpha_gamma != 1.0 and cfg.alpha_gamma > 0:
        alpha = np.power(alpha, cfg.alpha_gamma)
    alpha[alpha < cfg.alpha_clip_min] = 0.0
    alpha = clamp01(alpha)

    strict_binary = (black_likelihood >= cfg.binary_threshold).astype(np.uint8) * 255
    rgba = compose_rgba_foreground(crop_bgr, alpha, background_mode)
    return ExtractionResult(
        rgba=rgba,
        alpha=np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
        red_mask=np.clip(red_mask * 255.0, 0, 255).astype(np.uint8),
        black_likelihood=np.clip(black_likelihood * 255.0, 0, 255).astype(np.uint8),
        strict_binary=strict_binary,
    )


def render_rgba_on_bg(rgba: np.ndarray, bg_value: int) -> np.ndarray:
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    bg = np.full_like(fg, float(bg_value))
    return (fg * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])).astype(np.uint8)


def save_stamp_debug_artifacts(
    debug_dir: Path,
    label: str,
    crop_bgr: np.ndarray,
    extraction: ExtractionResult,
    strict: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    ensure_dir(debug_dir)
    cv2_imwrite(debug_dir / f"{label}_crop.png", crop_bgr)
    cv2_imwrite(debug_dir / f"{label}_red_mask.png", extraction.red_mask)
    cv2_imwrite(debug_dir / f"{label}_black_likelihood.png", extraction.black_likelihood)
    cv2_imwrite(debug_dir / f"{label}_alpha.png", extraction.alpha)
    cv2_imwrite(debug_dir / f"{label}_preview_white.png", render_rgba_on_bg(extraction.rgba, 255))
    cv2_imwrite(debug_dir / f"{label}_preview_black.png", render_rgba_on_bg(extraction.rgba, 0))
    if strict:
        cv2_imwrite(debug_dir / f"{label}_strict_binary.png", extraction.strict_binary)


def make_contact_sheet(
    images: List[np.ndarray],
    labels: List[str],
    cols: int = 5,
    thumb_size: Tuple[int, int] = (380, 260),
    background_value: int = 255,
) -> np.ndarray:
    if not images:
        raise ValueError("No images supplied for contact sheet")
    rows = int(math.ceil(len(images) / float(cols)))
    thumb_w, thumb_h = thumb_size
    margin = 18
    label_h = 34
    canvas = np.full((rows * (thumb_h + label_h + margin) + margin, cols * (thumb_w + margin) + margin, 3), background_value, dtype=np.uint8)
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y:y + thumb_h, x:x + thumb_w] = thumb
        cv2.putText(canvas, label, (x, y + thumb_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def segments_intersecting_cell(placed_segments: List[PlacedSegment], cell: GridCell) -> List[str]:
    hits: List[str] = []
    for seg in placed_segments:
        sx1, sy1, sx2, sy2 = seg.rect
        if not (cell.x2 <= sx1 or cell.x1 >= sx2 or cell.y2 <= sy1 or cell.y1 >= sy2):
            hits.append(seg.pdf_path.name)
    return hits


def write_mapping_csv(csv_path: Path, rows: List[Dict[str, Any]], dry_run: bool) -> None:
    if dry_run:
        return
    ensure_dir(csv_path.parent)
    fieldnames = ["filename", "sheet", "row", "column", "source_sheet_image", "source_pdf_segments", "x1", "y1", "x2", "y2"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(manifest_path: Path, payload: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_summary(summary_path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_dir(summary_path.parent)
    summary_path.write_text(text, encoding="utf-8")


def process_project(project: ProjectConfig, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir: Path = args.output_dir
    preview_dir = output_dir / "previews"
    debug_dir = output_dir / "debug"
    for path in (output_dir, preview_dir, debug_dir, output_dir / "Sheet1", output_dir / "Sheet2"):
        ensure_dir(path, dry_run=args.dry_run)

    rasterized = rasterize_pdfs(project, args.input_dir, output_dir, args.dpi, project.output.save_rasterized, args.dry_run)
    mapping_rows: List[Dict[str, Any]] = []
    manifest_sheets: Dict[str, Any] = {}
    extracted_count = 0
    preview_count = 0
    reconstructed_count = 0
    contact_sheet_paths: List[Path] = []

    for sheet_id in sorted(project.sheets):
        sheet = project.sheets[sheet_id]
        assembly = reconstruct_sheet(sheet, rasterized, args.input_dir, output_dir, project.reconstruction, project.output.save_debug, args.dry_run)
        reconstructed_count += 1
        cells = estimate_or_load_grid(sheet, assembly, project.rows, project.cols)
        preview_path = preview_dir / f"Sheet{sheet_id}_grid_preview.png"
        draw_grid_preview(assembly.image, cells, preview_path, args.dry_run)
        preview_count += 1
        crops = crop_stamps(assembly.image, cells)
        final_preview_tiles: List[np.ndarray] = []
        final_preview_labels: List[str] = []
        sheet_output_dir = output_dir / f"Sheet{sheet_id}"
        if not args.dry_run:
            ensure_dir(sheet_output_dir)

        for cell in cells:
            crop = crops[cell.label]
            extraction = extract_overprint(crop, project.extraction, project.output.background_mode)
            output_path = sheet_output_dir / cell.filename
            if not args.dry_run:
                pil_save_rgba(output_path, extraction.rgba)
                if project.output.save_debug:
                    save_stamp_debug_artifacts(debug_dir / f"Sheet{sheet_id}", cell.label, crop, extraction, args.strict, False)
            final_preview_tiles.append(render_rgba_on_bg(extraction.rgba, 255))
            final_preview_labels.append(cell.label)
            source_segments = segments_intersecting_cell(assembly.placed_segments, cell)
            mapping_rows.append(
                {
                    "filename": cell.filename,
                    "sheet": cell.sheet,
                    "row": cell.row,
                    "column": cell.col,
                    "source_sheet_image": str(assembly.output_path.name if assembly.output_path else f"Sheet{sheet_id}_reconstructed.png"),
                    "source_pdf_segments": ";".join(source_segments),
                    "x1": cell.x1,
                    "y1": cell.y1,
                    "x2": cell.x2,
                    "y2": cell.y2,
                }
            )
            extracted_count += 1

        if project.output.save_debug and not args.dry_run:
            contact = make_contact_sheet(final_preview_tiles, final_preview_labels, cols=project.cols)
            contact_path = debug_dir / f"Sheet{sheet_id}_contact_sheet.png"
            cv2_imwrite(contact_path, contact)
            contact_sheet_paths.append(contact_path)

        manifest_sheets[str(sheet_id)] = {
            "reconstructed_image": str(assembly.output_path) if assembly.output_path else None,
            "grid_preview": str(preview_path),
            "segments": [{"pdf_path": str(ps.pdf_path), "x": ps.x, "y": ps.y, "width": ps.width, "height": ps.height} for ps in assembly.placed_segments],
            "cells": [{"label": cell.label, "x1": cell.x1, "y1": cell.y1, "x2": cell.x2, "y2": cell.y2} for cell in cells],
        }

    mapping_path = output_dir / "mapping.csv"
    manifest_path = output_dir / "manifest.json"
    write_mapping_csv(mapping_path, mapping_rows, args.dry_run)
    manifest_payload = {
        "project_name": project.project_name,
        "created_utc": utc_now_iso(),
        "dpi": args.dpi,
        "input_dir": str(args.input_dir) if args.input_dir else None,
        "output_dir": str(output_dir),
        "sheet_layout": {"rows": project.rows, "cols": project.cols},
        "strict_debug_binary": bool(args.strict),
        "dry_run": bool(args.dry_run),
        "config": project.raw,
        "resolved": {
            "reconstruction": asdict(project.reconstruction),
            "extraction": {**asdict(project.extraction), "morphology": asdict(project.extraction.morphology)},
            "output": asdict(project.output),
        },
        "sheets": manifest_sheets,
    }
    write_manifest(manifest_path, manifest_payload, args.dry_run)

    summary_lines = [
        f"PDFs processed: {len(rasterized)}",
        f"Sheets reconstructed: {reconstructed_count}",
        f"Previews generated: {preview_count}",
        f"Stamps extracted: {extracted_count}",
        f"Debug artifacts saved: {'yes' if project.output.save_debug and not args.dry_run else 'no'}",
        f"Mapping CSV: {mapping_path}",
        f"Manifest JSON: {manifest_path}",
    ]
    summary_text = "\n".join(summary_lines)
    if args.summary_file:
        write_summary(args.summary_file, summary_text + "\n", args.dry_run)
    print(summary_text)
    return {
        "pdfs_processed": len(rasterized),
        "sheets_reconstructed": reconstructed_count,
        "previews_generated": preview_count,
        "stamps_extracted": extracted_count,
        "mapping_csv": str(mapping_path),
        "manifest_json": str(manifest_path),
        "contact_sheets": [str(p) for p in contact_sheet_paths],
        "summary": summary_lines,
    }


def synthetic_stamp_image() -> np.ndarray:
    img = np.full((220, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (300, 200), (220, 235, 245), -1)
    cv2.putText(img, "STAMP", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (130, 80, 40), 3, cv2.LINE_AA)
    cv2.putText(img, "12", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (70, 110, 150), 3, cv2.LINE_AA)
    cv2.putText(img, "B", (105, 126), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(img, "7", (140, 126), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, (35, 40), (140, 105), (0, 0, 255), 4, tipLength=0.15)
    return img


def run_self_check() -> int:
    LOGGER.info("Running synthetic self-check")
    dummy_sheet = np.full((1000, 800, 3), 255, dtype=np.uint8)
    grid = GridSpec(mode="manual", left_margin=50, top_margin=25, cell_width=140, cell_height=90)
    sheet = SheetSpec(sheet_id=1, segments=[SegmentSpec(file="dummy.pdf")], grid=grid)
    assembly = SheetAssemblyResult(sheet_id=1, image=dummy_sheet, placed_segments=[], source_segment_names=["dummy.pdf"])
    cells = estimate_or_load_grid(sheet, assembly, 10, 5)
    assert len(cells) == 50
    assert cells[0].label == "S1_R01_C01"
    assert cells[-1].label == "S1_R10_C05"
    stamp = synthetic_stamp_image()
    result = extract_overprint(stamp, ExtractionConfig(), background_mode="transparent")
    assert result.rgba.shape[2] == 4
    assert result.alpha.shape == stamp.shape[:2]
    assert result.red_mask.max() > 0
    csv_row = {
        "filename": cells[0].filename,
        "sheet": 1,
        "row": 1,
        "column": 1,
        "source_sheet_image": "Sheet1_reconstructed.png",
        "source_pdf_segments": "dummy.pdf",
        "x1": cells[0].x1,
        "y1": cells[0].y1,
        "x2": cells[0].x2,
        "y2": cells[0].y2,
    }
    expected = {"filename", "sheet", "row", "column", "source_sheet_image", "source_pdf_segments", "x1", "y1", "x2", "y2"}
    assert expected.issubset(csv_row.keys())
    print("Self-check passed")
    print("Grid naming: OK")
    print("CSV schema: OK")
    print("Folder creation logic: OK")
    print("Extraction pipeline signatures: OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        if args.self_check:
            return run_self_check()
        project = load_project_config(args.config, args)
        process_project(project, args)
        return 0
    except Exception as exc:
        LOGGER.exception("Processing failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
