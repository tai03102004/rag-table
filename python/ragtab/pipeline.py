from PIL import Image
from typing import List
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

from .utils import Cell, ocr
from .ocr import crop_and_ocr
from .detection import detect_header_cells_via_ocr, masks_to_cell_boxes
from .model_loader import load_model
from .drop import drop_empty_edge_columns, drop_empty_rows, drop_footer_rows
def cells_to_markdown(cells):
    if not cells:
        return ""
    max_row = max(c.row_idx for c in cells) + 1
    max_col = max(c.col_idx + c.col_span for c in cells)
    grid = [[None] * max_col for _ in range(max_row)]
    SPAN_MARKER = "__SPAN__"

    for cell in sorted(cells, key=lambda c: (c.row_idx, c.col_idx)):
        grid[cell.row_idx][cell.col_idx] = cell.text
        for off in range(1, cell.col_span):
            if cell.col_idx + off < max_col:
                grid[cell.row_idx][cell.col_idx + off] = SPAN_MARKER

    lines = []
    for r, row in enumerate(grid):
        cells_str = [("" if v in (None, SPAN_MARKER) else v) for v in row]
        lines.append("| " + " | ".join(cells_str) + " |")
        if r == 0:
            lines.append("| " + " | ".join(["---"] * max_col) + " |")
    return "\n".join(lines)
# ── Pipeline hoàn chỉnh ───────────────────────────────────
def image_to_markdown_v3(image_path, model, device, img_size=384, upscale=3, verbose=True):
    """Run full extraction pipeline with progress bar."""

    stages = [
        ("Loading image",        1),
        ("Segmentation",         2),
        ("Detecting structure",  1),
        ("OCR header rows",      2),
        ("OCR data cells",       5),  
        ("Post-processing",      1),
    ]
    total = sum(w for _, w in stages)

    pbar = tqdm(
        total=total,
        desc="Extracting table",
        disable=not verbose,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
        ncols=80,
    )
    
    def step(name, weight):
        pbar.set_description(f"▸ {name}")
        pbar.update(weight)

    # ── 1. Load image ──────────────────────────────────────
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    step("Loading image", 1)

    # ── 2. Segmentation ────────────────────────────────────
    draft = TF.resize(orig_img, (img_size, img_size),
                    interpolation=TF.InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(draft).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                preds = torch.sigmoid(model(img_t)).squeeze(0).float().cpu().numpy()
        else:
            preds = torch.sigmoid(model(img_t)).squeeze(0).cpu().numpy()

    row_mask        = (preds[0] > 0.5).astype(np.uint8)
    col_mask        = (preds[1] > 0.5).astype(np.uint8)
    col_header_mask = (preds[2] > 0.5).astype(np.uint8)
    row_header_mask = (preds[3] > 0.5).astype(np.uint8)
    span_mask       = (preds[4] > 0.5).astype(np.uint8)
    step("Segmentation", 2)

    # ── 3. Detect structure ────────────────────────────────
    cells, row_sep, col_sep = masks_to_cell_boxes(
        row_mask, col_mask, span_mask,
        col_header_mask, row_header_mask,
        orig_w, orig_h, img_size
    )

    num_rows_total = len(row_sep) - 1
    header_row_indices = []
    for r in range(num_rows_total):
        y1, y2 = row_sep[r], row_sep[r + 1]
        region = col_header_mask[y1:y2]
        if region.size > 0 and region.mean() > 0.5:
            header_row_indices.append(r)
    step("Detecting structure", 1)

    # ── 4. Header OCR ──────────────────────────────────────
    if header_row_indices:
        header_cells = detect_header_cells_via_ocr(
            orig_img, row_sep, col_sep, header_row_indices,
            ocr, orig_w, orig_h, img_size=img_size
        )
        cells = [c for c in cells if c.row_idx not in header_row_indices]
        cells.extend(header_cells)

    step("OCR header rows", 2)

    # ── 5. Data cells OCR ──────────────────────────────────
    cells = crop_and_ocr(orig_img, cells, ocr, upscale=upscale)
    step("OCR data cells", 5)

    # ── 6. Post-processing ─────────────────────────────────
    cells = drop_empty_edge_columns(cells) 
    cells = drop_footer_rows(cells)         
    cells = drop_empty_rows(cells)
    cells.sort(key=lambda c: (c.row_idx, c.col_idx))
    md = cells_to_markdown(cells)
    step("Post-processing", 1)

    pbar.set_description("✅ Done")
    pbar.close()

    return md, cells
    

def extract_table(image_path, model_path=None, ocr_engine="paddleocr", verbose=True):
    """
    Extract table from image to Markdown.
    
    Args:
        image_path: Path to table image (PNG/JPG/etc).
        model_path: Optional custom checkpoint path. Auto-downloads from 
                    HuggingFace if None.
        ocr_engine: OCR engine to use (currently only 'paddleocr').
        verbose: Show progress bar.
    
    Returns:
        (markdown: str, cells: List[Cell])
    """
    model, device = load_model(model_path=model_path)
    return image_to_markdown_v3(
        image_path, model, device, 
        img_size=384, upscale=3, verbose=verbose
    )