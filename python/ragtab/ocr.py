import math
import re
import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Pre-compiled regex
# ═══════════════════════════════════════════════════════════════
_RE_WHITESPACE    = re.compile(r"\s+")
_RE_BADCHARS      = re.compile(r"[^\w\s\.\,\-\(\)%±/=+:;|&'<>~]")
_RE_PERCENT_TAIL  = re.compile(r"%\d+")
_RE_PERCENT_96    = re.compile(r"(\d+\.\d)96\b")
_RE_PERCENT_VALUE = re.compile(r"(\d+\.\d+)%")
_RE_PM_SPACING    = re.compile(r"(\d)±\s*(\d)")


# ═══════════════════════════════════════════════════════════════
# Text cleaning 
# ═══════════════════════════════════════════════════════════════
def _normalize_percent_value(match):
    val = float(match.group(1))
    if val < 0.1:
        # Truncate (không round) đến 2 chữ số thập phân
        val = math.floor(val * 100) / 100
        return f"{val:.2f}%"
    return f"{val:.1f}%"


def clean_text(text, col_idx=None):
    if not text:
        return ""
    
    text = _RE_WHITESPACE.sub(" ", text.strip())
    text = _RE_BADCHARS.sub("", text)
    
    text = _RE_PERCENT_TAIL.sub("%", text)
    text = _RE_PERCENT_96.sub(r"\1%", text)
    text = _RE_PERCENT_VALUE.sub(_normalize_percent_value, text)
    text = _RE_PM_SPACING.sub(r"\1 ± \2", text)
    
    return text


# ═══════════════════════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════════════════════
def resize_with_padding(img, target_height=64, target_width=256):
    """Resize giữ tỉ lệ, pad trắng để đạt kích thước mục tiêu."""
    h, w = img.shape[:2]
    ratio = min(target_width / w, target_height / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


# ═══════════════════════════════════════════════════════════════
# Main OCR function 
# ═══════════════════════════════════════════════════════════════
def crop_and_ocr(image_pil, cells, ocr, upscale=3):
    img_np = np.array(image_pil.convert("RGB"))
    H, W = img_np.shape[:2]

    batch_imgs = []
    valid_cells = []

    for cell in cells:
        if cell.text:
            continue

        # Crop với padding
        pad = max(int(min(cell.w, cell.h) * 0.15), 3)
        x1 = max(0, cell.x - pad)
        y1 = max(0, cell.y - pad)
        x2 = min(W, cell.x + cell.w + pad)
        y2 = min(H, cell.y + cell.h + pad)

        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            cell.text = ""
            continue

        # Convert grayscale + adaptive scale
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        h_ratio = max(0.3, min(1.0, cell.h / 100.0))
        scale = 3.0 + (1 - h_ratio) * 2.0   # min=3.0, max=5.0

        new_w = int(gray.shape[1] * scale)
        new_h = int(gray.shape[0] * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Convert + pad to fixed OCR input size
        img_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_input = resize_with_padding(img_input, target_height=64, target_width=256)

        batch_imgs.append(img_input)
        valid_cells.append(cell)

    # Batch OCR
    results = ocr.predict(batch_imgs)

    # Assign text với cleaning
    for cell, result in zip(valid_cells, results):
        texts = []
        if result and isinstance(result, dict) and "rec_texts" in result:
            texts.extend(result["rec_texts"])
        
        cell.text = clean_text(" ".join(t for t in texts if t))
    
    return cells