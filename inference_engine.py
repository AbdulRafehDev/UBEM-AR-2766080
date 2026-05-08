from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

DEFAULT_MODEL_PATH = "yolov8m_seg.pt"
DEFAULT_FACADE_ID = 2
DEFAULT_WINDOW_ID = 8

# OpenCV uses BGR channel ordering.
FACADE_COLOR_BGR = (255, 0, 0)  # Blue
WINDOW_COLOR_BGR = (0, 255, 0)  # Green

# SegFormer fallback config
SEGFORMER_MODEL_ID = "Xpitfire/segformer-finetuned-segments-cmp-facade"
SEGFORMER_ANCHOR_LABELS = ["window", "door", "cornice", "sill", "balcony", "molding", "deco", "pillar"]
SEGFORMER_EXCLUDE_LABELS = ["unknown", "background"]
SEGFORMER_DILATION_RATIO = 0.18  # fraction of image size for anchor dilation

# Lazy-loaded globals so model is only downloaded/loaded once
_segformer_processor = None
_segformer_model = None


def _load_segformer():
    """Load SegFormer model once and cache globally."""
    global _segformer_processor, _segformer_model
    if _segformer_processor is None:
        print("[FALLBACK] Loading SegFormer facade model...")
        _segformer_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID)
        _segformer_model.eval()
        print("[FALLBACK] SegFormer loaded.")
    return _segformer_processor, _segformer_model


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """Load and return the YOLO model once, then reuse it."""
    return YOLO(model_path)


def _extract_classes(result) -> np.ndarray:
    if result.boxes is None or result.boxes.cls is None:
        return np.array([], dtype=int)
    return result.boxes.cls.cpu().numpy().astype(int)


def _normalize_masks(masks: Optional[np.ndarray], target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize masks to the image shape if the model output shape differs."""
    if masks is None or len(masks) == 0:
        return np.empty((0, target_shape[0], target_shape[1]), dtype=bool)

    h, w = target_shape
    normalized = []
    for mask in masks:
        if mask.shape != (h, w):
            resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            normalized.append(resized.astype(bool))
        else:
            normalized.append(mask.astype(bool))
    return np.asarray(normalized, dtype=bool)


def get_facade_data_segformer(image_path: str, image_shape: Tuple[int, int]):
    """
    SegFormer-based facade fallback.

    Returns:
    - facade_mask: boolean numpy array (H, W) of refined facade pixels
    - exclude_mask: boolean numpy array (H, W) — empty, SegFormer handles exclusion internally
    """
    h, w = image_shape
    processor, seg_model = _load_segformer()

    pil_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = seg_model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().numpy()

    label2id = {v: k for k, v in seg_model.config.id2label.items()}

    if "facade" not in label2id:
        print("[FALLBACK] SegFormer: 'facade' label not found in model output.")
        return np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)

    facade_mask = seg_map == label2id["facade"]

    # Build exclusion mask
    exclusion_mask = np.zeros((h, w), dtype=bool)
    for label in SEGFORMER_EXCLUDE_LABELS:
        if label in label2id:
            exclusion_mask |= (seg_map == label2id[label])

    # Build anchor mask from architectural elements
    anchor_mask = np.zeros((h, w), dtype=bool)
    for label in SEGFORMER_ANCHOR_LABELS:
        if label in label2id:
            anchor_mask |= (seg_map == label2id[label])

    # Dilate anchor mask to create valid facade zone
    dilation_px = max(1, int(min(h, w) * SEGFORMER_DILATION_RATIO))
    kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
    anchor_zone = cv2.dilate(anchor_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    # Refined facade: facade class + near architectural elements + not excluded
    refined_facade = facade_mask & anchor_zone & ~exclusion_mask

    # Vertical constraint: remove facade below the lowest anchor pixel
    if np.any(anchor_mask):
        lowest_anchor_row = int(np.max(np.where(anchor_mask)[0]))
        refined_facade[lowest_anchor_row:, :] = False

    print(f"[FALLBACK] SegFormer raw facade pixels:     {int(np.sum(facade_mask))}")
    print(f"[FALLBACK] SegFormer refined facade pixels: {int(np.sum(refined_facade))}")
    print(f"[FALLBACK] SegFormer facade coverage:       {100 * np.sum(refined_facade) / refined_facade.size:.1f}%")

    # Return empty exclude mask — refinement already handled exclusion
    return refined_facade, np.zeros((h, w), dtype=bool)


def compute_wwr_from_masks(
    masks: np.ndarray,
    classes: np.ndarray,
    facade_id: int = DEFAULT_FACADE_ID,
    window_id: int = DEFAULT_WINDOW_ID,
    exclude_ids: Optional[Iterable[int]] = None,
    facade_mask_override: Optional[np.ndarray] = None,
    exclude_mask_override: Optional[np.ndarray] = None,
) -> float:
    """Compute WWR as valid window area divided by facade area."""
    if masks is None or len(masks) == 0 or len(classes) == 0:
        return 0.0

    exclude_set: Set[int] = set(exclude_ids or [])
    count = min(len(masks), len(classes))
    masks = masks[:count]
    classes = classes[:count]

    if facade_mask_override is not None:
        facade_mask = facade_mask_override
        facade_support = binary_fill_holes(facade_mask).astype(bool)
        dilation_px = max(1, int(round(min(facade_support.shape) * 0.003)))
        kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
        facade_support = cv2.dilate(facade_support.astype(np.uint8), kernel, iterations=1).astype(bool)
        facade_area = int(np.sum(facade_support))
        if facade_area == 0:
            return 0.0
    else:
        facade_mask = None
        for idx, cls in enumerate(classes):
            cls_int = int(cls)
            current_mask = masks[idx]
            if cls_int == facade_id:
                facade_mask = current_mask.copy() if facade_mask is None else (facade_mask | current_mask)

        if facade_mask is None:
            return 0.0

        facade_support = binary_fill_holes(facade_mask).astype(bool)
        dilation_px = max(1, int(round(min(facade_support.shape) * 0.003)))
        kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
        facade_support = cv2.dilate(facade_support.astype(np.uint8), kernel, iterations=1).astype(bool)
        facade_area = int(np.sum(facade_support))
        if facade_area == 0:
            return 0.0

    if exclude_mask_override is not None:
        exclude_mask = exclude_mask_override
    else:
        exclude_mask = None
        for idx, cls in enumerate(classes):
            cls_int = int(cls)
            current_mask = masks[idx]
            if cls_int in exclude_set:
                exclude_mask = current_mask.copy() if exclude_mask is None else (exclude_mask | current_mask)

        if exclude_mask is None:
            exclude_mask = np.zeros_like(facade_mask, dtype=bool)

    window_masks = []
    for idx, cls in enumerate(classes):
        cls_int = int(cls)
        current_mask = masks[idx]
        if cls_int == window_id:
            window_masks.append(current_mask)

    valid_window_area = 0
    for window_mask in window_masks:
        window_area = int(np.sum(window_mask))
        if window_area == 0:
            continue

        exclude_overlap = float(np.sum(window_mask & exclude_mask)) / float(window_area)
        if exclude_overlap > 0.4:
            continue

        facade_overlap = float(np.sum(window_mask & facade_support)) / float(window_area)
        if facade_overlap >= 0.5:
            valid_window_area += window_area

    return float(valid_window_area) / float(facade_area)


def annotate_image_with_colors(
    image: np.ndarray,
    masks: np.ndarray,
    classes: np.ndarray,
    facade_id: int = DEFAULT_FACADE_ID,
    window_id: int = DEFAULT_WINDOW_ID,
    facade_color: Tuple[int, int, int] = FACADE_COLOR_BGR,
    window_color: Tuple[int, int, int] = WINDOW_COLOR_BGR,
    alpha: float = 0.45,
    facade_mask_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlay facade/window segmentation masks with explicit BGR colors."""
    if image is None:
        raise ValueError("image cannot be None")
    if masks is None or len(masks) == 0 or len(classes) == 0:
        return image.copy()

    count = min(len(masks), len(classes))
    masks = masks[:count]
    classes = classes[:count]

    overlay = image.copy()

    # Paint facade first, then windows so windows remain clearly visible.
    if facade_mask_override is not None:
        overlay[facade_mask_override] = facade_color
    else:
        for idx, cls in enumerate(classes):
            if int(cls) == facade_id:
                overlay[masks[idx]] = facade_color

    for idx, cls in enumerate(classes):
        if int(cls) == window_id:
            overlay[masks[idx]] = window_color

    annotated = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    # Draw mask contours for cleaner visual boundaries.
    if facade_mask_override is not None:
        contours, _ = cv2.findContours(
            facade_mask_override.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, facade_color, 2)

    for idx, cls in enumerate(classes):
        class_id = int(cls)
        if class_id not in {facade_id, window_id}:
            continue
        if facade_mask_override is not None and class_id == facade_id:
            continue  # already drawn
        contour_color = facade_color if class_id == facade_id else window_color
        contours, _ = cv2.findContours(
            masks[idx].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, contour_color, 2)

    return annotated


def run_inference_on_image(
    image_path: str,
    model: YOLO,
    output_path: Optional[str] = None,
    facade_id: int = DEFAULT_FACADE_ID,
    window_id: int = DEFAULT_WINDOW_ID,
    exclude_ids: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    """Run model inference, compute WWR, and optionally save custom annotation."""
    results = model.predict(
        source=image_path,
        save=False,
        classes=[facade_id, window_id],
        verbose=False,
    )
    if not results:
        raise RuntimeError(f"No inference result returned for image: {image_path}")

    result = results[0]
    classes = _extract_classes(result)

    class_counts: Dict[str, int] = {}
    if len(classes) > 0:
        unique, counts = np.unique(classes, return_counts=True)
        class_counts = {model.names[int(k)]: int(v) for k, v in zip(unique, counts)}

    image = cv2.imread(image_path)
    if image is None and hasattr(result, "orig_img") and result.orig_img is not None:
        image = result.orig_img.copy()
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    raw_masks = None
    if result.masks is not None and result.masks.data is not None:
        raw_masks = result.masks.data.cpu().numpy() > 0.5
    masks = _normalize_masks(raw_masks, target_shape=image.shape[:2])

    has_facade = any(cls == facade_id for cls in classes)
    facade_mask_override = None
    exclude_mask_override = None

    if not has_facade:
        print(f"[FALLBACK] No facade detected by local model for {Path(image_path).name} — using SegFormer.")
        facade_mask_override, exclude_mask_override = get_facade_data_segformer(
            image_path, image.shape[:2]
        )
        print(f"[FALLBACK] Facade mask pixels: {int(np.sum(facade_mask_override))}")
        print(f"[FALLBACK] Exclude mask pixels: {int(np.sum(exclude_mask_override))}")

    wwr = compute_wwr_from_masks(
        masks=masks,
        classes=classes,
        facade_id=facade_id,
        window_id=window_id,
        exclude_ids=exclude_ids,
        facade_mask_override=facade_mask_override,
        exclude_mask_override=exclude_mask_override,
    )

    annotated = annotate_image_with_colors(
        image=image,
        masks=masks,
        classes=classes,
        facade_id=facade_id,
        window_id=window_id,
        facade_mask_override=facade_mask_override,
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), annotated)

    return {
        "image_name": Path(image_path).name,
        "class_counts": class_counts,
        "wwr": wwr,
        "annotated_image": annotated,
    }


if __name__ == "__main__":
    import glob

    output_dir = Path("pretrained/results_finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(DEFAULT_MODEL_PATH)
    for img_file in glob.glob("all_images_90/*.jpg"):
        output_path = output_dir / Path(img_file).name
        inference = run_inference_on_image(
            image_path=img_file,
            model=model,
            output_path=str(output_path),
        )
        print(f"Detections for {inference['image_name']}: {inference['class_counts']}")
        print(f"WWR for {inference['image_name']}: {inference['wwr']:.4f}")