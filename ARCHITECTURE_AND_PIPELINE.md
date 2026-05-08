# Energy+ AI: Complete Architecture and Pipeline Guide

**Last Updated**: April 2026  
**Project Status**: Production (v5 inference engine, Flask web app, EnergyPlus integration)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Architecture Layers](#architecture-layers)
4. [Web Application Flow](#web-application-flow)
5. [Computer Vision Pipeline](#computer-vision-pipeline)
6. [Energy Simulation Pipeline](#energy-simulation-pipeline)
7. [Model Configuration](#model-configuration)
8. [Running the Application](#running-the-application)
9. [Known Issues and Maintenance Notes](#known-issues-and-maintenance-notes)

---

## Project Overview

**Energy+ AI** is a full-stack web application that predicts building energy consumption by:
1. **Vision**: Detecting windows and facades in building images using YOLO segmentation
2. **Geometry**: Calculating Window-to-Wall Ratio (WWR) from segmentation masks
3. **Simulation**: Generating EnergyPlus models and running annual energy simulations
4. **Comparison**: Comparing AI-predicted WWR vs. 20% baseline

The system supports:
- Multiple image inputs with averaging
- Street View image fetching via Google Maps API
- Building footprint retrieval from OpenStreetMap
- Parametric EnergyPlus IDF generation
- Web-based user interface with image cropping

---

## File Structure

```
.
├── app.py                    # Flask web application (routes, request handling)
├── inference_engine.py       # Core CV pipeline (YOLO segmentation, WWR computation)
├── generalization.py         # Full energy pipeline orchestration
├── address_pipeline.py       # [DEPRECATED] Duplicate of generalization.py — DELETE
├── requirements.txt          # Python dependencies
├── V5_PIPELINE_DOCUMENTATION.md  # Original v5 documentation
├── ARCHITECTURE_AND_PIPELINE.md  # This file (updated comprehensive guide)
│
├── Model Files
├── final_model.pt           # Trained YOLO model (facade + window classes)
├── yolov8m_seg.pt           # Base YOLO weights (reference)
│
├── EnergyPlus Files
├── V24-1-0-Energy+.idd      # EnergyPlus Input Data Dictionary
├── GBR_ENG_London-Heathrow.Intl.AP.037720_TMYx.2007-2021.epw  # Weather file (London)
│
├── Web Interface
├── templates/
│   └── index.html           # Main web page
├── static/
│   ├── script.js            # Frontend logic
│   ├── styles.css           # Styling
│   ├── uploads/             # User-uploaded and cropped images
│   ├── annotated/           # Inference output annotations
│   └── generated_assets/    # Generated IDFs, simulations, results
│
├── Testing
├── tests/
│   ├── conftest.py          # Pytest fixtures with dependency mocking
│   ├── test_app_routes.py   # Flask route tests
│   └── test_app_helpers.py  # Helper function tests
├── pytest.ini               # Test configuration
│
└── Notebooks
    └── geodecode_injection.ipynb  # Geospatial analysis/exploration
```

---

## Architecture Layers

### Layer 1: Web Interface (`app.py`)
**Role**: HTTP request handling, image upload/crop management, user workflows

**Key Components**:
- `_save_uploaded_images()` — Handle multipart file uploads
- `_save_cropped_image()` — Decode and save base64 cropped images
- `_process_image()` — Run inference on a single image
- Route handlers:
  - `GET/POST /` — Main page with upload/address forms
  - `POST /fetch-street-view` — Fetch and return Street View image
  - `POST /run-pipeline` — Execute full energy pipeline

**Configuration**:
- Building type → max images: `{'mid-terrace': 2, 'end-semi': 3, 'detached': 4}`
- Absolute ceiling: 4 images
- Directory structure: `uploads/`, `annotated/`, `generated_assets/`

### Layer 2: Pipeline Orchestration (`generalization.py`)
**Role**: Coordinate geospatial fetch, inference, IDF generation, and simulation

**Key Functions**:
1. `fetch_building_footprint(address, google_api_key)` — OSM → GeoDataFrame
2. `fetch_street_view(address, output_filename, api_key)` — Google Street View
3. `build_eppy_idf(gdf, wwr, idd_path, output_filename)` — Parametric IDF generation
4. `run_energyplus_simulation(idf_path, output_dir)` — Execute simulation
5. `extract_energy_consumption(output_dir)` — Parse CSV results
6. `run_generation_pipeline(...)` — Orchestrate all steps above

**Multi-Image Handling**:
- Processes each image independently
- Averages WWR across all images
- Stores image pairs: `[{"image_url": ..., "annotated_url": ...}, ...]`
- Backward compatibility: first pair → `image_url` / `annotated_url` fields

**Output Structure**:
```python
{
    "success": bool,
    "address": str,
    "wwr_ai": float,            # Predicted WWR (%)
    "wwr_baseline": float,       # Baseline 20% WWR
    "image_pairs": list,         # List of {image_url, annotated_url}
    "image_url": str,            # First image (backward compat)
    "annotated_url": str,        # First annotated (backward compat)
    "energy_ai": dict,           # {heating_kwh, cooling_kwh, total_kwh}
    "energy_baseline": dict,     # Same for baseline scenario
    "class_counts": dict,        # Aggregated {class_name: count}
    "error": str or None,        # Error message if failed
}
```

### Layer 3: Computer Vision (`inference_engine.py`)
**Role**: YOLO segmentation, fallback SegFormer, WWR computation, visualization

**Key Functions**:
- `load_model(model_path)` — Load YOLO once (reused)
- `run_inference_on_image(image_path, model, output_path)` — Full inference pipeline
- `compute_wwr_from_masks(masks, classes, facade_id, window_id, ...)` — WWR metric
- `annotate_image_with_colors(image, masks, classes, ...)` — Visualization
- `get_facade_data_segformer(image_path, image_shape)` — Fallback segmentation
- `_extract_classes(result)` — Parse class IDs from YOLO result
- `_normalize_masks(masks, target_shape)` — Align masks to image shape
- `_load_segformer()` — Lazy-load SegFormer once

**Inference Details**:
- Predicts only facade (class 2) and window (class 8)
- Returns segmentation masks, class IDs, and instance counts
- Falls back to SegFormer if no facade detected in YOLO output
- Computes WWR using strict overlap validation

---

## Web Application Flow

### Flow 1: Address Only (Legacy)
```
POST / with address="123 Main St"
  → run_generation_pipeline(address, no images)
  → Fetch Street View → Infer → Pipeline → Return results
```

### Flow 2: Images Only
```
POST / with images=[file1, file2, ...]
  → No address or pipeline
  → Just run inference on each image
  → Return WWR and annotations (no energy simulation)
```

### Flow 3: Address + Images (Primary)
```
POST / or /run-pipeline with address="123 Main St" + images=[file1, file2, ...]
  → Validate building type and image count
  → run_generation_pipeline(address, provided_image_paths)
  → Use provided images instead of Street View
  → Average WWR across images
  → Generate 2 IDFs (AI WWR + 20% baseline)
  → Run EnergyPlus for both
  → Return complete comparison
```

### Flow 4: Interactive Street View Crop
```
POST /fetch-street-view with address
  → Fetch Street View image
  → Return image_url to browser
  → User crops in UI (canvas → base64)
  → POST /run-pipeline with cropped_images data URLs
  → Backend saves cropped images
  → Proceed as Flow 3
```

---

## Computer Vision Pipeline

### 5.1 Model Architecture
- **Base Model**: YOLOv8m-seg (medium segmentation variant)
- **Training**: Fine-tuned on custom facade/window dataset
- **Deployment**: `final_model.pt` (serialized YOLO weights)

### 5.2 Class Setup
| Class ID | Class Name | RGB Color (CV: BGR) | Purpose |
|----------|------------|-------------------|---------|
| 2 | Facade | Blue (255, 0, 0) | Building exterior wall |
| 8 | Window | Green (0, 255, 0) | Window openings |

### 5.3 Inference Steps

**Step 1: Load Model Once**
```python
model = load_model("final_model.pt")  # Reused across requests
```

**Step 2: Run Prediction**
```python
results = model.predict(
    source=image_path,
    save=False,
    classes=[2, 8],  # Facade and window only
    verbose=False,
)
```
- Restricts detections to classes 2 and 8
- Returns segmentation masks and bounding boxes

**Step 3: Read Source Image**
```python
image = cv2.imread(image_path)
if image is None:
    image = result.orig_img.copy()  # Fallback from YOLO result
```

**Step 4: Normalize Masks**
- Threshold at 0.5 probability
- Resize to exact image shape if needed (nearest-neighbor)
- Return boolean arrays

**Step 5: Facade Detection Check**
```python
has_facade = any(cls == 2 for cls in classes)
```
- If YES → use YOLO facade mask
- If NO → fallback to SegFormer

**Step 6: SegFormer Fallback (if needed)**
- Model: `Xpitfire/segformer-finetuned-segments-cmp-facade`
- Lazy-loaded and cached globally
- Extracts facade pixels and architectural anchor points
- Applies dilation and vertical constraint
- Returns refined facade mask

### 5.4 WWR Computation

$$WWR = \frac{\text{Valid Window Area}}{\text{Facade Support Area}}$$

**Facade Support Computation**:
1. Union all YOLO facade masks (or use SegFormer override)
2. Fill internal holes (`binary_fill_holes`)
3. Dilate slightly: `kernel_size = max(1, round(min(H,W) * 0.003))`
4. Result = `facade_support` (used for overlap checks)

**Window Validity Checks** (both must pass):
1. **Exclude Overlap**: `area(window ∩ exclude_mask) / area(window) ≤ 0.4`
2. **Facade Overlap**: `area(window ∩ facade_support) / area(window) ≥ 0.5`

Only windows passing both checks contribute to `valid_window_area`.

**Edge Cases**:
- No masks → return 0.0
- No facade found → return 0.0
- Zero facade area → return 0.0
- Zero window area → return 0.0

### 5.5 Annotation Generation
```python
overlay = image.copy()
# Paint facade (blue, semi-transparent)
overlay[facade_mask] = (255, 0, 0)
# Paint windows (green, semi-transparent)
overlay[window_mask] = (0, 255, 0)
# Blend with original
annotated = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
# Draw contours for clarity
cv2.drawContours(annotated, contours, -1, color, thickness=2)
```

---

## Energy Simulation Pipeline

### 6.1 Building Geometry
- **Source**: OpenStreetMap building footprints (via osmnx)
- **Projection**: Local UTM projection (from geographic coordinates)
- **Vertices**: Building outline vertices (normalized to local origin)
- **Height**: Fixed at 9.0 meters (representative mid-rise residential)

### 6.2 IDF Generation (via eppy)

**Objects Created**:
1. **VERSION**: EnergyPlus 24.1
2. **BUILDING**: Name, north axis, terrain class
3. **GLOBALGEOMETRYRULES**: Vertex ordering, coordinate system
4. **MATERIALS**: Exterior wall, roof, floor, window glass
5. **CONSTRUCTIONS**: Exterior wall, roof, floor, window assemblies
6. **SCHEDULES**: Temperature setpoints, on/off schedules
7. **ZONE**: Single thermal zone for simplified building
8. **SURFACES**: Walls, roof, floor (from building footprint)
9. **WINDOWS**: One per wall, parametrized by WWR
10. **HVAC**: Ideal loads air system (simplified)
11. **OUTPUT:VARIABLE**: Heating and cooling energy (annual)

**Parametric Window Generation**:
- Window area varies with WWR
- Centered on each wall
- Vertices computed to match specified WWR ratio

### 6.3 EnergyPlus Simulation
```bash
energyplus.exe \
  --weather {epw_file} \
  --output-directory {output_dir} \
  --readvars \
  {idf_path}
```
- **Weather**: London Heathrow (Intl AP 037720, 2007-2021 TMYx)
- **Timeout**: 300 seconds per simulation
- **Output**: CSV file with hourly/annual energy values

### 6.4 Energy Extraction
- Parses `eplusout.csv` produced by `--readvars`
- Sums columns matching: `("heating" AND "energy")` or `("cooling" AND "energy")`
- Converts Joules → kWh: `kWh = J / 3,600,000`
- Returns: `{heating_kwh, cooling_kwh, total_kwh}`

---

## Model Configuration

### Training Setup (Reference)
```python
model = YOLO("yolov8m-seg.pt")
model.train(
    data="data.yaml",        # Custom dataset
    epochs=50,               # Upper bound
    imgsz=640,               # Input resolution
    batch=8,                 # Batch size
    lr0=1e-5,                # Conservative fine-tune LR
    device=0,                # GPU 0
    patience=15,             # Early stopping
    dropout=0.1,             # Regularization
)
# Result: final_model.pt
```

### Inference Hyperparameters
- **Confidence threshold**: Default YOLO (typically 0.25)
- **NMS IoU threshold**: Default YOLO (typically 0.45)
- **Class filter**: Classes [2, 8] only
- **Mask threshold**: > 0.5 probability

---

## Running the Application

### 6.1 Environment Setup
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 6.2 Configuration
Create `.env` file:
```
GOOGLE_MAPS_API_KEY=your_api_key_here
```

Or set environment variable:
```bash
export GOOGLE_MAPS_API_KEY=your_key
```

### 6.3 Run Flask App
```bash
python app.py
```
Starts on `http://localhost:5000`

### 6.4 Run Tests
```bash
pytest -q
```

### 6.5 Batch Inference (Debug Mode)
Uncomment and adapt the `if __name__ == "__main__"` block in `inference_engine.py`:
```python
if __name__ == "__main__":
    model = load_model("final_model.pt")
    for img in glob.glob("all_images_90/*.jpg"):
        result = run_inference_on_image(img, model, "output/")
        print(f"WWR: {result['wwr']:.4f}")
```

---

## Known Issues and Maintenance Notes

### Issue 1: Dead Code / Unused Imports
**Location**: inference_engine.py, app.py, generalization.py  
**Details**:
- `app.py` imports `json` but never uses it (uses `jsonify()` instead)
- `inference_engine.py` imports `ImageDraw` from PIL but never uses it
- `generalization.py` imports `shutil` and unused functions
- Variable `annotated_image_path` in `run_generation_pipeline` is created but never used

**Action**: Clean up in next refactor (see dead code analysis)

### Issue 2: Deprecated Files
**Location**: `address_pipeline.py`  
**Details**: Exact duplicate of `generalization.py`, left over from refactoring  
**Action**: **DELETE** this file

### Issue 3: Debug Code
**Location**: `inference_engine.py` — `if __name__ == "__main__"` block  
**Details**: Batch processing code for testing; not used in production  
**Action**: Consider moving to separate `debug_inference.py` script

### Issue 4: Hard-Coded Paths
**Locations**:
- `generalization.py`: `ENERGYPLUS_EXE`, `EPW_FILE`
- These are hard-coded for Windows system with specific EnergyPlus installation

**Action**: Consider environment-based configuration for portability

### Issue 5: Single-Zone IDF
**Details**: Generated IDFs use a single thermal zone for simplicity  
**Limitation**: Does not account for spatial variation in building  
**Future**: Could extend to multi-zone based on footprint subdivision

---

## Appendix: Quick Reference

### API Endpoints
| Method | Endpoint | Parameters | Response |
|--------|----------|------------|----------|
| GET/POST | `/` | files, address, building_type | HTML page with results |
| POST | `/fetch-street-view` | JSON: {address} | JSON: {image_url, image_name} |
| POST | `/run-pipeline` | form: address, building_type, cropped_images or images | JSON: full pipeline result |

### Key Constants
```python
DEFAULT_FACADE_ID = 2
DEFAULT_WINDOW_ID = 8
SEGFORMER_DILATION_RATIO = 0.18
BUILDING_TYPE_MAX = {'mid-terrace': 2, 'end-semi': 3, 'detached': 4}
DEFAULT_MAX_IMAGES = 4
BASELINE_WWR = 0.20  # 20% for comparison
DEFAULT_HEIGHT = 9.0  # meters
```

### Dependencies
- `flask` — Web framework
- `ultralytics` — YOLO models
- `torch` — Deep learning
- `transformers` — SegFormer fallback
- `opencv-python` — Image processing
- `numpy`, `scipy` — Numerical computing
- `requests` — HTTP (Street View, Geocoding)
- `osmnx` — OpenStreetMap data
- `eppy` — EnergyPlus IDF editing
- `Pillow` — Image I/O
