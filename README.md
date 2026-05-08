# Energy+ AI: Window Analysis & Energy Simulation

A full-stack web application that predicts building energy consumption using AI-powered window detection and EnergyPlus simulation.

## Quick Start

### 1. Setup
```bash
# Clone and navigate to project
cd "C:\path\to\Final"

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file in project root:
```
GOOGLE_MAPS_API_KEY=your_google_api_key_here
```

### 3. Run the App
```bash
python app.py
```
Open browser to `http://localhost:5000`

## Features

 **AI-Powered Window Detection**
- YOLO segmentation for facade and window detection
- Automatic fallback to SegFormer if needed
- Window-to-Wall Ratio (WWR) calculation

 **Building Energy Simulation**
- Automatic building footprint retrieval (OpenStreetMap)
- Parametric EnergyPlus model generation
- Annual energy simulation with London weather

 **Web Interface**
- Address lookup with Street View preview
- Image upload or interactive cropping
- Side-by-side energy comparison (AI vs. 20% baseline)

 **Multi-Image Support**
- Process multiple facade views
- Automatic WWR averaging
- Building type limits (mid-terrace: 2, end-semi: 3, detached: 4)

## Usage Flows

### Flow 1: Address + Images
1. Enter address
2. Upload or crop facade images
3. Select building type
4. Submit → Get energy comparison

### Flow 2: Street View + Crop
1. Enter address
2. Click "Fetch Street View"
3. Crop interesting facade area in UI
4. Submit → Get energy comparison

### Flow 3: Images Only
1. Upload/crop images
2. Skip address
3. Get WWR and facade annotations (no energy data)

## Documentation

📖 **[ARCHITECTURE_AND_PIPELINE.md](ARCHITECTURE_AND_PIPELINE.md)** — Single canonical guide for the full app, CV pipeline, and EnergyPlus flow

## Project Structure

```
├── app.py                    # Flask web app
├── inference_engine.py       # YOLO segmentation + WWR
├── generalization.py         # Energy pipeline orchestration
├── address_pipeline.py       # Deprecated duplicate of generalization.py
├── requirements.txt          # Dependencies
├── final_model.pt           # Trained YOLO model
├── templates/index.html      # Web interface
├── static/                   # Assets and results
└── ARCHITECTURE_AND_PIPELINE.md
```



## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET/POST | `/` | Main page with upload/address |
| POST | `/fetch-street-view` | Get Street View image |
| POST | `/run-pipeline` | Execute full energy analysis |

## System Requirements

- Python 3.11+
- Windows (EnergyPlus 24.1 installation required)
- Google Maps API key (for Street View + Geocoding)
- ~3 GB RAM (model loading)
- 5-10 GB free disk (simulations)

## Key Technologies

- **Framework**: Flask (Python web)
- **Computer Vision**: YOLO v8 segmentation, SegFormer fallback
- **Energy Simulation**: EnergyPlus 24.1
- **Geospatial**: OpenStreetMap (via osmnx)
- **Image Processing**: OpenCV, NumPy, SciPy
- **IDF Generation**: eppy (EnergyPlus Python API)

## Troubleshooting

**Q: "EnergyPlus failed" error**
- Verify `C:\EnergyPlusV24-1-0\energyplus.exe` exists
- Update path in `generalization.py` if different

**Q: No facade detected**
- Application falls back to SegFormer automatically
- Try different image angles or lighting

**Q: Google API errors**
- Check GOOGLE_MAPS_API_KEY is set and valid
- Verify Street View API is enabled in Google Cloud Console

## Maintenance

⚠️ **Known Issues**:
- `address_pipeline.py` is a duplicate of `generalization.py` — can be deleted
- Hard-coded paths for Windows (see generalization.py)
- Single-zone IDF model (could extend for multi-zone)
- Unused imports (see ARCHITECTURE_AND_PIPELINE.md for details)

## Contact & Support

For issues or questions, refer to:
1. [ARCHITECTURE_AND_PIPELINE.md](ARCHITECTURE_AND_PIPELINE.md) for the complete system design
2. Code comments in [inference_engine.py](inference_engine.py) for algorithm details
