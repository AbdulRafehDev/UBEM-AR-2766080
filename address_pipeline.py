import os
import io
import csv
import math
import logging
import subprocess
import requests
import warnings
import shutil
import osmnx as ox
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
from eppy.modeleditor import IDF

from inference_engine import load_model, run_inference_on_image, DEFAULT_MODEL_PATH

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

DEFAULT_SIZE = "640x640"
ENERGYPLUS_EXE = r"/Applications/EnergyPlus-24-1-0/energyplus-24.1.0"
EPW_FILE = "GBR_ENG_London-Heathrow.Intl.AP.037720_TMYx.2007-2021.epw"

# ==========================================
# --- IMAGE FETCHER FUNCTIONS ---
# ==========================================

def _resolve_api_key(api_key: Optional[str] = None) -> str:
    resolved_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    if not resolved_key:
        raise ValueError("Google Street View API key is missing. Pass api_key or set GOOGLE_MAPS_API_KEY.")
    return resolved_key


def fetch_street_view(
    address: str,
    output_filename,
    api_key: Optional[str] = None,
    size: str = DEFAULT_SIZE,
    fov: int = 90,
    pitch: int = 0,
    timeout: int = 30,
) -> str:
    key = _resolve_api_key(api_key)
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {"size": size, "location": address, "fov": fov, "pitch": pitch, "key": key}
    response = requests.get(base_url, params=params, timeout=timeout)

    if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image"):
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return str(output_path)

    body_preview = response.text[:200] if response.text else ""
    raise RuntimeError(f"Street View fetch failed (status={response.status_code}): {body_preview}")


# ==========================================
# --- GEOSPATIAL FUNCTIONS ---
# ==========================================

def fetch_building_footprint(address: str, google_api_key: str) -> Any:
    logger.info(f"Geocoding: {address}")
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={google_api_key}"
    response = requests.get(geocode_url).json()
    if response['status'] != 'OK':
        raise ValueError(f"Google Geocoding failed: {response['status']}")

    location = response['results'][0]['geometry']['location']
    lat, lng = location['lat'], location['lng']
    logger.info(f"Geocoded: Lat {lat}, Lng {lng}")

    logger.info("Fetching OSM building footprint...")
    gdf = ox.features_from_point((lat, lng), tags={"building": True}, dist=30)
    if gdf.empty:
        raise ValueError("No building footprints found on OSM.")
    return gdf


# ==========================================
# --- IDF GEOMETRY HELPERS ---
# ==========================================

def calculate_window_vertices(
    p1: Tuple[float, float], p2: Tuple[float, float], height: float, wwr: float
) -> Optional[List[Tuple[float, float, float]]]:
    if wwr <= 0.0:
        return None
    scale = math.sqrt(wwr)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    cx, cy, cz = p1[0] + dx / 2, p1[1] + dy / 2, height / 2
    half_w_dx, half_w_dy, half_h = (dx * scale) / 2, (dy * scale) / 2, (height * scale) / 2
    return [
        (cx - half_w_dx, cy - half_w_dy, cz + half_h),
        (cx - half_w_dx, cy - half_w_dy, cz - half_h),
        (cx + half_w_dx, cy + half_w_dy, cz - half_h),
        (cx + half_w_dx, cy + half_w_dy, cz + half_h),
    ]


def set_vertices(ep_obj: Any, vertices: List[Tuple[float, float, float]]):
    ep_obj.Number_of_Vertices = len(vertices)
    for j, v in enumerate(vertices):
        ep_obj[f"Vertex_{j+1}_Xcoordinate"] = round(v[0], 4)
        ep_obj[f"Vertex_{j+1}_Ycoordinate"] = round(v[1], 4)
        ep_obj[f"Vertex_{j+1}_Zcoordinate"] = round(v[2], 4)


def build_eppy_idf(
    gdf: Any,
    wwr: float,
    height: float = 9.0,
    idd_path: str = "V24-1-0-Energy+.idd",
    template_path: Optional[str] = None,
    output_filename: str = "final_eppy.idf",
):
    logger.info(f"Building IDF with WWR={wwr*100:.1f}%...")
    IDF.setiddname(idd_path)

    if template_path and os.path.exists(template_path):
        idf = IDF(template_path)
    else:
        idf = IDF(io.StringIO(""))

        idf.newidfobject("VERSION", Version_Identifier="24.1")
        idf.newidfobject(
            "BUILDING",
            Name="Generated Building",
            North_Axis=0,
            Terrain="Suburbs",
            Loads_Convergence_Tolerance_Value=0.04,
            Temperature_Convergence_Tolerance_Value=0.40,
            Solar_Distribution="FullInteriorAndExterior",
            Maximum_Number_of_Warmup_Days=25,
            Minimum_Number_of_Warmup_Days=6,
        )
        idf.newidfobject(
            "GLOBALGEOMETRYRULES",
            Starting_Vertex_Position="UpperLeftCorner",
            Vertex_Entry_Direction="Counterclockwise",
            Coordinate_System="Relative",
            Daylighting_Reference_Point_Coordinate_System="Relative",
            Rectangular_Surface_Coordinate_System="Relative",
        )

        idf.newidfobject("MATERIAL", Name="Ext Wall Mat", Roughness="MediumRough", Thickness=0.2, Conductivity=0.5, Density=1000, Specific_Heat=1000)
        idf.newidfobject("MATERIAL", Name="Roof Mat", Roughness="MediumRough", Thickness=0.2, Conductivity=0.5, Density=1000, Specific_Heat=1000)
        idf.newidfobject("MATERIAL", Name="Floor Mat", Roughness="MediumRough", Thickness=0.2, Conductivity=0.5, Density=1000, Specific_Heat=1000)
        idf.newidfobject("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", Name="Window Mat", UFactor=3.0, Solar_Heat_Gain_Coefficient=0.7, Visible_Transmittance=0.8)

        ext_wall_const = idf.newidfobject("CONSTRUCTION", Name="Ext Wall", Outside_Layer="Ext Wall Mat")
        roof_const = idf.newidfobject("CONSTRUCTION", Name="Roof", Outside_Layer="Roof Mat")
        floor_const = idf.newidfobject("CONSTRUCTION", Name="Floor", Outside_Layer="Floor Mat")
        window_const = idf.newidfobject("CONSTRUCTION", Name="Window", Outside_Layer="Window Mat")

        idf.newidfobject("SCHEDULETYPELIMITS", Name="Fraction", Lower_Limit_Value=0.0, Upper_Limit_Value=1.0, Numeric_Type="CONTINUOUS")
        idf.newidfobject("SCHEDULETYPELIMITS", Name="Temperature", Lower_Limit_Value=-50, Upper_Limit_Value=150, Numeric_Type="CONTINUOUS")
        idf.newidfobject("SCHEDULETYPELIMITS", Name="Control Type", Lower_Limit_Value=0, Upper_Limit_Value=4, Numeric_Type="DISCRETE")
        idf.newidfobject("SCHEDULE:COMPACT", Name="Always On", Schedule_Type_Limits_Name="Fraction", Field_1="Through: 12/31", Field_2="For: AllDays", Field_3="Until: 24:00", Field_4=1)
        idf.newidfobject("SCHEDULE:COMPACT", Name="Yearly Thermostat", Schedule_Type_Limits_Name="Control Type", Field_1="Through: 12/31", Field_2="For: AllDays", Field_3="Until: 24:00", Field_4=4)
        idf.newidfobject("SCHEDULE:COMPACT", Name="Zone Heating Setpoint Schedule", Schedule_Type_Limits_Name="Temperature", Field_1="Through: 12/31", Field_2="For: AllDays", Field_3="Until: 24:00", Field_4=22)
        idf.newidfobject("SCHEDULE:COMPACT", Name="Zone Cooling Setpoint Schedule", Schedule_Type_Limits_Name="Temperature", Field_1="Through: 12/31", Field_2="For: AllDays", Field_3="Until: 24:00", Field_4=24)
        idf.newidfobject("RUNPERIOD", Name="Annual", Begin_Month=1, Begin_Day_of_Month=1, End_Month=12, End_Day_of_Month=31)
        idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Heating Energy", Reporting_Frequency="Annual")
        idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Cooling Energy", Reporting_Frequency="Annual")

    gdf_proj = ox.projection.project_gdf(gdf)
    geom = gdf_proj.iloc[0].geometry

    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        coords = list(geom.geoms[0].exterior.coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    min_x, min_y = min(c[0] for c in coords), min(c[1] for c in coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    local_coords = [(c[0] - min_x, c[1] - min_y) for c in coords]

    zone = idf.newidfobject("ZONE", Name="Dynamic_Zone")
    zone_supply_air_node_name = f"{zone.Name} SupplyAirNode"
    zone_air_node_name = f"{zone.Name} ZoneAirNode"
    zone_return_air_node_name = f"{zone.Name} ZoneReturnAirNode"

    idf.newidfobject(
        "ZONECONTROL:THERMOSTAT",
        Name=f"{zone.Name} Thermostat",
        Zone_or_ZoneList_Name=zone.Name,
        Control_Type_Schedule_Name="Yearly Thermostat",
        Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
        Control_1_Name=f"Dual Setpoint - {zone.Name}",
    )
    idf.newidfobject(
        "THERMOSTATSETPOINT:DUALSETPOINT",
        Name=f"Dual Setpoint - {zone.Name}",
        Heating_Setpoint_Temperature_Schedule_Name="Zone Heating Setpoint Schedule",
        Cooling_Setpoint_Temperature_Schedule_Name="Zone Cooling Setpoint Schedule",
    )
    idf.newidfobject(
        "ZONEHVAC:IDEALLOADSAIRSYSTEM",
        Name=f"{zone.Name} IdealLoads",
        Availability_Schedule_Name="Always On",
        Zone_Supply_Air_Node_Name=zone_supply_air_node_name,
        Zone_Exhaust_Air_Node_Name="",
        System_Inlet_Air_Node_Name="",
        Maximum_Heating_Supply_Air_Temperature=50,
        Minimum_Cooling_Supply_Air_Temperature=13,
        Maximum_Heating_Supply_Air_Humidity_Ratio=0.015,
        Minimum_Cooling_Supply_Air_Humidity_Ratio=0.009,
        Heating_Limit="NoLimit",
        Cooling_Limit="NoLimit",
        Dehumidification_Control_Type="ConstantSupplyHumidityRatio",
        Humidification_Control_Type="ConstantSupplyHumidityRatio",
        Outdoor_Air_Economizer_Type="NoEconomizer",
        Heat_Recovery_Type="None",
    )
    idf.newidfobject(
        "ZONEHVAC:EQUIPMENTLIST",
        Name=f"{zone.Name} EquipmentList",
        Load_Distribution_Scheme="SequentialLoad",
        Zone_Equipment_1_Object_Type="ZoneHVAC:IdealLoadsAirSystem",
        Zone_Equipment_1_Name=f"{zone.Name} IdealLoads",
        Zone_Equipment_1_Cooling_Sequence=1,
        Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
    )
    idf.newidfobject(
        "ZONEHVAC:EQUIPMENTCONNECTIONS",
        Zone_Name=zone.Name,
        Zone_Conditioning_Equipment_List_Name=f"{zone.Name} EquipmentList",
        Zone_Air_Inlet_Node_or_NodeList_Name=zone_supply_air_node_name,
        Zone_Air_Exhaust_Node_or_NodeList_Name="",
        Zone_Air_Node_Name=zone_air_node_name,
        Zone_Return_Air_Node_or_NodeList_Name=zone_return_air_node_name,
    )

    for i in range(len(local_coords)):
        p1 = local_coords[i]
        p2 = local_coords[(i + 1) % len(local_coords)]
        v1, v2 = (p1[0], p1[1], height), (p1[0], p1[1], 0.0)
        v3, v4 = (p2[0], p2[1], 0.0), (p2[0], p2[1], height)

        wall_name = f"Wall_{i+1}"
        wall = idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name=wall_name,
            Surface_Type="Wall",
            Construction_Name=ext_wall_const.Name,
            Zone_Name=zone.Name,
            Outside_Boundary_Condition="Outdoors",
            Sun_Exposure="SunExposed",
            Wind_Exposure="WindExposed",
        )
        set_vertices(wall, [v1, v2, v3, v4])

        win_coords = calculate_window_vertices(p1, p2, height, wwr)
        if win_coords:
            window = idf.newidfobject(
                "FENESTRATIONSURFACE:DETAILED",
                Name=f"Window_{i+1}",
                Surface_Type="Window",
                Construction_Name=window_const.Name,
                Building_Surface_Name=wall_name,
            )
            set_vertices(window, win_coords)

    roof = idf.newidfobject(
        "BUILDINGSURFACE:DETAILED",
        Name="Roof_Main",
        Surface_Type="Roof",
        Construction_Name=roof_const.Name,
        Zone_Name=zone.Name,
        Outside_Boundary_Condition="Outdoors",
        Sun_Exposure="SunExposed",
        Wind_Exposure="WindExposed",
    )
    set_vertices(roof, [(c[0], c[1], height) for c in local_coords])

    floor = idf.newidfobject(
        "BUILDINGSURFACE:DETAILED",
        Name="Floor_Main",
        Surface_Type="Floor",
        Construction_Name=floor_const.Name,
        Zone_Name=zone.Name,
        Outside_Boundary_Condition="Ground",
    )
    set_vertices(floor, [(c[0], c[1], 0.0) for c in reversed(local_coords)])

    idf.saveas(output_filename)
    logger.info(f"IDF saved: {output_filename}")
    return output_filename


# ==========================================
# --- ENERGYPLUS SIMULATION ---
# ==========================================

def run_energyplus_simulation(idf_path: str, output_dir: str) -> Dict[str, Any]:
    """Run EnergyPlus and return the output directory path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        ENERGYPLUS_EXE,
        "--weather", EPW_FILE,
        "--output-directory", str(output_dir),
        "--readvars",
        str(idf_path),
    ]

    logger.info(f"Running EnergyPlus: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(
            f"EnergyPlus failed (code {result.returncode}):\n{result.stderr[-1000:]}"
        )

    logger.info("EnergyPlus simulation complete.")
    return {"output_dir": str(output_dir), "stdout": result.stdout}


def extract_energy_consumption(output_dir: str) -> Dict[str, float]:
    """
    Parse EnergyPlus CSV output for annual energy totals.
    Looks for the eplusout.csv file produced by --readvars.
    Returns a dict with keys: heating_kwh, cooling_kwh, total_kwh
    """
    output_path = Path(output_dir)
    csv_file = output_path / "eplusout.csv"

    if not csv_file.exists():
        # Fallback: look for any CSV
        csvs = list(output_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV output found in {output_dir}")
        csv_file = csvs[0]

    logger.info(f"Parsing energy output: {csv_file}")

    heating_j = 0.0
    cooling_j = 0.0

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if not value:
                    continue
                key_lower = key.lower()
                try:
                    val = float(value)
                except ValueError:
                    continue
                if "heating" in key_lower and "energy" in key_lower:
                    heating_j += val
                elif "cooling" in key_lower and "energy" in key_lower:
                    cooling_j += val

    # Convert Joules to kWh (1 kWh = 3,600,000 J)
    j_to_kwh = 1 / 3_600_000
    heating_kwh = round(heating_j * j_to_kwh, 2)
    cooling_kwh = round(cooling_j * j_to_kwh, 2)
    total_kwh = round((heating_j + cooling_j) * j_to_kwh, 2)

    return {
        "heating_kwh": heating_kwh,
        "cooling_kwh": cooling_kwh,
        "total_kwh": total_kwh,
    }


# ==========================================
# --- MAIN PIPELINE ---
# ==========================================

def run_generation_pipeline(
    address: str,
    ai_model: Any,
    static_folder: str = "static",
    idd_path: str = "V24-1-0-Energy+.idd",
    provided_image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Fetch building footprint (OSM)
      2. Fetch Street View image (skipped if provided_image_paths is given)
      3. Run AI inference → predicted WWR (averaged if multiple images)
      4. Build IDF with AI WWR → simulate → extract energy
      5. Build IDF with 20% baseline WWR → simulate → extract energy
      6. Return all results for rendering

    If provided_image_paths is given, those images are used directly instead of
    fetching from Street View. WWR is computed per image then averaged.
    """
    save_dir = Path(static_folder) / "generated_assets"
    save_dir.mkdir(parents=True, exist_ok=True)

    uid = uuid4().hex[:8]
    annotated_image_path = save_dir / f"{uid}_annotated.jpg"
    idf_ai_path = save_dir / f"{uid}_ai.idf"
    idf_baseline_path = save_dir / f"{uid}_baseline.idf"
    sim_ai_dir = save_dir / f"{uid}_sim_ai"
    sim_baseline_dir = save_dir / f"{uid}_sim_baseline"

    API_KEY = _resolve_api_key()

    results = {
        "success": False,
        "address": address,
        "wwr_ai": None,
        "wwr_baseline": 20.0,
        "image_url": "",        # kept for backward compat (first image)
        "annotated_url": "",    # kept for backward compat (first annotated)
        "image_pairs": [],      # list of {image_url, annotated_url} for all images
        "energy_ai": {},
        "energy_baseline": {},
        "error": None,
    }

    try:
        # Phase 1: Geospatial
        building_gdf = fetch_building_footprint(address, API_KEY)

        # Phase 2: Image acquisition + AI inference
        if provided_image_paths and len(provided_image_paths) > 0:
            image_paths_to_infer = provided_image_paths
        else:
            temp_image_path = save_dir / f"{uid}_streetview.jpg"
            fetch_street_view(address, temp_image_path, api_key=API_KEY)
            image_paths_to_infer = [str(temp_image_path)]

        # Run inference on each image, collect WWR values and image pairs
        wwr_values = []
        all_class_counts: Dict[str, int] = {}
        image_pairs = []

        for i, img_path in enumerate(image_paths_to_infer):
            img_path = Path(img_path)
            ann_path = save_dir / f"{uid}_annotated_{i}_{img_path.stem}.jpg"

            inference = run_inference_on_image(
                image_path=str(img_path),
                model=ai_model,
                output_path=str(ann_path),
            )
            wwr_values.append(float(inference.get("wwr", 0.0)))
            for k, v in inference.get("class_counts", {}).items():
                all_class_counts[k] = all_class_counts.get(k, 0) + v

            # Build static-relative URLs for this pair
            # Original image: could be in uploads/ or generated_assets/
            if img_path.parent.resolve() == save_dir.resolve():
                orig_url = f"generated_assets/{img_path.name}"
            else:
                # It's in uploads/ (cropped images saved there by app.py)
                orig_url = f"uploads/{img_path.name}"

            ann_url = f"generated_assets/{ann_path.name}"
            image_pairs.append({"image_url": orig_url, "annotated_url": ann_url})

        results["image_pairs"] = image_pairs
        # Backward-compat single fields = first pair
        if image_pairs:
            results["image_url"]    = image_pairs[0]["image_url"]
            results["annotated_url"] = image_pairs[0]["annotated_url"]

        # Average WWR across all images
        predicted_wwr = sum(wwr_values) / len(wwr_values) if wwr_values else 0.0
        if predicted_wwr <= 0.0:
            logger.warning("AI predicted 0% WWR — defaulting to 15%.")
            predicted_wwr = 0.15

        results["wwr_ai"] = round(predicted_wwr * 100, 2)
        results["class_counts"] = all_class_counts

        # Phase 3a: IDF + simulation with AI WWR
        build_eppy_idf(
            gdf=building_gdf,
            wwr=predicted_wwr,
            idd_path=idd_path,
            output_filename=str(idf_ai_path),
        )
        run_energyplus_simulation(str(idf_ai_path), str(sim_ai_dir))
        results["energy_ai"] = extract_energy_consumption(str(sim_ai_dir))

        # Phase 3b: IDF + simulation with 20% baseline WWR
        build_eppy_idf(
            gdf=building_gdf,
            wwr=0.20,
            idd_path=idd_path,
            output_filename=str(idf_baseline_path),
        )
        run_energyplus_simulation(str(idf_baseline_path), str(sim_baseline_dir))
        results["energy_baseline"] = extract_energy_consumption(str(sim_baseline_dir))

        results["success"] = True

    except Exception as e:
        logger.error(f"Pipeline failed for '{address}': {e}", exc_info=True)
        results["error"] = str(e)

    return results