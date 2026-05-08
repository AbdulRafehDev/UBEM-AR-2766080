from pathlib import Path
from uuid import uuid4
import base64
import json

from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename

from inference_engine import DEFAULT_MODEL_PATH, load_model, run_inference_on_image
from address_pipeline import fetch_street_view, run_generation_pipeline

BUILDING_TYPE_MAX = {
    'mid-terrace': 2,
    'end-semi': 3,
    'detached': 4,
}
DEFAULT_MAX_IMAGES = 4  # absolute ceiling regardless of type

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
ANNOTATED_DIR = STATIC_DIR / "annotated"
GENERATED_DIR = STATIC_DIR / "generated_assets"

for directory in (UPLOAD_DIR, ANNOTATED_DIR, GENERATED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
model = load_model(str(BASE_DIR / DEFAULT_MODEL_PATH))


def _save_uploaded_images(uploaded_files):
    saved = []
    for uploaded_file in uploaded_files:
        if not uploaded_file or not uploaded_file.filename:
            continue
        safe_name = secure_filename(uploaded_file.filename) or "upload.jpg"
        unique_name = f"{uuid4().hex}_{safe_name}"
        output_path = UPLOAD_DIR / unique_name
        uploaded_file.save(output_path)
        saved.append((output_path, safe_name))
    return saved


def _save_cropped_image(data_url: str, prefix: str = "cropped") -> Path:
    """Decode a base64 data URL and save to UPLOAD_DIR."""
    header, encoded = data_url.split(",", 1)
    ext = "png" if "png" in header else "jpg"
    filename = f"{uuid4().hex}_{prefix}.{ext}"
    output_path = UPLOAD_DIR / filename
    output_path.write_bytes(base64.b64decode(encoded))
    return output_path


def _process_image(image_path, display_name):
    annotated_name = f"annotated_{image_path.name}"
    annotated_path = ANNOTATED_DIR / annotated_name

    inference = run_inference_on_image(
        image_path=str(image_path),
        model=model,
        output_path=str(annotated_path),
    )

    return {
        "name": display_name,
        "wwr": f"{inference['wwr']:.4f}",
        "class_counts": inference["class_counts"],
        "original_url": url_for("static", filename=f"uploads/{image_path.name}"),
        "annotated_url": url_for("static", filename=f"annotated/{annotated_name}"),
    }


@app.route("/fetch-street-view", methods=["POST"])
def fetch_street_view_endpoint():
    """Fetch a Street View image for the given address and return its URL for cropping."""
    data = request.get_json()
    address = (data or {}).get("address", "").strip()
    if not address:
        return jsonify({"error": "Address is required"}), 400

    try:
        uid = uuid4().hex[:8]
        image_path = GENERATED_DIR / f"{uid}_streetview.jpg"
        fetch_street_view(address, str(image_path))
        image_url = url_for("static", filename=f"generated_assets/{image_path.name}")
        return jsonify({"image_url": image_url, "image_name": image_path.name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-pipeline", methods=["POST"])
def run_pipeline_endpoint():
    """Run full energy pipeline with pre-supplied (optionally cropped) images."""
    address = request.form.get("address", "").strip()
    if not address:
        return jsonify({"error": "Address is required"}), 400

    building_type = request.form.get("building_type", "detached")
    max_images = BUILDING_TYPE_MAX.get(building_type, DEFAULT_MAX_IMAGES)

    provided_paths = []

    # Cropped images sent as base64 data URLs from the JS crop UI
    cropped_data = request.form.getlist("cropped_images")
    for idx, data_url in enumerate(cropped_data):
        if data_url:
            saved = _save_cropped_image(data_url, prefix=f"cropped_{idx}")
            provided_paths.append(str(saved))

    # Fallback: raw file uploads
    if not provided_paths:
        for f in request.files.getlist("images"):
            if f and f.filename:
                safe = secure_filename(f.filename) or "upload.jpg"
                p = UPLOAD_DIR / f"{uuid4().hex}_{safe}"
                f.save(p)
                provided_paths.append(str(p))

    if len(provided_paths) > max_images:
        return jsonify({"error": f"Too many images. Maximum allowed for this building type is {max_images}."}), 400

    generalization_result = run_generation_pipeline(
        address=address,
        ai_model=model,
        static_folder=str(STATIC_DIR),
        idd_path=str(BASE_DIR / "V24-1-0-Energy+.idd"),
        provided_image_paths=provided_paths if provided_paths else None,
    )

    if not generalization_result["success"]:
        return jsonify({"error": generalization_result.get("error")}), 500

    return jsonify(generalization_result)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None
    address = ""
    generalization_result = None

    if request.method == "POST":
        address = request.form.get("address", "").strip()

        # Address-only flow (no images) — legacy POST
        if address and not request.files.getlist("images") and not request.form.getlist("cropped_images"):
            generalization_result = run_generation_pipeline(
                address=address,
                ai_model=model,
                static_folder=str(STATIC_DIR),
                idd_path=str(BASE_DIR / "V24-1-0-Energy+.idd"),
            )
            if not generalization_result["success"]:
                error = generalization_result.get("error")

            return render_template(
                "index.html",
                generalization_result=generalization_result,
                results=[],
                error=error,
                address=address,
            )

        # Image upload flow (with or without address)
        building_type = request.form.get("building_type", "detached")
        max_images = BUILDING_TYPE_MAX.get(building_type, DEFAULT_MAX_IMAGES)

        cropped_data = request.form.getlist("cropped_images")
        image_paths_and_names = []

        if cropped_data and any(cropped_data):
            for idx, data_url in enumerate(cropped_data):
                if data_url:
                    saved = _save_cropped_image(data_url, prefix=f"cropped_{idx}")
                    image_paths_and_names.append((saved, saved.name))
        else:
            image_paths_and_names = list(_save_uploaded_images(request.files.getlist("images")))

        if len(image_paths_and_names) > max_images:
            error = f"Too many images uploaded. Maximum for this building type is {max_images}."
            return render_template("index.html", results=[], error=error, address=address, generalization_result=None)

        if address and image_paths_and_names:
            # Address + images → pipeline with provided images
            generalization_result = run_generation_pipeline(
                address=address,
                ai_model=model,
                static_folder=str(STATIC_DIR),
                idd_path=str(BASE_DIR / "V24-1-0-Energy+.idd"),
                provided_image_paths=[str(p) for p, _ in image_paths_and_names],
            )
            if not generalization_result["success"]:
                error = generalization_result.get("error")

            return render_template(
                "index.html",
                generalization_result=generalization_result,
                results=[],
                error=error,
                address=address,
            )

        # Images only — standard inference
        try:
            if not image_paths_and_names:
                error = "Enter an address or upload at least one image."
            else:
                for image_path, display_name in image_paths_and_names:
                    results.append(_process_image(image_path, display_name))
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        results=results,
        error=error,
        address=address,
        generalization_result=generalization_result,
    )


if __name__ == "__main__":
    app.run(debug=True)