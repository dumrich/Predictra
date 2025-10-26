# main.py
# This is the main FastAPI application file - it's like the "brain" of your API

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  # Add this import
import os
import subprocess
import json
import csv
import asyncio
from typing import List, Dict, Any

# Create the FastAPI app instance - this is your API!
app = FastAPI(
    title="Dataset Libraries API",
    description="API to list CSV datasets, upload new ones, and stream data live.",
    version="1.1.0"
)

# Configuration - where your CSV files and thumbnails live
# Get the directory where this file (main.py) is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_FOLDER = os.path.join(BASE_DIR, "datasets")  # Folder containing your CSV files
THUMBNAILS_FOLDER = os.path.join(BASE_DIR, "thumbnails")  # Folder for thumbnail images
CSV_CLEANER_PATH = os.path.join(BASE_DIR, "util", "csvCleaner.py")  # Path to your CSV cleaner script

# In-memory storage for available CSV libraries
library_catalog: Dict[str, Dict[str, str]] = {}


def scan_libraries():
    """Scan the datasets folder and register all CSV libraries."""
    print("Scanning datasets folder for CSV files...")

    if not os.path.exists(DATASETS_FOLDER):
        print(f"Warning: {DATASETS_FOLDER} not found. Creating it...")
        os.makedirs(DATASETS_FOLDER)
        return

    csv_files = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return

    for csv_file in csv_files:
        library_name = csv_file.replace('.csv', '')
        thumbnail_path = find_thumbnail(library_name)
        library_catalog[library_name] = {
            "name": library_name,
            "csv_file": csv_file,
            "thumbnail": thumbnail_path
        }
        print(f"✓ Cataloged: {csv_file} (thumbnail: {thumbnail_path or 'none'})")

    print(f"Total libraries cataloged: {len(library_catalog)}")


def find_thumbnail(library_name: str) -> str:
    """Find a thumbnail image that matches a CSV name."""
    if not os.path.exists(THUMBNAILS_FOLDER):
        os.makedirs(THUMBNAILS_FOLDER)
        return None

    for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        thumbnail_file = f"{library_name}{ext}"
        thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_file)
        if os.path.exists(thumbnail_path):
            return f"/thumbnails/{thumbnail_file}"

    return None


# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Run automatically when the API starts."""
    scan_libraries()


@app.get("/")
async def root():
    """Root endpoint - welcome message and docs link."""
    return {
        "message": "Welcome to the Dataset Libraries API!",
        "endpoints": {
            "list_libraries": "/libraries",
            "get_library_info": "/libraries/{library_name}",
            "upload_dataset": "/upload",
            "rescan": "/rescan",
            "stream_data": "/stream/{library_name}"
        }
    }


@app.get("/libraries")
async def list_libraries():
    """List all available CSV datasets."""
    if not library_catalog:
        return {
            "libraries": [],
            "message": "No CSV files found. Add files to 'datasets' or call /rescan."
        }

    return {
        "libraries": list(library_catalog.values()),
        "total_libraries": len(library_catalog)
    }


@app.get("/libraries/{library_name}")
async def get_library_info(library_name: str):
    """Get information about one CSV library."""
    if library_name not in library_catalog:
        raise HTTPException(
            status_code=404,
            detail=f"Library '{library_name}' not found. Available: {list(library_catalog.keys())}"
        )

    return library_catalog[library_name]


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process a CSV file using csvCleaner.py."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files are allowed.")

    try:
        file_path = os.path.join(DATASETS_FOLDER, file.filename)
        contents = await file.read()

        with open(file_path, 'wb') as f:
            f.write(contents)

        print(f"✓ Saved file: {file.filename}")

        try:
            result = subprocess.run(
                ['python', CSV_CLEANER_PATH, file.filename],
                capture_output=True, text=True, check=True, cwd=DATASETS_FOLDER
            )
            cleaner_output = result.stdout.strip()
            try:
                cleaned_data = json.loads(cleaner_output)
            except json.JSONDecodeError:
                cleaned_data = {"output": cleaner_output}

            scan_libraries()

            return {
                "success": True,
                "message": f"File '{file.filename}' uploaded and processed successfully",
                "cleaned_data": cleaned_data
            }

        except subprocess.CalledProcessError as e:
            raise HTTPException(500, f"CSV cleaner failed: {e.stderr}")

    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")


@app.post("/rescan")
async def rescan_libraries():
    """Manually rescan the datasets folder for new CSVs."""
    library_catalog.clear()
    scan_libraries()
    return {
        "message": "Rescan complete.",
        "libraries": list(library_catalog.values())
    }


# ==================== NEW: REAL-TIME CSV STREAMING ====================

@app.websocket("/stream/{library_name}")
async def stream_data(websocket: WebSocket, library_name: str):
    """
    WebSocket endpoint that streams CSV data row by row.
    Frontend connects here to receive live updates for graphing.
    """
    await websocket.accept()
    csv_path = os.path.join(DATASETS_FOLDER, f"{library_name}.csv")

    if not os.path.exists(csv_path):
        await websocket.send_json({"error": f"CSV '{library_name}' not found"})
        await websocket.close()
        return

    try:
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Send each row to frontend as JSON
                await websocket.send_json(row)
                await asyncio.sleep(0.5)  # simulate live streaming delay

        await websocket.send_json({"message": "Stream complete"})
        await websocket.close()

    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
