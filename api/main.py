# main.py
# This is the main FastAPI application file - it's like the "brain" of your API

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import List, Dict, Any

# Create the FastAPI app instance - this is your API!
app = FastAPI(
    title="Dataset Libraries API",
    description="API to list CSV datasets and their thumbnails",
    version="1.0.0"
)

# Configuration - where your CSV files and thumbnails live
# Get the directory where this file (main.py) is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_FOLDER = os.path.join(BASE_DIR, "datasets")  # Folder containing your CSV files
THUMBNAILS_FOLDER = os.path.join(BASE_DIR, "thumbnails")  # Folder for thumbnail images

# In-memory storage - this dictionary will hold library names and their thumbnails
# Structure: {"library_name": {"name": "housing", "csv_file": "housing.csv", "thumbnail": "/thumbnails/housing.png"}}
library_catalog: Dict[str, Dict[str, str]] = {}


def scan_libraries():
    """
    This function runs when the API starts up.
    It finds all CSV files in the datasets folder and matches them with thumbnails.
    
    What it does:
    1. Scans the datasets folder for .csv files
    2. Gets just the filenames (no parsing!)
    3. Looks for matching thumbnail images
    4. Stores the info in library_catalog
    """
    print("Scanning datasets folder for CSV files...")
    
    # Check if the datasets folder exists
    if not os.path.exists(DATASETS_FOLDER):
        print(f"Warning: {DATASETS_FOLDER} folder not found. Creating it...")
        os.makedirs(DATASETS_FOLDER)
        return
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in datasets folder.")
        return
    
    # Loop through each CSV file and catalog it
    for csv_file in csv_files:
        # Get the library name (filename without .csv extension)
        library_name = csv_file.replace('.csv', '')
        
        # Look for a matching thumbnail image
        thumbnail_path = find_thumbnail(library_name)
        
        # Store metadata about this library
        library_catalog[library_name] = {
            "name": library_name,
            "csv_file": csv_file,
            "thumbnail": thumbnail_path
        }
        
        thumbnail_status = "✓" if thumbnail_path else "✗"
        print(f"  {thumbnail_status} Found: {csv_file} (thumbnail: {thumbnail_path or 'none'})")
    
    print(f"Total libraries cataloged: {len(library_catalog)}")


def find_thumbnail(library_name: str) -> str:
    """
    Looks for a thumbnail image that matches the library name.
    Checks for common image extensions: .png, .jpg, .jpeg, .gif, .webp
    
    Args:
        library_name: Name of the library (without extension)
    
    Returns:
        Relative path to thumbnail or None if not found
    """
    if not os.path.exists(THUMBNAILS_FOLDER):
        os.makedirs(THUMBNAILS_FOLDER)
        return None
    
    # Common image extensions to check
    extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    
    for ext in extensions:
        thumbnail_file = f"{library_name}{ext}"
        thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_file)
        
        if os.path.exists(thumbnail_path):
            # Return the relative path (what the client will use)
            return f"/thumbnails/{thumbnail_file}"
    
    return None  # No thumbnail found


# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """
    This runs automatically when the API starts.
    It scans for all CSV files and their thumbnails.
    """
    scan_libraries()


@app.get("/")
async def root():
    """
    Root endpoint - just a welcome message
    URL: http://localhost:8000/
    """
    return {
        "message": "Welcome to the Dataset Libraries API!",
        "endpoints": {
            "list_libraries": "/libraries",
            "get_library_info": "/libraries/{library_name}",
            "rescan": "/rescan"
        }
    }


@app.get("/libraries")
async def list_libraries():
    """
    GET /libraries
    
    Returns a list of all available libraries (CSV filenames) with their thumbnails.
    
    Example response:
    {
        "libraries": [
            {
                "name": "housing",
                "csv_file": "housing.csv",
                "thumbnail": "/thumbnails/housing.png"
            },
            {
                "name": "sales",
                "csv_file": "sales.csv",
                "thumbnail": "/thumbnails/sales.jpg"
            }
        ]
    }
    """
    if not library_catalog:
        return {
            "libraries": [],
            "message": "No CSV files found. Add CSV files to the 'datasets' folder and restart or call /rescan."
        }
    
    return {
        "libraries": list(library_catalog.values()),
        "total_libraries": len(library_catalog)
    }


@app.get("/libraries/{library_name}")
async def get_library_info(library_name: str):
    """
    GET /libraries/{library_name}
    
    Returns information about a specific library.
    
    Example: GET /libraries/housing
    
    Args:
        library_name: Name of the library (CSV filename without .csv)
    
    Returns:
        JSON object with library info (name, csv_file, thumbnail)
    """
    # Check if the library exists
    if library_name not in library_catalog:
        raise HTTPException(
            status_code=404,
            detail=f"Library '{library_name}' not found. Available libraries: {list(library_catalog.keys())}"
        )
    
    # Return the library info
    return library_catalog[library_name]


@app.post("/rescan")
async def rescan_libraries():
    """
    POST /rescan
    
    Manually rescan the datasets folder for CSV files without restarting the server.
    Useful when you add new CSV files or thumbnails.
    """
    library_catalog.clear()
    scan_libraries()
    
    return {
        "message": "Libraries rescanned successfully",
        "libraries_found": len(library_catalog),
        "libraries": list(library_catalog.values())
    }









# ==================== HOW TO RUN ====================
"""
1. Install dependencies:
   pip install fastapi uvicorn

2. Your folder structure:
   your_project/
   ├── api/
   │   └── main.py (this file)
   ├── datasets/ (put your CSV files here)
   │   ├── housing.csv
   │   └── xxxx.csv
   └── thumbnails/ (put matching images here)
       ├── housing.png
       └── xxxx.jpg

3. Run the server (from your_project/ directory):
    uvicorn api.main:app --reload --host 0.0.0.0 --port 80
    
4. Test the API:
   - Open browser: http://localhost:8000
   - List libraries: http://localhost:8000/libraries
   - Get specific library: http://localhost:8000/libraries/housing
   - Interactive docs: http://localhost:8000/docs (automatically generated!)

5. What you'll get:
   GET /libraries returns:
   {
     "libraries": [
       {
         "name": "housing",
         "csv_file": "housing.csv", 
         "thumbnail": "/thumbnails/housing.png"
       }
     ]
   }
"""