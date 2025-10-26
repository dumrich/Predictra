from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  # Add this import
import os
import subprocess
import json
from typing import List, Dict, Any
from util.csvCleaner import CSVCleaner

# Create the FastAPI app instance - this is your API!
app = FastAPI(
    title="Dataset Libraries API",
    description="API to list CSV datasets and their thumbnails",
    version="1.0.0"
)

origins = [
    "http://100.72.88.122:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_FOLDER = os.path.join(BASE_DIR, "datasets")  # Folder containing CSV files
THUMBNAILS_FOLDER = os.path.join(BASE_DIR, "thumbnails")  # Folder for thumbnail images
CSV_CLEANER_PATH = os.path.join(BASE_DIR, "util", "csvCleaner.py")  # Path to CSV cleaner script

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
            "upload_dataset": "/upload",
            "analyze_dataset": "/analyze",
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


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    POST /upload
    
    Uploads a CSV file, saves it to datasets folder, runs csvCleaner.py on it,
    and returns the cleaned keys/output.
    
    How it works:
    1. Receives CSV file from frontend
    2. Saves it to datasets/ folder
    3. Runs util/csvCleaner.py on the file
    4. Returns the output (keys) back to frontend
    
    Example usage (from frontend):
    ```javascript
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    fetch('http://localhost:8000/upload', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => console.log(data));
    ```
    
    Args:
        file: The CSV file uploaded from frontend
    
    Returns:
        JSON with cleaned keys and file info
    """
    # Validate that it's a CSV file
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed. Please upload a .csv file."
        )
    
    try:
        # Save the uploaded file to datasets folder
        file_path = os.path.join(DATASETS_FOLDER, file.filename)
        
        # Read and write the file
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        print(f"✓ Saved file: {file.filename} to {DATASETS_FOLDER}")
        
        try:
            scan_libraries()
            # Return success response with cleaned data
            return {
                "success": True,
                "message": f"File '{file.filename}' uploaded and processed successfully",
                "filename": file.filename,
                "file_path": file_path,
            }
            
        except subprocess.CalledProcessError as e:
            # CSV cleaner script failed
            raise HTTPException(
                status_code=500,
                detail=f"CSV cleaner script failed: {e.stderr}"
            )
        
        except FileNotFoundError:
            # csvCleaner.py not found
            raise HTTPException(
                status_code=500,
                detail=f"CSV cleaner script not found at: {CSV_CLEANER_PATH}"
            )
    
    except Exception as e:
        # General error handling
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.get("/analyze")
async def analyze_dataset(dataset_name: str):
    """
    GET /analyze
    
    Analyzes a CSV dataset by running it through the CSV cleaner
    and returns the headers of the cleaned dataset.
    
    Args:
        dataset_name: Name of the dataset (CSV filename without .csv extension)
    
    Returns:
        JSON object with the cleaned headers
    
    Example request:
    GET /analyze?dataset_name=housing
    """
    try:
        
        # Construct the full path to the CSV file
        csv_filename = f"../datasets/{dataset_name}.csv"
        file_path = os.path.join(DATASETS_FOLDER, csv_filename)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found. Available datasets: {list(library_catalog.keys())}"
            )
        
        # Create CSVCleaner instance and load the CSV
        cleaner = CSVCleaner(file_path)
        cleaner.load_csv()
        # Return the headers
        return {
            "success": True,
            "dataset": dataset_name,
            "headers": cleaner.header,
            "total_headers": len(cleaner.header)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing dataset: {str(e)}"
        )

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
