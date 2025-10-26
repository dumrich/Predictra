from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  # Add this import
from pydantic import BaseModel
import os
import subprocess
import json
import csv
import asyncio
from typing import List, Dict, Any
import sys
import threading

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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_FOLDER = os.path.join(BASE_DIR, "datasets")  # Folder containing CSV files
THUMBNAILS_FOLDER = os.path.join(BASE_DIR, "thumbnails")  # Folder for thumbnail images


sys.path.append(BASE_DIR)

app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_FOLDER), name="thumbnails")
from util.csvCleaner import CSVCleaner
from util.createNeuralNet import *

# Pydantic model for train request
class TrainRequest(BaseModel):
    dataset_name: str
    num_epochs: int
    test_size: float
    predict_field: str

# Pydantic model for prediction request
class PredictRequest(BaseModel):
    features: Dict[str, Any]  # Key-value pairs of feature names and values

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
            "train_model": "/train",
            "rescan": "/rescan",
            "stream_csv": "/stream/{library_name}"
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
        
        scan_libraries()
        # Return success response with cleaned data
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded and processed successfully",
            "filename": file.filename,
            "file_path": file_path,
        }
            
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

ann = None
scaler_x = None
scaler_y = None
cleaner = None
active_websockets: List[WebSocket] = []
training_active = False

async def async_train_model(dataset_name: str, num_epochs: int, test_size: float, predict_field: str):
    """
    Asynchronous training function that runs in background and broadcasts loss data
    """
    global ann, scaler_x, scaler_y, cleaner, training_active

    try:
        training_active = True

        # Construct the full path to the CSV file
        csv_filename = f"../datasets/{dataset_name}.csv"
        file_path = os.path.join(DATASETS_FOLDER, csv_filename)

        # Create dataset and model
        from util.createNeuralNet import create_dataset, create_ann
        import torch.optim as optim

        train_loader, test_loader, x_shape, scaler_y_local, scaler_x_local, cleaner_local = create_dataset(file_path, predict_field, test_size=test_size)

        # Store globally for predictions
        ann = create_ann(x_shape)
        scaler_x = scaler_x_local
        scaler_y = scaler_y_local
        cleaner = cleaner_local

        optimizer = optim.Adam(ann.parameters(), lr=0.001, weight_decay=1e-5)

        # Start training with WebSocket broadcasting (send averages every 2 epochs)
        broadcast_interval = 2  # Send loss data every 2 epochs
        await ann.train_model_with_websocket(optimizer, train_loader, test_loader, num_epochs, broadcast_loss, broadcast_interval)

        # Send completion message
        if active_websockets:
            completion_message = {
                "type": "training_complete",
                "message": "Training completed successfully"
            }
            for websocket in active_websockets:
                try:
                    await websocket.send_json(completion_message)
                except:
                    pass

    except Exception as e:
        print(f"Training error: {e}")
        # Send error message to websockets
        if active_websockets:
            error_message = {
                "type": "error",
                "message": f"Training failed: {str(e)}"
            }
            for websocket in active_websockets:
                try:
                    await websocket.send_json(error_message)
                except:
                    pass
    finally:
        training_active = False

@app.post("/train")
async def train_model(request: TrainRequest):
    global training_active

    try:
        # Check if training is already active
        if training_active:
            return {
                "success": False,
                "message": "Training is already in progress. Please wait for it to complete."
            }

        # Construct the full path to the CSV file
        csv_filename = f"../datasets/{request.dataset_name}.csv"
        file_path = os.path.join(DATASETS_FOLDER, csv_filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{request.dataset_name}' not found. Available datasets: {list(library_catalog.keys())}"
            )

        # Start async training task
        asyncio.create_task(async_train_model(
            request.dataset_name,
            request.num_epochs,
            request.test_size,
            request.predict_field
        ))

        # Return immediate response
        return {
            "success": True,
            "dataset": request.dataset_name,
            "message": "Training started successfully. Connect to /training-loss WebSocket endpoint to receive real-time loss updates.",
            "received_parameters": {
                "num_epochs": request.num_epochs,
                "test_size": request.test_size,
                "predict_field": request.predict_field
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting training: {str(e)}"
        )

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    POST /predict

    Makes a prediction using the trained model with user-provided feature values.
    Handles both numeric and categorical features using the stored encoders.

    Args:
        request: PredictRequest with features dict

    Returns:
        JSON with prediction result and processed features info
    """
    global ann, scaler_x, scaler_y, cleaner

    try:
        # Check if model is trained
        if ann is None or scaler_x is None or scaler_y is None or cleaner is None:
            raise HTTPException(
                status_code=400,
                detail="No trained model available. Please train a model first."
            )

        # Get feature names (exclude the target column from cleaner.header)
        # We need to figure out what the target column was
        all_headers = cleaner.header.copy()
        feature_headers = [h for h in all_headers if h in request.features]

        if len(feature_headers) != len(all_headers) - 1:  # -1 for target column
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(all_headers) - 1} features, got {len(feature_headers)}. Available features: {all_headers}"
            )

        # Process the features
        processed_features = []
        feature_info = {}

        for header in all_headers:
            if header in request.features:
                value = request.features[header]

                # Check if this field is categorical (has encoders)
                if header in cleaner.encoders:
                    # Categorical field - convert to encoded value
                    value_lower = str(value).strip().lower()
                    if value_lower in cleaner.encoders[header]:
                        encoded_value = cleaner.encoders[header][value_lower]
                        processed_features.append(encoded_value)
                        feature_info[header] = {
                            "original_value": value,
                            "encoded_value": encoded_value,
                            "type": "categorical"
                        }
                    else:
                        # Unknown categorical value
                        available_values = list(cleaner.encoders[header].keys())
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unknown value '{value}' for categorical field '{header}'. Available values: {available_values}"
                        )
                else:
                    # Numeric field - convert to float
                    try:
                        numeric_value = float(value)
                        processed_features.append(numeric_value)
                        feature_info[header] = {
                            "original_value": value,
                            "numeric_value": numeric_value,
                            "type": "numeric"
                        }
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid numeric value '{value}' for field '{header}'"
                        )

        # Convert to numpy array and scale
        import numpy as np
        import torch

        features_array = np.array([processed_features])
        features_scaled = scaler_x.transform(features_array)

        # Convert to PyTorch tensor
        features_tensor = torch.from_numpy(features_scaled).float()

        # Make prediction
        ann.eval()
        with torch.no_grad():
            prediction_scaled = ann(features_tensor)
            prediction_numpy = prediction_scaled.numpy()

        # Inverse transform to get readable prediction
        prediction_readable = scaler_y.inverse_transform(prediction_numpy)
        final_prediction = float(prediction_readable[0][0])

        return {
            "success": True,
            "prediction": final_prediction,
            "processed_features": feature_info,
            "message": "Prediction completed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """
    GET /model-info

    Returns information about the trained model including field types and categorical options.

    Returns:
        JSON with model status, field types, and categorical options
    """
    global ann, scaler_x, scaler_y, cleaner

    try:
        # Check if model is trained
        if ann is None or scaler_x is None or scaler_y is None or cleaner is None:
            return {
                "model_trained": False,
                "message": "No trained model available. Please train a model first."
            }

        # Get field information
        field_info = {}
        for header in cleaner.header:
            if header in cleaner.encoders:
                # Categorical field
                field_info[header] = {
                    "type": "categorical",
                    "options": list(cleaner.encoders[header].keys())
                }
            else:
                # Numeric field
                field_info[header] = {
                    "type": "numeric"
                }

        return {
            "model_trained": True,
            "fields": field_info,
            "total_fields": len(cleaner.header),
            "message": "Model information retrieved successfully"
        }

    except Exception as e:
        print(f"Model info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )

@app.get("/dataset-distribution")
async def get_dataset_distribution(dataset_name: str):
    """
    GET /dataset-distribution

    Analyzes and returns statistical distribution data for a dataset.
    Provides histograms, summary stats, and distribution info for visualization.

    Args:
        dataset_name: Name of the dataset to analyze

    Returns:
        JSON with distribution data for all columns
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

        # Create CSVCleaner instance and analyze
        from util.csvCleaner import CSVCleaner
        import numpy as np

        cleaner = CSVCleaner(file_path)
        cleaner.load_csv()
        numeric_data = cleaner.encode_data()

        # Analyze each column
        distribution_data = {}

        for i, column_name in enumerate(cleaner.header):
            column_data = numeric_data[:, i]

            if column_name in cleaner.encoders:
                # Categorical column
                # Get original categorical values and their frequencies
                original_values = []
                encoded_values = []

                # Count occurrences of each category in original data
                category_counts = {}
                for row in cleaner.data:
                    original_value = row[i].strip().lower()
                    category_counts[original_value] = category_counts.get(original_value, 0) + 1

                # Prepare data for frontend
                for category, count in category_counts.items():
                    original_values.append(category)
                    encoded_values.append(count)

                distribution_data[column_name] = {
                    "type": "categorical",
                    "categories": original_values,
                    "counts": encoded_values,
                    "total_count": len(cleaner.data),
                    "unique_values": len(category_counts),
                    "encodings": cleaner.encoders[column_name]
                }
            else:
                # Numeric column
                # Calculate histogram and statistics
                non_nan_data = column_data[~np.isnan(column_data)] if np.any(np.isnan(column_data)) else column_data

                if len(non_nan_data) == 0:
                    continue

                # Create histogram bins
                hist_counts, bin_edges = np.histogram(non_nan_data, bins=20)

                # Calculate bin centers for plotting
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                distribution_data[column_name] = {
                    "type": "numeric",
                    "histogram": {
                        "counts": hist_counts.tolist(),
                        "bin_centers": bin_centers.tolist(),
                        "bin_edges": bin_edges.tolist()
                    },
                    "statistics": {
                        "mean": float(np.mean(non_nan_data)),
                        "median": float(np.median(non_nan_data)),
                        "std": float(np.std(non_nan_data)),
                        "min": float(np.min(non_nan_data)),
                        "max": float(np.max(non_nan_data)),
                        "q25": float(np.percentile(non_nan_data, 25)),
                        "q75": float(np.percentile(non_nan_data, 75))
                    },
                    "total_count": len(column_data),
                    "non_null_count": len(non_nan_data)
                }

        return {
            "success": True,
            "dataset": dataset_name,
            "columns": list(cleaner.header),
            "total_rows": len(cleaner.data),
            "distributions": distribution_data,
            "message": "Dataset distribution analysis completed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Distribution analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing dataset distribution: {str(e)}"
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


# ==================== TRAINING LOSS STREAMING ====================

@app.websocket("/training-loss")
async def training_loss_stream(websocket: WebSocket):
    """
    WebSocket endpoint that streams training loss data in real-time.
    Frontend connects here to receive live loss updates for graphing.
    """
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        # Keep connection alive and wait for training updates
        while True:
            # Send heartbeat to keep connection alive
            await asyncio.sleep(1)
            if not training_active:
                await websocket.send_json({"type": "status", "message": "Waiting for training to start..."})
                await asyncio.sleep(5)  # Check less frequently when not training

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

async def broadcast_loss(epoch: int, train_loss: float, test_loss: float, is_averaged=True, interval_size=2):
    """
    Broadcast training loss to all connected WebSocket clients
    """
    if active_websockets:
        message = {
            "type": "loss_update",
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "is_averaged": is_averaged,
            "interval_size": interval_size,
            "timestamp": str(asyncio.get_event_loop().time())
        }

        # Send to all connected clients
        disconnected = []
        for websocket in active_websockets:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Failed to send to websocket: {e}")
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            active_websockets.remove(ws)
