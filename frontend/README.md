# Frontend for FastAPI Dataset Library

This React frontend is designed to work with a FastAPI backend for managing and uploading datasets.

## Features

- View available datasets in a grid layout
- Upload new CSV files
- Responsive design with modern UI
- Configurable API endpoints

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure the API URL:
   - Create a `.env` file in the frontend directory
   - Add: `REACT_APP_API_URL=http://localhost:8000` (or your FastAPI server URL)

3. Start the development server:
```bash
npm start
```

## API Endpoints Expected

The frontend expects the following FastAPI endpoints:

- `GET /libraries` - Returns a list of available datasets
- `POST /upload` - Accepts CSV file uploads
- `GET /thumbnails/{filename}` - Serves dataset thumbnails

## Expected API Response Format

### Libraries Endpoint
```json
{
  "libraries": [
    {
      "name": "dataset_name",
      "thumbnail": "/thumbnails/dataset.png",
      "description": "Dataset description"
    }
  ]
}
```

### Upload Endpoint
```json
{
  "message": "File uploaded successfully",
  "filename": "uploaded_file.csv"
}
```

## Environment Variables

- `REACT_APP_API_URL`: The base URL of your FastAPI backend (default: http://localhost:8000)

## Development

The app will automatically reload when you make changes. The build process will also include any linting errors in the console.