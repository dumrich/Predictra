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