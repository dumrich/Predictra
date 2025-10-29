# 🚀 Predictra - Intelligent Data Analysis Platform

Predictra is a full-stack web application that enables users to upload CSV datasets, perform AI-powered predictive analytics, and explore data through interactive visualizations and machine learning models.

## 🌟 Features

- **📊 Dataset Management**: Upload, browse, and manage CSV datasets with thumbnail previews
- **🤖 AI-Powered Predictions**: Train neural network models using PyTorch for regression tasks
- **📈 Real-time Training Visualization**: Monitor training progress with live loss graphs via WebSocket
- **📉 Data Distribution Analysis**: Interactive histograms and statistical summaries for all dataset columns
- **💬 Interactive Chat Assistant**: Get insights about your data distributions through an AI chatbot
- **🎨 Modern UI/UX**: Beautiful, responsive interface with light/dark theme support
- **🔄 Automatic Data Preprocessing**: Handles both numeric and categorical features with encoding
- **🔍 Dataset Search**: Quickly find datasets with integrated search functionality

## 🛠️ Tech Stack

### Frontend
- **React 19** - Modern UI framework
- **React Router v7** - Client-side routing
- **Chart.js + react-chartjs-2** - Data visualization
- **Create React App** - Build tooling and development server

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning library for neural networks
- **scikit-learn** - Data preprocessing and train/test splitting
- **NumPy** - Numerical computations
- **WebSocket** - Real-time training loss streaming
- **Uvicorn** - ASGI web server

### Utilities
- **Custom CSV Cleaner** - Automatic feature detection and encoding
- **Dynamic Neural Network** - Configurable ANN architecture with dropout regularization

## 📁 Project Structure

```
Predictra/
├── api/
│   └── main.py                 # FastAPI backend application
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── LibraryPage.jsx
│   │   │   ├── AnalysisPage.jsx
│   │   │   ├── TrainingGraph.jsx
│   │   │   ├── DataVisualization.jsx
│   │   │   ├── ChatBot.jsx
│   │   │   └── ...
│   │   ├── contexts/          # React contexts
│   │   ├── config.js          # API configuration
│   │   └── App.js             # Main app component
│   └── package.json
├── util/
│   ├── csvCleaner.py          # CSV preprocessing utility
│   └── createNeuralNet.py     # Neural network creation
├── datasets/                   # CSV dataset storage
├── thumbnails/                 # Dataset thumbnail images
└── venv/                       # Python virtual environment
```

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.12+** (or compatible Python 3.x)
- **Node.js 16+** and npm
- **Git** (optional, for cloning)

### Backend Setup

1. **Navigate to the project directory:**
   ```bash
   cd Predictra
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install Python dependencies:**
   ```bash
   pip install fastapi uvicorn torch scikit-learn numpy
   ```

   If you prefer to install from a requirements file, create `requirements.txt`:
   ```
   fastapi>=0.104.0
   uvicorn[standard]>=0.24.0
   torch>=2.0.0
   scikit-learn>=1.3.0
   numpy>=1.24.0
   pydantic>=2.0.0
   python-multipart>=0.0.6
   websockets>=12.0
   ```

4. **Prepare directories:**
   ```bash
   mkdir -p datasets thumbnails
   ```

5. **Start the FastAPI server:**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at:
   - **API Base**: http://localhost:8000
   - **Interactive Docs**: http://localhost:8000/docs
   - **Alternative Docs**: http://localhost:8000/redoc

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Configure API endpoint** (if needed):
   
   Edit `frontend/src/config.js` to match your backend URL:
   ```javascript
   BASE_URL: "http://localhost:8000"
   ```

4. **Start the development server:**
   ```bash
   npm start
   ```

   The React app will open at http://localhost:3000

## 📖 Usage Guide

### 1. Upload a Dataset

- Click **"📁 Choose CSV File"** on the homepage
- Select a CSV file from your computer
- Click **"🚀 Upload"** to upload to the server
- The dataset will appear in your library after upload

### 2. Analyze a Dataset

- Click on any dataset card to open the Analysis Page
- The page automatically fetches and displays dataset headers
- Select a target field (column) you want to predict
- View distribution visualizations for all columns

### 3. Train a Model

1. On the Analysis Page, select your **target field** (what you want to predict)
2. Configure training parameters:
   - **Epochs**: Number of training iterations (default: 10)
   - **Test Size**: Proportion of data for testing (default: 0.1)
3. Click **"🚀 Train Model"**
4. Monitor training progress in real-time via the Training Graph
5. Training loss updates stream via WebSocket every 2 epochs

### 4. Predictions

1. After training completes, scroll to the Prediction Section
2. Fill in feature values based on the form generated from your dataset
3. For categorical fields, select from available options
4. For numeric fields, enter numeric values
5. Click **"🔮 Predict"** to get your prediction
6. View the predicted value and processed feature information

### 5. Distributions

- Click **"📊 View Distributions"** to analyze column distributions
- View histograms for numeric columns
- See category counts for categorical columns
- Interact with the ChatBot to ask questions about distributions

## 🔌 API Endpoints

### Dataset Management
- `GET /libraries` - List all available datasets
- `GET /libraries/{library_name}` - Get specific dataset info
- `POST /upload` - Upload a new CSV file
- `POST /rescan` - Rescan datasets folder

### Analysis & Training
- `GET /analyze?dataset_name={name}` - Get dataset headers
- `GET /dataset-distribution?dataset_name={name}` - Get distribution data
- `POST /train` - Start model training
- `GET /model-info` - Get trained model information

### Predictions
- `POST /predict` - Make a prediction with feature values

### Real-time Training
- `WS /training-loss` - WebSocket endpoint for live training loss updates

## 🧠 Machine Learning Details

### Neural Network Architecture

The default model uses a multi-layer perceptron with:
- **Input Layer**: Dynamic size based on dataset features
- **Hidden Layer 1**: 64 neurons + ReLU + Dropout (0.2)
- **Hidden Layer 2**: 64 neurons + ReLU + Dropout (0.2)
- **Hidden Layer 3**: 32 neurons + ReLU + Dropout (0.2)
- **Output Layer**: 1 neuron (for regression)

### Data Preprocessing

- **Feature Detection**: Automatically identifies numeric vs categorical columns
- **Categorical Encoding**: Label encoding with stored mappings
- **Data Scaling**: StandardScaler applied to both features and target
- **Train/Test Split**: Configurable split ratio (default: 0.2)

### Training Configuration

- **Optimizer**: Adam (learning rate: 0.001, weight decay: 1e-5)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **WebSocket Updates**: Averaged loss sent every 2 epochs

## 📊 Example Datasets

The project includes several example datasets:
- `housing.csv` - Housing price prediction
- `heart.csv` - Heart disease data
- `breast_cancer.csv` - Medical classification data
- `lebron.csv` - Basketball statistics
- `crop_production.csv` - Agricultural data
- And more...

