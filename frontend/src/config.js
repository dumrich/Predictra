// Configuration for the FastAPI backend
export const API_CONFIG = {
  // Point to your FastAPI server (running on port 80)
  BASE_URL: "http://localhost:8000",
  
  // API endpoints
  ENDPOINTS: {
    LIBRARIES: "/libraries",
    UPLOAD: "/upload",
    THUMBNAILS: "/thumbnails",
    ANALYZE: "/analyze",
    TRAIN: "/train",
    PREDICT: "/predict",
    MODEL_INFO: "/model-info",
    DATASET_DISTRIBUTION: "/dataset-distribution"
  }
};

// Helper function to get full API URL
export const getApiUrl = (endpoint) => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};
