import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { getApiUrl, API_CONFIG } from "../config";

export default function AnalysisPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const dataset = location.state?.dataset;
  
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [epochs, setEpochs] = useState(10);
  const [textSize, setTextSize] = useState(0.1);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [trainingError, setTrainingError] = useState(null);

  const handleBackToLibrary = () => {
    navigate("/");
  };

  // Fetch analysis data from FastAPI backend
  const fetchAnalysis = async () => {
    if (!dataset) {
      setAnalysisError("No dataset selected");
      return;
    }

    setAnalysisLoading(true);
    setAnalysisError(null);

    try {
      // Pass dataset name as query parameter
      const apiUrl = `${getApiUrl(API_CONFIG.ENDPOINTS.ANALYZE)}?dataset_name=${encodeURIComponent(dataset.name)}`;
      console.log("Fetching analysis from:", apiUrl);
      console.log("Dataset:", dataset);

      const res = await fetch(apiUrl, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      console.log("Analysis response status:", res.status);

      if (!res.ok) {
        throw new Error(`Analysis failed with status ${res.status}`);
      }

      const data = await res.json();
      console.log("Analysis data:", data);
      setAnalysisData(data);
    } catch (err) {
      console.error("Error fetching analysis:", err);
      setAnalysisError(err.message);
    } finally {
      setAnalysisLoading(false);
    }
  };

  // Train model with selected parameters
  const handleTrain = async () => {
    if (!selectedAnalysis || !dataset) {
      setTrainingError("Please select an analysis type and ensure dataset is available");
      return;
    }

    setTrainingLoading(true);
    setTrainingError(null);
    setTrainingResult(null);

    try {
      const apiUrl = getApiUrl(API_CONFIG.ENDPOINTS.TRAIN);
      console.log("Training model at:", apiUrl);
      console.log("Training parameters:", {
        num_epochs: epochs,
        test_size: textSize,
        predict_field: selectedAnalysis
      });

      // Pass parameters as query parameters to match FastAPI function signature
      const params = new URLSearchParams({
        num_epochs: epochs.toString(),
        test_size: textSize.toString(),
        predict_field: selectedAnalysis
      });
      
      const response = await fetch(`${apiUrl}?${params}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      console.log("Training response status:", response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Training failed with status ${response.status}`);
      }

      const data = await response.json();
      console.log("Training result:", data);
      setTrainingResult(data);
      
      // Auto-scroll to prediction section after training completes
      setTimeout(() => {
        const predictionSection = document.getElementById('prediction-section');
        if (predictionSection) {
          predictionSection.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
          });
        }
      }, 500);
    } catch (err) {
      console.error("Error training model:", err);
      setTrainingError(err.message);
    } finally {
      setTrainingLoading(false);
    }
  };

  // Fetch analysis when component mounts
  useEffect(() => {
    fetchAnalysis();
  }, [dataset]);

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)",
      padding: "0",
      position: "relative",
      overflow: "hidden"
    }}>
      {/* Animated Background Elements */}
      <div style={{
        position: "absolute",
        top: "-50%",
        left: "-50%",
        width: "200%",
        height: "200%",
        background: "radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px)",
        backgroundSize: "50px 50px",
        animation: "float 20s ease-in-out infinite"
      }}></div>
      
      <div style={{
        position: "absolute",
        top: "20%",
        right: "-10%",
        width: "300px",
        height: "300px",
        background: "linear-gradient(45deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))",
        borderRadius: "50%",
        animation: "pulse 4s ease-in-out infinite"
      }}></div>
      
      <div style={{
        position: "absolute",
        bottom: "10%",
        left: "-5%",
        width: "200px",
        height: "200px",
        background: "linear-gradient(45deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03))",
        borderRadius: "50%",
        animation: "pulse 6s ease-in-out infinite reverse"
      }}></div>

      {/* Header */}
      <div style={{
        background: "rgba(255, 255, 255, 0.1)",
        backdropFilter: "blur(20px)",
        padding: "30px 0",
        boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
        position: "sticky",
        top: 0,
        zIndex: 100,
        borderBottom: "1px solid rgba(255,255,255,0.2)"
      }}>
        <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 20px" }}>
          <div style={{ 
            display: "flex", 
            justifyContent: "space-between", 
            alignItems: "center" 
          }}>
            <div>
              <h1 
                onClick={() => navigate("/")}
                style={{ 
                  fontSize: "48px", 
                  fontWeight: "800", 
                  margin: 0,
                  background: "linear-gradient(135deg, #ffffff, #f0f0f0, #ffffff)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  textShadow: "0 0 30px rgba(255,255,255,0.5)",
                  letterSpacing: "-1px",
                  cursor: "pointer",
                  transition: "all 0.3s ease",
                  display: "inline-block"
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.05)";
                  e.currentTarget.style.textShadow = "0 0 40px rgba(255,255,255,0.7)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.style.textShadow = "0 0 30px rgba(255,255,255,0.5)";
                }}
              >
                🚀 Predictra
              </h1>
              <p style={{ 
                color: "rgba(255,255,255,0.9)", 
                margin: "8px 0 0 0",
                fontSize: "18px",
                fontWeight: "300",
                textShadow: "0 2px 10px rgba(0,0,0,0.3)"
              }}>
                AI-Powered Data Analytics Platform
              </p>
            </div>
            
            <button
              onClick={handleBackToLibrary}
              style={{
                background: "rgba(255,255,255,0.2)",
                color: "rgba(255,255,255,0.9)",
                border: "1px solid rgba(255,255,255,0.3)",
                borderRadius: "25px",
                padding: "12px 24px",
                fontSize: "16px",
                fontWeight: "600",
                cursor: "pointer",
                transition: "all 0.3s ease",
                backdropFilter: "blur(10px)"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(255,255,255,0.3)";
                e.currentTarget.style.transform = "translateY(-2px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(255,255,255,0.2)";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              ← Back to Library
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "50px 20px", position: "relative", zIndex: 10 }}>
        
        {/* Dataset Info Section */}
        {dataset && (
          <div style={{
            background: "rgba(255, 255, 255, 0.95)",
            borderRadius: "24px",
            padding: "40px",
            boxShadow: "0 25px 50px rgba(0,0,0,0.15)",
            marginBottom: "40px",
            backdropFilter: "blur(10px)",
            border: "1px solid rgba(255,255,255,0.3)"
          }}>
            <div style={{ 
              display: "flex", 
              alignItems: "center",
              marginBottom: "30px"
            }}>
              {/* Thumbnail Image */}
              <div style={{
                width: "120px",
                height: "120px",
                borderRadius: "20px",
                overflow: "hidden",
                marginRight: "30px",
                background: "linear-gradient(135deg, #667eea, #764ba2, #f093fb)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 8px 25px rgba(0,0,0,0.15)"
              }}>
                {dataset.thumbnail ? (
                  <img
                    src={`http://100.109.58.81:80${dataset.thumbnail}`}
                    alt={dataset.name}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover"
                    }}
                    onError={(e) => {
                      e.target.style.display = "none";
                      e.target.nextSibling.style.display = "flex";
                    }}
                  />
                ) : null}
                <div style={{
                  display: dataset.thumbnail ? "none" : "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "48px",
                  color: "rgba(255,255,255,0.9)",
                  textShadow: "0 4px 20px rgba(0,0,0,0.3)"
                }}>
                  📊
                </div>
              </div>
              
              {/* Dataset Info */}
              <div style={{ flex: 1 }}>
                <h2 style={{ 
                  fontSize: "32px", 
                  fontWeight: "700", 
                  color: "#2d3748",
                  margin: "0 0 8px 0",
                  background: "linear-gradient(135deg, #667eea, #764ba2)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent"
                }}>
                  {dataset.name}
                </h2>
                <p style={{
                  color: "#718096",
                  margin: "0 0 12px 0",
                  fontSize: "16px",
                  fontWeight: "400"
                }}>
                  📁 {dataset.csv_file || 'Unknown file'}
                </p>
                <div style={{
                  display: "inline-block",
                  background: "linear-gradient(135deg, #667eea, #764ba2)",
                  color: "white",
                  padding: "8px 20px",
                  borderRadius: "20px",
                  fontSize: "14px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  boxShadow: "0 4px 15px rgba(102, 126, 234, 0.3)"
                }}>
                  ✨ Ready for AI Analysis
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analysis Section */}
        <div style={{
          background: "rgba(255, 255, 255, 0.95)",
          borderRadius: "24px",
          padding: "50px",
          boxShadow: "0 25px 50px rgba(0,0,0,0.15)",
          backdropFilter: "blur(10px)",
          border: "1px solid rgba(255,255,255,0.3)",
          textAlign: "center"
        }}>
          <div style={{ marginBottom: "40px" }}>
            <h2 style={{ 
              fontSize: "32px", 
              fontWeight: "700", 
              color: "#2d3748",
              margin: "0 0 12px 0",
              background: "linear-gradient(135deg, #667eea, #764ba2)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent"
            }}>
              🔬 Data Analysis
            </h2>
            <p style={{ 
              color: "#718096", 
              margin: 0,
              fontSize: "18px",
              fontWeight: "400"
            }}>
              AI-powered insights and predictions
            </p>
          </div>

          {/* Analysis Loading State */}
          {analysisLoading && (
            <div style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              padding: "60px 0",
              flexDirection: "column"
            }}>
              <div style={{
                width: "60px",
                height: "60px",
                border: "4px solid #e2e8f0",
                borderTop: "4px solid #667eea",
                borderRadius: "50%",
                animation: "spin 1s linear infinite"
              }}></div>
              <p style={{ 
                marginTop: "24px", 
                color: "#718096",
                fontSize: "18px",
                fontWeight: "500"
              }}>
                Analyzing dataset...
              </p>
            </div>
          )}

          {/* Analysis Error State */}
          {analysisError && !analysisLoading && (
            <div style={{
              padding: "30px",
              borderRadius: "16px",
              background: "#fed7d7",
              color: "#c53030",
              border: "1px solid #feb2b2",
              marginBottom: "30px"
            }}>
              <h3 style={{ margin: "0 0 12px 0", fontSize: "18px", fontWeight: "600" }}>
                ❌ Analysis Failed
              </h3>
              <p style={{ margin: 0, fontSize: "16px" }}>
                {analysisError}
              </p>
              <button
                onClick={fetchAnalysis}
                style={{
                  marginTop: "16px",
                  background: "#c53030",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  padding: "8px 16px",
                  fontSize: "14px",
                  fontWeight: "600",
                  cursor: "pointer"
                }}
              >
                🔄 Retry Analysis
              </button>
            </div>
          )}

          {/* Analysis Results */}
          {analysisData && !analysisLoading && (
            <div style={{
              textAlign: "left",
              background: "rgba(255,255,255,0.8)",
              borderRadius: "16px",
              padding: "30px",
              marginBottom: "30px",
              border: "1px solid rgba(102, 126, 234, 0.2)"
            }}>
              <h3 style={{ 
                margin: "0 0 20px 0", 
                color: "#2d3748", 
                fontSize: "20px", 
                fontWeight: "600",
                textAlign: "center"
              }}>
                📊 Analysis Options
              </h3>
              
              {/* Dropdown Menu */}
              <div style={{ marginBottom: "20px" }}>
                <label style={{
                  display: "block",
                  marginBottom: "8px",
                  fontSize: "16px",
                  fontWeight: "600",
                  color: "#4a5568"
                }}>
                  Select Analysis Type:
                </label>
                <select
                  value={selectedAnalysis || ""}
                  onChange={(e) => setSelectedAnalysis(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "12px 16px",
                    borderRadius: "12px",
                    border: "2px solid #e2e8f0",
                    fontSize: "16px",
                    fontWeight: "500",
                    color: "#2d3748",
                    background: "white",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = "#667eea";
                    e.target.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.2)";
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = "#e2e8f0";
                    e.target.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
                  }}
                >
                  <option value="" disabled>
                    Choose an analysis option...
                  </option>
                  {Array.isArray(analysisData?.headers) ? analysisData.headers.map((option, index) => (
                    <option key={index} value={option}>
                      {option}
                    </option>
                  )) : Array.isArray(analysisData) ? analysisData.map((option, index) => (
                    <option key={index} value={option}>
                      {option}
                    </option>
                  )) : (
                    <option value={analysisData}>
                      {analysisData}
                    </option>
                  )}
                </select>
              </div>

              {/* Epochs Input */}
              <div style={{ marginBottom: "20px" }}>
                <label style={{
                  display: "block",
                  marginBottom: "8px",
                  fontSize: "16px",
                  fontWeight: "600",
                  color: "#4a5568"
                }}>
                  Number of epochs:
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={epochs}
                    onChange={(e) => {
                      const value = parseInt(e.target.value);
                      if (value >= 1 && value <= 100) {
                        setEpochs(value);
                      }
                    }}
                    style={{
                      width: "120px",
                      padding: "12px 16px",
                      borderRadius: "12px",
                      border: "2px solid #e2e8f0",
                      fontSize: "16px",
                      fontWeight: "500",
                      color: "#2d3748",
                      background: "white",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = "#667eea";
                      e.target.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.2)";
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = "#e2e8f0";
                      e.target.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
                    }}
                  />
                  <div style={{
                    background: "linear-gradient(135deg, #667eea, #764ba2)",
                    color: "white",
                    padding: "8px 16px",
                    borderRadius: "20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                    boxShadow: "0 4px 15px rgba(102, 126, 234, 0.3)"
                  }}>
                    Epochs
                  </div>
                  <div style={{
                    fontSize: "14px",
                    color: "#718096",
                    fontWeight: "500"
                  }}>
                    (1-100)
                  </div>
                </div>
                <div style={{
                  marginTop: "8px",
                  fontSize: "12px",
                  color: "#a0aec0",
                  fontStyle: "italic"
                }}>
                  💡 Higher epochs = more training iterations, but longer processing time
                </div>
              </div>

              {/* Text Size Input */}
              <div style={{ marginBottom: "20px" }}>
                <label style={{
                  display: "block",
                  marginBottom: "8px",
                  fontSize: "16px",
                  fontWeight: "600",
                  color: "#4a5568"
                }}>
                  Test size:
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <input
                    type="number"
                    min="0"
                    max="0.3"
                    step="0.01"
                    value={textSize}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value);
                      if (value >= 0 && value <= 0.3) {
                        setTextSize(value);
                      }
                    }}
                    style={{
                      width: "120px",
                      padding: "12px 16px",
                      borderRadius: "12px",
                      border: "2px solid #e2e8f0",
                      fontSize: "16px",
                      fontWeight: "500",
                      color: "#2d3748",
                      background: "white",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = "#667eea";
                      e.target.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.2)";
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = "#e2e8f0";
                      e.target.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
                    }}
                  />
                  <div style={{
                    background: "linear-gradient(135deg, #667eea, #764ba2)",
                    color: "white",
                    padding: "8px 16px",
                    borderRadius: "20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                    boxShadow: "0 4px 15px rgba(102, 126, 234, 0.3)"
                  }}>
                    Test
                  </div>
                  <div style={{
                    fontSize: "14px",
                    color: "#718096",
                    fontWeight: "500"
                  }}>
                    (0-0.3)
                  </div>
                </div>
                <div style={{
                  marginTop: "8px",
                  fontSize: "12px",
                  color: "#a0aec0",
                  fontStyle: "italic"
                }}>
                  💡 Controls the proportion of data used for testing (0-0.3)
                </div>
              </div>

              {/* Selected Analysis Display */}
              {selectedAnalysis && (
                <div style={{
                  background: "#f7fafc",
                  borderRadius: "12px",
                  padding: "20px",
                  border: "1px solid #e2e8f0",
                  marginTop: "20px"
                }}>
                  <h4 style={{
                    margin: "0 0 12px 0",
                    color: "#2d3748",
                    fontSize: "18px",
                    fontWeight: "600"
                  }}>
                    Selected Analysis: {selectedAnalysis}
                  </h4>
                  <p style={{
                    margin: "0 0 8px 0",
                    color: "#718096",
                    fontSize: "14px",
                    lineHeight: "1.5"
                  }}>
                    Analysis type: <strong>{selectedAnalysis}</strong>
                  </p>
                  <p style={{
                    margin: "0 0 8px 0",
                    color: "#718096",
                    fontSize: "14px",
                    lineHeight: "1.5"
                  }}>
                    Number of epochs: <strong>{epochs}</strong>
                  </p>
                  <p style={{
                    margin: "0 0 16px 0",
                    color: "#718096",
                    fontSize: "14px",
                    lineHeight: "1.5"
                  }}>
                    Test size: <strong>{textSize}</strong>
                  </p>
                  <p style={{
                    margin: 0,
                    color: "#718096",
                    fontSize: "14px",
                    lineHeight: "1.5"
                  }}>
                    Ready to run analysis with these parameters.
                  </p>
                  
                  {/* Action Button */}
                  <button
                    onClick={handleTrain}
                    disabled={trainingLoading}
                    style={{
                      marginTop: "16px",
                      background: trainingLoading ? "#e2e8f0" : "linear-gradient(135deg, #667eea, #764ba2)",
                      color: trainingLoading ? "#a0aec0" : "white",
                      border: "none",
                      borderRadius: "10px",
                      padding: "12px 24px",
                      fontSize: "16px",
                      fontWeight: "600",
                      cursor: trainingLoading ? "not-allowed" : "pointer",
                      transition: "all 0.3s ease",
                      boxShadow: trainingLoading ? "none" : "0 4px 15px rgba(102, 126, 234, 0.3)"
                    }}
                    onMouseEnter={(e) => {
                      if (!trainingLoading) {
                        e.currentTarget.style.transform = "translateY(-2px)";
                        e.currentTarget.style.boxShadow = "0 6px 20px rgba(102, 126, 234, 0.4)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!trainingLoading) {
                        e.currentTarget.style.transform = "translateY(0)";
                        e.currentTarget.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.3)";
                      }
                    }}
                  >
                    {trainingLoading ? "🔄 Training..." : "🚀 Train Model"}
                  </button>
                </div>
              )}

              {/* Training Loading State */}
              {trainingLoading && (
                <div style={{
                  marginTop: "30px",
                  padding: "30px",
                  borderRadius: "16px",
                  background: "#fef5e7",
                  border: "1px solid #fbd38d",
                  textAlign: "center"
                }}>
                  <div style={{
                    width: "40px",
                    height: "40px",
                    border: "4px solid #fbd38d",
                    borderTop: "4px solid #f6ad55",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                    margin: "0 auto 16px auto"
                  }}></div>
                  <h3 style={{ margin: "0 0 8px 0", color: "#c05621", fontSize: "18px", fontWeight: "600" }}>
                    🔄 Training Model
                  </h3>
                  <p style={{ margin: 0, color: "#c05621", fontSize: "14px" }}>
                    Please wait while the model is being trained with your parameters...
                  </p>
                </div>
              )}

              {/* Training Error State */}
              {trainingError && !trainingLoading && (
                <div style={{
                  marginTop: "30px",
                  padding: "20px",
                  borderRadius: "16px",
                  background: "#fed7d7",
                  color: "#c53030",
                  border: "1px solid #feb2b2"
                }}>
                  <h3 style={{ margin: "0 0 12px 0", fontSize: "18px", fontWeight: "600" }}>
                    ❌ Training Failed
                  </h3>
                  <p style={{ margin: "0 0 16px 0", fontSize: "14px" }}>
                    {trainingError}
                  </p>
                  <button
                    onClick={handleTrain}
                    style={{
                      background: "#c53030",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "8px 16px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer"
                    }}
                  >
                    🔄 Retry Training
                  </button>
                </div>
              )}

              {/* Training Success State */}
              {trainingResult && !trainingLoading && (
                <div style={{
                  marginTop: "30px",
                  padding: "20px",
                  borderRadius: "16px",
                  background: "#f0fff4",
                  border: "1px solid #9ae6b4"
                }}>
                  <h3 style={{ margin: "0 0 12px 0", fontSize: "18px", fontWeight: "600", color: "#22543d" }}>
                    ✅ Training Complete
                  </h3>
                  <p style={{ margin: "0 0 16px 0", fontSize: "14px", color: "#22543d" }}>
                    Model has been successfully trained with your parameters.
                  </p>
                  <details style={{ marginTop: "16px" }}>
                    <summary style={{
                      cursor: "pointer",
                      fontSize: "14px",
                      color: "#22543d",
                      fontWeight: "500"
                    }}>
                      🔍 View Training Results
                    </summary>
                    <pre style={{
                      background: "#f7fafc",
                      padding: "16px",
                      borderRadius: "8px",
                      overflow: "auto",
                      fontSize: "12px",
                      color: "#4a5568",
                      border: "1px solid #e2e8f0",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      marginTop: "12px"
                    }}>
                      {JSON.stringify(trainingResult, null, 2)}
                    </pre>
                  </details>
                </div>
              )}

              {/* Debug Info (collapsible) */}
              <details style={{ marginTop: "20px" }}>
                <summary style={{
                  cursor: "pointer",
                  fontSize: "14px",
                  color: "#718096",
                  fontWeight: "500",
                  padding: "8px 0"
                }}>
                  🔍 View Raw Data
                </summary>
                <pre style={{
                  background: "#f7fafc",
                  padding: "20px",
                  borderRadius: "12px",
                  overflow: "auto",
                  fontSize: "12px",
                  color: "#4a5568",
                  border: "1px solid #e2e8f0",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  marginTop: "12px"
                }}>
                  {JSON.stringify(analysisData, null, 2)}
                </pre>
              </details>
            </div>
          )}

          {/* Placeholder Content (when no analysis data and not loading) */}
          {!analysisData && !analysisLoading && !analysisError && (
            <div style={{
              border: "3px dashed #cbd5e0",
              borderRadius: "20px",
              padding: "80px 30px",
              background: "linear-gradient(135deg, #f7fafc, #edf2f7)",
              marginBottom: "40px",
              boxShadow: "inset 0 2px 10px rgba(0,0,0,0.05)"
            }}>
              <div style={{ fontSize: "80px", marginBottom: "24px" }}>🤖</div>
              <h3 style={{ 
                fontSize: "24px", 
                margin: "0 0 16px 0", 
                color: "#4a5568",
                fontWeight: "600"
              }}>
                Analysis Features Coming Soon
              </h3>
              <p style={{ 
                margin: "0 0 20px 0", 
                color: "#718096",
                fontSize: "16px",
                fontWeight: "400"
              }}>
                We're building powerful AI tools to help you discover insights in your data
              </p>
              
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: "20px",
                marginTop: "30px"
              }}>
                <div style={{
                  background: "rgba(255,255,255,0.8)",
                  padding: "20px",
                  borderRadius: "16px",
                  border: "1px solid rgba(102, 126, 234, 0.2)"
                }}>
                  <div style={{ fontSize: "32px", marginBottom: "12px" }}>📈</div>
                  <h4 style={{ margin: "0 0 8px 0", color: "#2d3748", fontSize: "16px", fontWeight: "600" }}>
                    Statistical Analysis
                  </h4>
                  <p style={{ margin: 0, color: "#718096", fontSize: "14px" }}>
                    Descriptive statistics and data profiling
                  </p>
                </div>
                
                <div style={{
                  background: "rgba(255,255,255,0.8)",
                  padding: "20px",
                  borderRadius: "16px",
                  border: "1px solid rgba(102, 126, 234, 0.2)"
                }}>
                  <div style={{ fontSize: "32px", marginBottom: "12px" }}>🔮</div>
                  <h4 style={{ margin: "0 0 8px 0", color: "#2d3748", fontSize: "16px", fontWeight: "600" }}>
                    Predictive Modeling
                  </h4>
                  <p style={{ margin: 0, color: "#718096", fontSize: "14px" }}>
                    Machine learning predictions and forecasts
                  </p>
                </div>
                
                <div style={{
                  background: "rgba(255,255,255,0.8)",
                  padding: "20px",
                  borderRadius: "16px",
                  border: "1px solid rgba(102, 126, 234, 0.2)"
                }}>
                  <div style={{ fontSize: "32px", marginBottom: "12px" }}>📊</div>
                  <h4 style={{ margin: "0 0 8px 0", color: "#2d3748", fontSize: "16px", fontWeight: "600" }}>
                    Data Visualization
                  </h4>
                  <p style={{ margin: 0, color: "#718096", fontSize: "14px" }}>
                    Interactive charts and graphs
                  </p>
                </div>
              </div>
            </div>
          )}

          <button
            onClick={handleBackToLibrary}
            style={{
              background: "linear-gradient(135deg, #667eea, #764ba2)",
              color: "white",
              border: "none",
              borderRadius: "16px",
              padding: "16px 40px",
              fontSize: "18px",
              fontWeight: "700",
              cursor: "pointer",
              transition: "all 0.3s ease",
              boxShadow: "0 8px 25px rgba(102, 126, 234, 0.4)",
              textTransform: "uppercase",
              letterSpacing: "0.5px"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow = "0 12px 35px rgba(102, 126, 234, 0.5)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "0 8px 25px rgba(102, 126, 234, 0.4)";
            }}
          >
            ← Return to Library
          </button>
        </div>

        {/* Prediction Section - Shows after training */}
        {trainingResult && !trainingLoading && (
          <div id="prediction-section" style={{
            marginTop: "60px",
            padding: "50px 20px",
            background: "rgba(255, 255, 255, 0.95)",
            borderRadius: "24px",
            boxShadow: "0 25px 50px rgba(0,0,0,0.15)",
            backdropFilter: "blur(10px)",
            border: "1px solid rgba(255,255,255,0.3)"
          }}>
            <div style={{ textAlign: "center", marginBottom: "40px" }}>
              <h2 style={{ 
                fontSize: "32px", 
                fontWeight: "700", 
                color: "#2d3748",
                margin: "0 0 12px 0",
                background: "linear-gradient(135deg, #667eea, #764ba2)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent"
              }}>
                🔮 Model Predictions
              </h2>
              <p style={{ 
                color: "#718096", 
                margin: 0,
                fontSize: "18px",
                fontWeight: "400"
              }}>
                Your trained model is ready for predictions
              </p>
            </div>

            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "40px",
              alignItems: "start"
            }}>
              {/* Left Side - Graph Placeholder */}
              <div style={{
                background: "linear-gradient(135deg, #f7fafc, #edf2f7)",
                borderRadius: "20px",
                padding: "40px",
                border: "2px dashed #cbd5e0",
                textAlign: "center",
                minHeight: "400px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center"
              }}>
                <div style={{
                  fontSize: "80px",
                  marginBottom: "20px",
                  opacity: 0.6
                }}>
                  📊
                </div>
                <h3 style={{
                  fontSize: "24px",
                  fontWeight: "600",
                  color: "#4a5568",
                  margin: "0 0 12px 0"
                }}>
                  Prediction Graph
                </h3>
                <p style={{
                  color: "#718096",
                  fontSize: "16px",
                  margin: 0,
                  lineHeight: "1.5"
                }}>
                  Results will be displayed here after running predictions
                </p>
              </div>

              {/* Right Side - Analysis Type Dropdowns */}
              <div style={{
                background: "rgba(255,255,255,0.8)",
                borderRadius: "20px",
                padding: "30px",
                border: "1px solid rgba(102, 126, 234, 0.2)"
              }}>
                <h3 style={{
                  fontSize: "20px",
                  fontWeight: "600",
                  color: "#2d3748",
                  margin: "0 0 24px 0",
                  textAlign: "center"
                }}>
                  📊 Analysis Options
                </h3>

                {/* Analysis Type Dropdowns */}
                {Array.isArray(analysisData?.headers) ? analysisData.headers
                  .filter(option => option !== selectedAnalysis)
                  .map((option, index) => (
                  <div key={index} style={{ marginBottom: "16px" }}>
                    <label style={{
                      display: "block",
                      marginBottom: "8px",
                      fontSize: "14px",
                      fontWeight: "600",
                      color: "#4a5568"
                    }}>
                      {option}:
                    </label>
                    <select
                      style={{
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "12px",
                        border: "2px solid #e2e8f0",
                        fontSize: "16px",
                        fontWeight: "500",
                        color: "#2d3748",
                        background: "white",
                        cursor: "pointer",
                        transition: "all 0.3s ease",
                        boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                      }}
                      onFocus={(e) => {
                        e.target.style.borderColor = "#667eea";
                        e.target.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.2)";
                      }}
                      onBlur={(e) => {
                        e.target.style.borderColor = "#e2e8f0";
                        e.target.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
                      }}
                    >
                      <option value="" disabled>
                        Select {option}...
                      </option>
                      {/* You can add specific options for each analysis type here */}
                      <option value="option1">Option 1</option>
                      <option value="option2">Option 2</option>
                      <option value="option3">Option 3</option>
                    </select>
                  </div>
                )) : (
                  <div style={{ marginBottom: "16px" }}>
                    <label style={{
                      display: "block",
                      marginBottom: "8px",
                      fontSize: "14px",
                      fontWeight: "600",
                      color: "#4a5568"
                    }}>
                      Analysis Type:
                    </label>
                    <select
                      style={{
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "12px",
                        border: "2px solid #e2e8f0",
                        fontSize: "16px",
                        fontWeight: "500",
                        color: "#2d3748",
                        background: "white",
                        cursor: "pointer",
                        transition: "all 0.3s ease",
                        boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                      }}
                      onFocus={(e) => {
                        e.target.style.borderColor = "#667eea";
                        e.target.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.2)";
                      }}
                      onBlur={(e) => {
                        e.target.style.borderColor = "#e2e8f0";
                        e.target.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
                      }}
                    >
                      <option value="" disabled>
                        Select analysis type...
                      </option>
                      <option value="option1">Option 1</option>
                      <option value="option2">Option 2</option>
                      <option value="option3">Option 3</option>
                    </select>
                  </div>
                )}

                {/* Selected Analysis Type Display */}
                <div style={{ 
                  marginTop: "24px", 
                  marginBottom: "24px",
                  padding: "16px",
                  background: "linear-gradient(135deg, #667eea, #764ba2)",
                  borderRadius: "12px",
                  color: "white"
                }}>
                  <div style={{
                    fontSize: "14px",
                    fontWeight: "600",
                    marginBottom: "8px",
                    opacity: 0.9
                  }}>
                    Selected Analysis Type:
                  </div>
                  <div style={{
                    fontSize: "18px",
                    fontWeight: "700"
                  }}>
                    {selectedAnalysis}
                  </div>
                </div>

                {/* Predict Button */}
                <button
                  style={{
                    width: "100%",
                    background: "linear-gradient(135deg, #667eea, #764ba2)",
                    color: "white",
                    border: "none",
                    borderRadius: "16px",
                    padding: "16px 24px",
                    fontSize: "18px",
                    fontWeight: "700",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                    boxShadow: "0 8px 25px rgba(102, 126, 234, 0.4)",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-2px)";
                    e.currentTarget.style.boxShadow = "0 12px 35px rgba(102, 126, 234, 0.5)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "0 8px 25px rgba(102, 126, 234, 0.4)";
                  }}
                >
                  🔮 Predict
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 0.7; }
          50% { transform: scale(1.1); opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
