import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import DatasetCard from "./DatasetCard";
import { getApiUrl, API_CONFIG } from "../config";

export default function LibraryPage() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  // Fetch datasets from FastAPI backend
  const fetchLibraries = async () => {
    setLoading(true);
    try {
      const apiUrl = getApiUrl(API_CONFIG.ENDPOINTS.LIBRARIES);
      console.log("Fetching libraries from:", API_CONFIG.ENDPOINTS.LIBRARIES);
      
      const res = await fetch(apiUrl);
      
      console.log("Fetch response status:", res.status);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("Libraries data:", data);
      
      // Handle both array and object responses
      const librariesArray = Array.isArray(data) ? data : (data.libraries || []);
      setDatasets(librariesArray);
      setLoading(false);
    } catch (err) {
      console.error("Error fetching libraries:", err);
      setStatus(`❌ Failed to load datasets: ${err.message}`);
      setDatasets([]); // Set empty array on error
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLibraries();
  }, []);

  // Upload CSV file to FastAPI backend
  const handleUpload = async () => {
    if (!file) {
      setStatus("⚠️ Please select a CSV file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStatus("⏳ Uploading...");
      const apiUrl = getApiUrl(API_CONFIG.ENDPOINTS.UPLOAD);
      console.log("Uploading to:", API_CONFIG.ENDPOINTS.UPLOAD);
      
      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      console.log("Upload response status:", res.status);

      if (!res.ok) {
        let errorMessage = `Upload failed with status ${res.status}`;
        try {
          const errorData = await res.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (parseError) {
          // If response isn't JSON, try to get text
          try {
            const errorText = await res.text();
            if (errorText) errorMessage = errorText;
          } catch (textError) {
            // Keep default error message
          }
        }
        throw new Error(errorMessage);
      }

      // Try to parse JSON response, but don't fail if it's not JSON
      let data;
      try {
        data = await res.json();
        console.log("Upload response data:", data);
      } catch (parseError) {
        console.log("Response is not JSON, treating as successful");
        data = { message: "File uploaded successfully" };
      }

      setStatus("✅ File uploaded successfully!");
      console.log("Upload successful");

      // Reset input and clear file input element
      setFile(null);
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = '';

      // Refresh library after a brief delay
      setTimeout(() => {
        fetchLibraries();
      }, 500);
    } catch (err) {
      console.error("Upload error:", err);
      setStatus("❌ Upload failed: " + err.message);
    }
  };

  const handleSelectDataset = (dataset) => {
    // This function is no longer used since DatasetCard handles navigation directly
    console.log("Dataset selected:", dataset.name);
  };

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      padding: "0"
    }}>
      {/* Header */}
      <div style={{
        textAlign: "center",
        padding: "60px 20px 40px 20px",
        background: "rgba(255, 255, 255, 0.1)",
        backdropFilter: "blur(10px)",
        borderBottom: "1px solid rgba(255, 255, 255, 0.2)",
        marginBottom: "50px"
      }}>
        <h1 
          onClick={() => navigate("/")}
          style={{
            fontSize: "48px",
            fontWeight: "800",
            color: "white",
            margin: "0 0 16px 0",
            textShadow: "0 4px 20px rgba(0,0,0,0.3)",
            background: "linear-gradient(135deg, #ffffff, #f0f4ff)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            cursor: "pointer",
            transition: "all 0.3s ease",
            display: "inline-block"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "scale(1.05)";
            e.currentTarget.style.textShadow = "0 6px 25px rgba(0,0,0,0.4)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "scale(1)";
            e.currentTarget.style.textShadow = "0 4px 20px rgba(0,0,0,0.3)";
          }}
        >
          🚀 Predictra
        </h1>
        <p style={{
          fontSize: "20px",
          color: "rgba(255, 255, 255, 0.9)",
          margin: 0,
          fontWeight: "500",
          textShadow: "0 2px 10px rgba(0,0,0,0.2)"
        }}>
          AI-Powered Data Analysis Platform
        </p>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "40px 20px" }}>
        {/* Library Section */}
        <div style={{
          background: "rgba(255, 255, 255, 0.95)",
          borderRadius: "20px",
          padding: "40px",
          boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
          marginBottom: "40px"
        }}>
          <div style={{ 
            display: "flex", 
            justifyContent: "space-between", 
            alignItems: "center",
            marginBottom: "30px"
          }}>
            <h2 style={{ 
              fontSize: "24px", 
              fontWeight: "600", 
              color: "#2d3748",
              margin: 0
            }}>
              📊 Dataset Library
            </h2>
            <div style={{
              background: "#f7fafc",
              padding: "8px 16px",
              borderRadius: "20px",
              fontSize: "14px",
              color: "#4a5568",
              fontWeight: "500"
            }}>
              {datasets.length} dataset{datasets.length !== 1 ? 's' : ''}
            </div>
          </div>

          {/* Library Grid */}
          {loading ? (
            <div style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              padding: "60px 0",
              flexDirection: "column"
            }}>
              <div style={{
                width: "40px",
                height: "40px",
                border: "4px solid #e2e8f0",
                borderTop: "4px solid #667eea",
                borderRadius: "50%",
                animation: "spin 1s linear infinite"
              }}></div>
              <p style={{ 
                marginTop: "16px", 
                color: "#718096",
                fontSize: "16px"
              }}>
                Loading datasets...
              </p>
            </div>
          ) : datasets.length === 0 ? (
            <div style={{
              textAlign: "center",
              padding: "60px 0",
              color: "#718096"
            }}>
              <div style={{ fontSize: "48px", marginBottom: "16px" }}>📁</div>
              <h3 style={{ fontSize: "18px", margin: "0 0 8px 0", color: "#4a5568" }}>
                No datasets found
              </h3>
              <p style={{ margin: 0, fontSize: "14px" }}>
                Upload your first dataset to get started
              </p>
            </div>
          ) : (
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: "24px"
            }}>
              {datasets.map((d) => (
                <DatasetCard key={d.name} dataset={d} onSelect={handleSelectDataset} />
              ))}
            </div>
          )}
        </div>

        {/* Upload Section */}
        <div style={{
          background: "rgba(255, 255, 255, 0.95)",
          borderRadius: "20px",
          padding: "40px",
          boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
          textAlign: "center"
        }}>
          <div style={{ marginBottom: "30px" }}>
            <h2 style={{ 
              fontSize: "24px", 
              fontWeight: "600", 
              color: "#2d3748",
              margin: "0 0 8px 0"
            }}>
              🚀 Upload New Dataset
            </h2>
            <p style={{ 
              color: "#718096", 
              margin: 0,
              fontSize: "16px"
            }}>
              Add your CSV files to unlock AI-powered insights
            </p>
          </div>

          <div style={{
            border: "2px dashed #cbd5e0",
            borderRadius: "16px",
            padding: "40px 20px",
            background: "#f7fafc",
            transition: "all 0.3s ease",
            cursor: "pointer",
            position: "relative"
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.currentTarget.style.borderColor = "#667eea";
            e.currentTarget.style.background = "#edf2f7";
          }}
          onDragLeave={(e) => {
            e.currentTarget.style.borderColor = "#cbd5e0";
            e.currentTarget.style.background = "#f7fafc";
          }}
          onDrop={(e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === "text/csv") {
              setFile(files[0]);
            }
            e.currentTarget.style.borderColor = "#cbd5e0";
            e.currentTarget.style.background = "#f7fafc";
          }}
          onClick={() => document.querySelector('input[type="file"]').click()}
          >
            <div style={{ fontSize: "48px", marginBottom: "16px" }}>📤</div>
            <p style={{ 
              margin: "0 0 16px 0", 
              color: "#4a5568",
              fontSize: "16px",
              fontWeight: "500"
            }}>
              {file ? file.name : "Click to select or drag & drop CSV file"}
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
              style={{ 
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                opacity: 0,
                cursor: "pointer"
              }}
            />
          </div>

          <button
            onClick={handleUpload}
            disabled={!file}
            style={{
              marginTop: "24px",
              background: file ? "linear-gradient(135deg, #667eea, #764ba2)" : "#e2e8f0",
              color: file ? "white" : "#a0aec0",
              border: "none",
              borderRadius: "12px",
              padding: "12px 32px",
              fontSize: "16px",
              fontWeight: "600",
              cursor: file ? "pointer" : "not-allowed",
              transition: "all 0.3s ease",
              boxShadow: file ? "0 4px 15px rgba(102, 126, 234, 0.4)" : "none"
            }}
          >
            {file ? "🚀 Upload Dataset" : "Select a file first"}
          </button>

          {status && (
            <div style={{
              marginTop: "20px",
              padding: "12px 20px",
              borderRadius: "8px",
              fontSize: "14px",
              fontWeight: "500",
              background: status.includes("✅") ? "#f0fff4" : status.includes("❌") ? "#fed7d7" : "#fef5e7",
              color: status.includes("✅") ? "#22543d" : status.includes("❌") ? "#c53030" : "#c05621",
              border: `1px solid ${status.includes("✅") ? "#9ae6b4" : status.includes("❌") ? "#feb2b2" : "#fbd38d"}`
            }}>
              {status}
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}