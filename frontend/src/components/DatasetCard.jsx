import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { getApiUrl } from "../config";

export default function DatasetCard({ dataset, onSelect }) {
  const navigate = useNavigate();
  const [imageError, setImageError] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const handleAnalyze = (e) => {
    e.stopPropagation();
    // Navigate to analysis page with dataset data
    navigate("/analysis", { state: { dataset } });
  };

  return (
    <div
      style={{
        background: "white",
        borderRadius: "16px",
        padding: "0",
        overflow: "hidden",
        boxShadow: "0 8px 30px rgba(0,0,0,0.12)",
        transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
        cursor: "pointer",
        border: "1px solid rgba(255,255,255,0.2)",
        position: "relative",
        transform: isHovered ? "translateY(-8px) scale(1.02)" : "translateY(0) scale(1)"
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => onSelect(dataset)}
    >
      {/* Image Section */}
      <div style={{
        height: "200px",
        background: "linear-gradient(135deg, #667eea, #764ba2, #f093fb)",
        position: "relative",
        overflow: "hidden"
      }}>
        {!imageError && dataset.thumbnail ? (
          <img
            src={getApiUrl(dataset.thumbnail)}
            alt={dataset.name}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
            onError={() => setImageError(true)}
          />
        ) : (
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            fontSize: "64px",
            color: "rgba(255,255,255,0.9)",
            textShadow: "0 4px 20px rgba(0,0,0,0.3)"
          }}>
            📊
          </div>
        )}
        
        {/* Overlay */}
        <div style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "linear-gradient(135deg, rgba(102, 126, 234, 0.8), rgba(118, 75, 162, 0.8))",
          opacity: 0,
          transition: "opacity 0.3s ease",
          display: "flex",
          alignItems: "center",
          justifyContent: "center"
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.opacity = "1";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.opacity = "0";
        }}
        >
          <div style={{
            background: "rgba(255,255,255,0.9)",
            color: "#4a5568",
            padding: "8px 16px",
            borderRadius: "20px",
            fontSize: "14px",
            fontWeight: "600"
          }}>
            Click to use
          </div>
        </div>
      </div>

      {/* Content Section */}
      <div style={{ padding: "20px" }}>
        <h3 style={{ 
          margin: "0 0 8px 0", 
          fontSize: "18px",
          fontWeight: "600",
          color: "#2d3748",
          lineHeight: "1.4"
        }}>
          {dataset.name}
        </h3>
        
        {dataset.description && (
          <p style={{
            margin: "0 0 16px 0",
            fontSize: "14px",
            color: "#718096",
            lineHeight: "1.5",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden"
          }}>
            {dataset.description}
          </p>
        )}

        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: "16px"
        }}>
          <div style={{
            background: "#f7fafc",
            color: "#4a5568",
            padding: "4px 12px",
            borderRadius: "12px",
            fontSize: "12px",
            fontWeight: "500"
          }}>
            CSV Dataset
          </div>
          
          <button
            onClick={handleAnalyze}
            style={{
              background: "linear-gradient(135deg, #667eea, #764ba2)",
              color: "white",
              border: "none",
              borderRadius: "8px",
              padding: "8px 16px",
              fontSize: "14px",
              fontWeight: "600",
              cursor: "pointer",
              transition: "all 0.2s ease"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "scale(1.05)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }}
          >
            🚀 Analyze
          </button>
        </div>
      </div>
    </div>
  );
}
