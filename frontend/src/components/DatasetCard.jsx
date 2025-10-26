import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { getApiUrl } from "../config";

export default function DatasetCard({ dataset, theme, onSelect }) {
  const navigate = useNavigate();
  const [imageError, setImageError] = useState(false);

  const handleAnalyze = (e) => {
    e.stopPropagation();
    // Navigate to analysis page with dataset data
    navigate("/analysis", { state: { dataset } });
  };

  return (
    <>
      {/* Thumbnail/Preview */}
      <div style={{
        width: "100%",
        height: "140px",
        background: `${theme.primary}10`,
        borderRadius: "12px",
        marginBottom: "16px",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      }}>
        {dataset.thumbnail && !imageError ? (
          <img
            src={getApiUrl(dataset.thumbnail)}
            alt={dataset.name}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover"
            }}
            onError={() => setImageError(true)}
          />
        ) : (
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "8px"
          }}>
            <div style={{
              fontSize: "48px",
              opacity: 0.6
            }}>
              📊
            </div>
            <div style={{
              fontSize: "12px",
              color: theme.textMuted,
              fontWeight: "500"
            }}>
              Data Preview
            </div>
          </div>
        )}

        {/* File Format Badge */}
        <div style={{
          position: "absolute",
          top: "8px",
          right: "8px",
          padding: "4px 8px",
          background: theme.background,
          color: "white",
          borderRadius: "6px",
          fontSize: "10px",
          fontWeight: "600",
          textTransform: "uppercase",
          letterSpacing: "0.5px"
        }}>
          CSV
        </div>
      </div>

      {/* Content */}
      <div>
        <h3 style={{
          fontSize: "18px",
          fontWeight: "700",
          color: theme.text,
          margin: "0 0 8px 0",
          lineHeight: "1.3"
        }}>
          {dataset.name || "Unknown Dataset"}
        </h3>

        <p style={{
          fontSize: "13px",
          color: theme.textMuted,
          margin: "0 0 16px 0",
          lineHeight: "1.4"
        }}>
          {dataset.csv_file || "No filename available"}
        </p>

        {/* Stats */}
        <div style={{
          display: "flex",
          gap: "12px",
          marginBottom: "16px"
        }}>
          <div style={{
            background: theme.surfaceDark,
            borderRadius: "8px",
            padding: "6px 10px",
            fontSize: "11px",
            fontWeight: "600",
            color: theme.textMuted,
            textAlign: "center",
            flex: 1
          }}>
            <div style={{ color: theme.primary, fontWeight: "700", fontSize: "14px" }}>
              📈
            </div>
            <div>Ready</div>
          </div>
          <div style={{
            background: theme.surfaceDark,
            borderRadius: "8px",
            padding: "6px 10px",
            fontSize: "11px",
            fontWeight: "600",
            color: theme.textMuted,
            textAlign: "center",
            flex: 1
          }}>
            <div style={{ color: theme.accent, fontWeight: "700", fontSize: "14px" }}>
              🎯
            </div>
            <div>Analysis</div>
          </div>
        </div>

        {/* Action Button */}
        <button
          onClick={handleAnalyze}
          style={{
            width: "100%",
            padding: "12px 16px",
            background: theme.background,
            color: "white",
            border: "none",
            borderRadius: "10px",
            fontSize: "13px",
            fontWeight: "600",
            cursor: "pointer",
            transition: "all 0.2s ease",
            textTransform: "uppercase",
            letterSpacing: "0.5px"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
            e.currentTarget.style.boxShadow = "0 6px 20px rgba(0,0,0,0.15)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "none";
          }}
        >
          🚀 Analyze Data
        </button>
      </div>
    </>
  );
}