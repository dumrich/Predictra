import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext";
import DatasetCard from "./DatasetCard";
import { getApiUrl, API_CONFIG } from "../config";

export default function LibraryPage() {
  const navigate = useNavigate();
  const { theme } = useTheme();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [uploadLoading, setUploadLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

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

    setUploadLoading(true);
    setStatus("📤 Uploading your dataset...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const apiUrl = getApiUrl(API_CONFIG.ENDPOINTS.UPLOAD);
      console.log("Uploading to:", apiUrl);

      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      console.log("Upload success:", data);

      setStatus(`✅ ${data.message || 'Dataset uploaded successfully!'}`);
      setFile(null); // Clear file input

      // Refresh the libraries list to show the new upload
      setTimeout(fetchLibraries, 1000);

    } catch (err) {
      console.error("Upload error:", err);
      setStatus(`❌ Upload failed: ${err.message}`);
    } finally {
      setUploadLoading(false);
    }
  };

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type === 'text/csv' || selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
        setStatus(`📄 Selected: ${selectedFile.name}`);
      } else {
        setStatus("❌ Please select a CSV file only!");
        setFile(null);
        e.target.value = '';
      }
    }
  };

  // Filter datasets based on search term
  const filteredDatasets = datasets.filter(dataset =>
    dataset.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    dataset.csv_file?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Handle dataset card click
  const handleCardClick = (dataset) => {
    navigate("/analysis", { state: { dataset } });
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: theme.background,
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
    }}>
      {/* Hero Header Section */}
      <div style={{
        padding: "80px 40px 60px 40px",
        textAlign: "center",
        position: "relative",
        overflow: "hidden"
      }}>
        {/* Background Decorations */}
        <div style={{
          position: "absolute",
          top: "-50px",
          left: "-50px",
          width: "200px",
          height: "200px",
          background: `linear-gradient(135deg, ${theme.primary}40, ${theme.primaryDark}20)`,
          borderRadius: "50%",
          filter: "blur(40px)",
          animation: "float 6s ease-in-out infinite"
        }} />
        <div style={{
          position: "absolute",
          bottom: "-50px",
          right: "-50px",
          width: "300px",
          height: "300px",
          background: `linear-gradient(135deg, ${theme.accent}30, ${theme.primary}20)`,
          borderRadius: "50%",
          filter: "blur(60px)",
          animation: "float 8s ease-in-out infinite reverse"
        }} />

        <div style={{
          position: "relative",
          zIndex: 2,
          maxWidth: "800px",
          margin: "0 auto"
        }}>
          <div style={{
            fontSize: "72px",
            marginBottom: "20px",
            color: "white",
            fontWeight: "900",
            letterSpacing: "-2px",
            textShadow: "0 4px 20px rgba(0,0,0,0.5), 0 0 40px rgba(255,255,255,0.3)",
            filter: "drop-shadow(0 2px 8px rgba(0,0,0,0.3))"
          }}>
            🚀 Predictra
          </div>

          <h1 style={{
            fontSize: "48px",
            fontWeight: "800",
            color: "white",
            margin: "0 0 24px 0",
            textShadow: "0 4px 20px rgba(0,0,0,0.3)",
            letterSpacing: "-1px"
          }}>
            Intelligent Data Analysis Platform
          </h1>

          <p style={{
            fontSize: "20px",
            color: "rgba(255,255,255,0.9)",
            margin: "0 0 40px 0",
            lineHeight: "1.6",
            fontWeight: "400",
            textShadow: "0 2px 10px rgba(0,0,0,0.2)"
          }}>
            Upload your datasets and harness the power of AI for predictive analytics,
            statistical insights, and interactive data exploration.
          </p>

          {/* Key Features */}
          <div style={{
            display: "flex",
            justifyContent: "center",
            flexWrap: "wrap",
            gap: "20px",
            marginTop: "40px",
            maxWidth: "800px",
            margin: "40px auto 0 auto"
          }}>
            {[
              { icon: "🤖", title: "AI-Powered", desc: "Machine Learning" },
              { icon: "📊", title: "Real-time", desc: "Visualizations" },
              { icon: "🔮", title: "Predictive", desc: "Analytics" },
              { icon: "💬", title: "Interactive", desc: "Chat Assistant" }
            ].map((feature, idx) => (
              <div key={idx} style={{
                background: "rgba(255,255,255,0.1)",
                borderRadius: "16px",
                padding: "20px",
                textAlign: "center",
                backdropFilter: "blur(10px)",
                border: "1px solid rgba(255,255,255,0.2)",
                transition: "all 0.3s ease",
                minWidth: "180px",
                flex: "0 0 auto"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-5px)";
                e.currentTarget.style.background = "rgba(255,255,255,0.15)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.background = "rgba(255,255,255,0.1)";
              }}>
                <div style={{ fontSize: "32px", marginBottom: "8px" }}>{feature.icon}</div>
                <div style={{
                  fontSize: "16px",
                  fontWeight: "600",
                  color: "white",
                  marginBottom: "4px"
                }}>{feature.title}</div>
                <div style={{
                  fontSize: "12px",
                  color: "rgba(255,255,255,0.8)"
                }}>{feature.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content Section */}
      <div style={{
        background: theme.surfaceLight,
        minHeight: "60vh",
        padding: "60px 40px",
        borderRadius: "40px 40px 0 0",
        position: "relative",
        zIndex: 3,
        boxShadow: "0 -10px 40px rgba(0,0,0,0.1)"
      }}>
        <div style={{
          maxWidth: "1200px",
          margin: "0 auto"
        }}>
          {/* Upload Section */}
          <div style={{
            background: theme.surface,
            borderRadius: "24px",
            padding: "40px",
            marginBottom: "50px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.08)",
            border: `1px solid ${theme.borderLight}`
          }}>
            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr auto",
              gap: "40px",
              alignItems: "center"
            }}>
              <div>
                <h2 style={{
                  fontSize: "28px",
                  fontWeight: "700",
                  color: theme.text,
                  margin: "0 0 12px 0"
                }}>
                  📤 Upload New Dataset
                </h2>
                <p style={{
                  fontSize: "16px",
                  color: theme.textLight,
                  margin: "0 0 24px 0",
                  lineHeight: "1.5"
                }}>
                  Upload your CSV files to begin advanced data analysis and predictive modeling
                </p>

                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "16px",
                  flexWrap: "wrap"
                }}>
                  <label style={{
                    display: "inline-block",
                    padding: "12px 24px",
                    background: theme.surfaceDark,
                    border: `2px dashed ${theme.border}`,
                    borderRadius: "12px",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                    fontSize: "14px",
                    fontWeight: "500",
                    color: theme.text
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = theme.primary;
                    e.currentTarget.style.background = `${theme.primary}10`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = theme.border;
                    e.currentTarget.style.background = theme.surfaceDark;
                  }}>
                    📁 Choose CSV File
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      style={{ display: "none" }}
                    />
                  </label>

                  <button
                    onClick={handleUpload}
                    disabled={!file || uploadLoading}
                    style={{
                      padding: "12px 32px",
                      background: (!file || uploadLoading) ? theme.border : theme.background,
                      color: (!file || uploadLoading) ? theme.textMuted : "white",
                      border: "none",
                      borderRadius: "12px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: (!file || uploadLoading) ? "not-allowed" : "pointer",
                      transition: "all 0.3s ease",
                      textTransform: "uppercase",
                      letterSpacing: "0.5px"
                    }}
                    onMouseEnter={(e) => {
                      if (file && !uploadLoading) {
                        e.currentTarget.style.transform = "translateY(-2px)";
                        e.currentTarget.style.boxShadow = "0 8px 25px rgba(0,0,0,0.2)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (file && !uploadLoading) {
                        e.currentTarget.style.transform = "translateY(0)";
                        e.currentTarget.style.boxShadow = "none";
                      }
                    }}
                  >
                    {uploadLoading ? "⏳ Uploading..." : "🚀 Upload"}
                  </button>
                </div>

                {status && (
                  <div style={{
                    marginTop: "16px",
                    padding: "12px 16px",
                    borderRadius: "8px",
                    background: status.includes("❌") ? `${theme.error}15` :
                                status.includes("✅") ? `${theme.success}15` :
                                `${theme.info}15`,
                    border: `1px solid ${status.includes("❌") ? theme.error + "30" :
                                        status.includes("✅") ? theme.success + "30" :
                                        theme.info + "30"}`,
                    color: status.includes("❌") ? theme.error :
                          status.includes("✅") ? theme.success : theme.info,
                    fontSize: "14px",
                    fontWeight: "500"
                  }}>
                    {status}
                  </div>
                )}
              </div>

              {/* Upload Illustration */}
              <div style={{
                width: "120px",
                height: "120px",
                background: `${theme.primary}15`,
                borderRadius: "20px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "48px",
                animation: "pulse 3s infinite"
              }}>
                📊
              </div>
            </div>
          </div>

          {/* Datasets Section Header */}
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "30px",
            flexWrap: "wrap",
            gap: "20px"
          }}>
            <div>
              <h2 style={{
                fontSize: "32px",
                fontWeight: "700",
                color: theme.text,
                margin: "0 0 8px 0"
              }}>
                📚 Your Datasets
              </h2>
              <p style={{
                fontSize: "16px",
                color: theme.textLight,
                margin: 0
              }}>
                {datasets.length} dataset{datasets.length !== 1 ? 's' : ''} available for analysis
              </p>
            </div>

            {/* Search Bar */}
            <div style={{ position: "relative" }}>
              <input
                type="text"
                placeholder="🔍 Search datasets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  padding: "12px 16px 12px 40px",
                  borderRadius: "12px",
                  border: `2px solid ${theme.border}`,
                  fontSize: "14px",
                  fontWeight: "500",
                  color: theme.text,
                  background: theme.surface,
                  minWidth: "250px",
                  transition: "all 0.3s ease",
                  outline: "none"
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = theme.primary;
                  e.target.style.boxShadow = `0 0 0 3px ${theme.primary}20`;
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = theme.border;
                  e.target.style.boxShadow = "none";
                }}
              />
              <div style={{
                position: "absolute",
                left: "12px",
                top: "50%",
                transform: "translateY(-50%)",
                fontSize: "16px",
                color: theme.textMuted
              }}>
                🔍
              </div>
            </div>
          </div>

          {/* Datasets Grid */}
          {loading ? (
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
              gap: "24px"
            }}>
              {[...Array(6)].map((_, i) => (
                <div key={i} style={{
                  background: theme.surface,
                  borderRadius: "20px",
                  padding: "24px",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
                  border: `1px solid ${theme.borderLight}`,
                  animation: "pulse 2s infinite"
                }}>
                  <div style={{
                    width: "100%",
                    height: "120px",
                    background: theme.surfaceDark,
                    borderRadius: "12px",
                    marginBottom: "16px"
                  }} />
                  <div style={{
                    width: "80%",
                    height: "20px",
                    background: theme.surfaceDark,
                    borderRadius: "4px",
                    marginBottom: "8px"
                  }} />
                  <div style={{
                    width: "60%",
                    height: "16px",
                    background: theme.surfaceDark,
                    borderRadius: "4px"
                  }} />
                </div>
              ))}
            </div>
          ) : filteredDatasets.length === 0 ? (
            <div style={{
              textAlign: "center",
              padding: "80px 20px",
              color: theme.textLight
            }}>
              <div style={{
                fontSize: "80px",
                marginBottom: "24px",
                opacity: 0.5
              }}>
                {searchTerm ? "🔍" : "📊"}
              </div>
              <h3 style={{
                fontSize: "24px",
                fontWeight: "600",
                margin: "0 0 12px 0",
                color: theme.text
              }}>
                {searchTerm ? "No datasets found" : "No datasets yet"}
              </h3>
              <p style={{
                fontSize: "16px",
                margin: "0 0 32px 0",
                lineHeight: "1.5"
              }}>
                {searchTerm
                  ? `No datasets match "${searchTerm}". Try a different search term.`
                  : "Upload your first CSV file to get started with AI-powered data analysis"
                }
              </p>
              {searchTerm && (
                <button
                  onClick={() => setSearchTerm("")}
                  style={{
                    padding: "12px 24px",
                    background: theme.background,
                    color: "white",
                    border: "none",
                    borderRadius: "12px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer"
                  }}
                >
                  Clear Search
                </button>
              )}
            </div>
          ) : (
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
              gap: "24px"
            }}>
              {filteredDatasets.map((dataset, index) => (
                <div
                  key={index}
                  onClick={() => handleCardClick(dataset)}
                  style={{
                    background: theme.surface,
                    borderRadius: "20px",
                    padding: "24px",
                    cursor: "pointer",
                    transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                    boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
                    border: `1px solid ${theme.borderLight}`,
                    position: "relative",
                    overflow: "hidden"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-8px)";
                    e.currentTarget.style.boxShadow = "0 20px 40px rgba(0,0,0,0.15)";
                    e.currentTarget.style.borderColor = theme.primary;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "0 4px 20px rgba(0,0,0,0.08)";
                    e.currentTarget.style.borderColor = theme.borderLight;
                  }}
                >
                  <DatasetCard dataset={dataset} theme={theme} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </div>
  );
}