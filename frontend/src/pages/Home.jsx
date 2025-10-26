import React from "react";
import DatasetLibrary from "../components/DatasetLibrary";
import FileUpload from "../components/FileUpload";

export default function Home() {
  const handleSelectDataset = async (dataset) => {
    console.log("User selected dataset:", dataset);

    // Example: send request to backend to get predictions for selected dataset
    try {
      const res = await fetch(
        `http://localhost:5000/api/predict?file=${dataset.csv_file}`
      );
      if (!res.ok) throw new Error("Prediction request failed");

      const predictionData = await res.json();
      console.log("Prediction result:", predictionData);
      alert(`Predictions received for ${dataset.name}! Check console.`);
    } catch (err) {
      console.error(err);
      alert("Failed to fetch predictions");
    }
  };

  return (
    <div style={{ padding: "40px", fontFamily: "sans-serif" }}>
      <h1 style={{ textAlign: "center", marginBottom: "30px" }}>
        📊 Predictify — Smart Data Prediction Tool
      </h1>

      <DatasetLibrary onSelect={handleSelectDataset} />
      <FileUpload />
    </div>
  );
}
