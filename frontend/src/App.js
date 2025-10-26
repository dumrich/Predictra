import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LibraryPage from "./components/LibraryPage";
import AnalysisPage from "./components/AnalysisPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LibraryPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
      </Routes>
    </Router>
  );
}

export default App;
