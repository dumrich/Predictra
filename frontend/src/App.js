import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./contexts/ThemeContext";
import LibraryPage from "./components/LibraryPage";
import AnalysisPage from "./components/AnalysisPage";
import ThemeSelector from "./components/ThemeSelector";

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/" element={<LibraryPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
        </Routes>
        {/* Floating theme selector - always available */}
        <ThemeSelector isCompact={true} />
      </Router>
    </ThemeProvider>
  );
}

export default App;
