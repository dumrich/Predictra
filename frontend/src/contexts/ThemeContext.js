import React, { createContext, useContext, useState, useEffect } from 'react';

// Define theme configurations
const themes = {
  default: {
    name: 'Default Purple',
    primary: '#667eea',
    primaryDark: '#764ba2',
    secondary: '#75192c',
    accent: '#22c55e',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    surfaceLight: 'rgba(255, 255, 255, 0.95)',
    surface: 'white',
    surfaceDark: '#f8fafc',
    text: '#2d3748',
    textLight: '#718096',
    textMuted: '#64748b',
    border: '#e2e8f0',
    borderLight: 'rgba(102, 126, 234, 0.2)',
    success: '#22c55e',
    error: '#ef4444',
    warning: '#f59e0b',
    info: '#3b82f6'
  },
  ocean: {
    name: 'Ocean Blue',
    primary: '#0ea5e9',
    primaryDark: '#0284c7',
    secondary: '#06b6d4',
    accent: '#10b981',
    background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
    surfaceLight: 'rgba(255, 255, 255, 0.95)',
    surface: 'white',
    surfaceDark: '#f0f9ff',
    text: '#1e293b',
    textLight: '#64748b',
    textMuted: '#94a3b8',
    border: '#e2e8f0',
    borderLight: 'rgba(14, 165, 233, 0.2)',
    success: '#10b981',
    error: '#ef4444',
    warning: '#f59e0b',
    info: '#0ea5e9'
  },
  sunset: {
    name: 'Sunset Orange',
    primary: '#f97316',
    primaryDark: '#ea580c',
    secondary: '#ef4444',
    accent: '#eab308',
    background: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
    surfaceLight: 'rgba(255, 255, 255, 0.95)',
    surface: 'white',
    surfaceDark: '#fffbeb',
    text: '#1c1917',
    textLight: '#78716c',
    textMuted: '#a8a29e',
    border: '#e7e5e4',
    borderLight: 'rgba(249, 115, 22, 0.2)',
    success: '#22c55e',
    error: '#dc2626',
    warning: '#f59e0b',
    info: '#3b82f6'
  },
  forest: {
    name: 'Forest Green',
    primary: '#059669',
    primaryDark: '#047857',
    secondary: '#0d9488',
    accent: '#84cc16',
    background: 'linear-gradient(135deg, #059669 0%, #047857 100%)',
    surfaceLight: 'rgba(255, 255, 255, 0.95)',
    surface: 'white',
    surfaceDark: '#f0fdf4',
    text: '#14532d',
    textLight: '#6b7280',
    textMuted: '#9ca3af',
    border: '#d1d5db',
    borderLight: 'rgba(5, 150, 105, 0.2)',
    success: '#059669',
    error: '#dc2626',
    warning: '#f59e0b',
    info: '#3b82f6'
  },
  midnight: {
    name: 'Midnight Dark',
    primary: '#8b5cf6',
    primaryDark: '#7c3aed',
    secondary: '#a855f7',
    accent: '#06b6d4',
    background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)',
    surfaceLight: 'rgba(30, 41, 59, 0.95)',
    surface: '#1e293b',
    surfaceDark: '#0f172a',
    text: '#f1f5f9',
    textLight: '#cbd5e1',
    textMuted: '#94a3b8',
    border: '#334155',
    borderLight: 'rgba(139, 92, 246, 0.2)',
    success: '#10b981',
    error: '#f87171',
    warning: '#fbbf24',
    info: '#60a5fa'
  },
  rose: {
    name: 'Rose Pink',
    primary: '#e11d48',
    primaryDark: '#be123c',
    secondary: '#f43f5e',
    accent: '#06b6d4',
    background: 'linear-gradient(135deg, #e11d48 0%, #be123c 100%)',
    surfaceLight: 'rgba(255, 255, 255, 0.95)',
    surface: 'white',
    surfaceDark: '#fdf2f8',
    text: '#1f2937',
    textLight: '#6b7280',
    textMuted: '#9ca3af',
    border: '#e5e7eb',
    borderLight: 'rgba(225, 29, 72, 0.2)',
    success: '#10b981',
    error: '#dc2626',
    warning: '#f59e0b',
    info: '#3b82f6'
  }
};

// Create theme context
const ThemeContext = createContext();

// Theme provider component
export const ThemeProvider = ({ children }) => {
  const [currentTheme, setCurrentTheme] = useState('default');

  // Load saved theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('predictra-theme');
    if (savedTheme && themes[savedTheme]) {
      setCurrentTheme(savedTheme);
    }
  }, []);

  // Save theme to localStorage when changed
  useEffect(() => {
    localStorage.setItem('predictra-theme', currentTheme);
  }, [currentTheme]);

  const changeTheme = (themeKey) => {
    if (themes[themeKey]) {
      setCurrentTheme(themeKey);
    }
  };

  const value = {
    theme: themes[currentTheme],
    currentTheme,
    changeTheme,
    availableThemes: Object.keys(themes).map(key => ({
      key,
      name: themes[key].name,
      primary: themes[key].primary,
      primaryDark: themes[key].primaryDark
    }))
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook to use theme
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export default ThemeContext;