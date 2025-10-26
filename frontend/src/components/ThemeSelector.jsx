import React, { useState } from 'react';
import { useTheme } from '../contexts/ThemeContext';

const ThemeSelector = ({ isCompact = false }) => {
  const { theme, currentTheme, changeTheme, availableThemes } = useTheme();
  const [isOpen, setIsOpen] = useState(false);

  if (isCompact) {
    // Compact floating theme selector
    return (
      <div style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 1000
      }}>
        <div style={{ position: 'relative' }}>
          {/* Theme Toggle Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            style={{
              width: '50px',
              height: '50px',
              borderRadius: '50%',
              border: 'none',
              background: theme.background,
              boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px',
              transition: 'all 0.3s ease',
              backdropFilter: 'blur(10px)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.1)';
              e.currentTarget.style.boxShadow = '0 6px 25px rgba(0,0,0,0.2)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.15)';
            }}
          >
            🎨
          </button>

          {/* Theme Dropdown */}
          {isOpen && (
            <div style={{
              position: 'absolute',
              top: '60px',
              right: '0',
              background: theme.surface,
              borderRadius: '16px',
              boxShadow: '0 10px 40px rgba(0,0,0,0.15)',
              border: `1px solid ${theme.border}`,
              padding: '12px',
              minWidth: '200px',
              backdropFilter: 'blur(10px)'
            }}>
              <h4 style={{
                margin: '0 0 12px 0',
                fontSize: '14px',
                fontWeight: '600',
                color: theme.text,
                padding: '0 8px'
              }}>
                Choose Theme
              </h4>
              <div style={{
                display: 'grid',
                gap: '8px'
              }}>
                {availableThemes.map((themeOption) => (
                  <button
                    key={themeOption.key}
                    onClick={() => {
                      changeTheme(themeOption.key);
                      setIsOpen(false);
                    }}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '10px 12px',
                      borderRadius: '10px',
                      border: 'none',
                      background: currentTheme === themeOption.key
                        ? `linear-gradient(135deg, ${themeOption.primary}, ${themeOption.primaryDark})`
                        : theme.surfaceDark,
                      color: currentTheme === themeOption.key ? 'white' : theme.text,
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: '500',
                      transition: 'all 0.2s ease',
                      textAlign: 'left'
                    }}
                    onMouseEnter={(e) => {
                      if (currentTheme !== themeOption.key) {
                        e.currentTarget.style.background = theme.borderLight;
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (currentTheme !== themeOption.key) {
                        e.currentTarget.style.background = theme.surfaceDark;
                      }
                    }}
                  >
                    <div style={{
                      width: '20px',
                      height: '20px',
                      borderRadius: '50%',
                      background: `linear-gradient(135deg, ${themeOption.primary}, ${themeOption.primaryDark})`,
                      flexShrink: 0
                    }} />
                    <span>{themeOption.name}</span>
                    {currentTheme === themeOption.key && (
                      <span style={{ marginLeft: 'auto', fontSize: '12px' }}>✓</span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Backdrop */}
        {isOpen && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: -1
            }}
            onClick={() => setIsOpen(false)}
          />
        )}
      </div>
    );
  }

  // Full theme selector component
  return (
    <div style={{
      background: theme.surface,
      borderRadius: '20px',
      padding: '30px',
      boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
      border: `1px solid ${theme.borderLight}`
    }}>
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{
          margin: '0 0 8px 0',
          fontSize: '20px',
          fontWeight: '700',
          color: theme.text
        }}>
          🎨 Theme Settings
        </h3>
        <p style={{
          margin: 0,
          fontSize: '14px',
          color: theme.textLight
        }}>
          Customize your visual experience
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px'
      }}>
        {availableThemes.map((themeOption) => (
          <button
            key={themeOption.key}
            onClick={() => changeTheme(themeOption.key)}
            style={{
              padding: '20px',
              borderRadius: '16px',
              border: currentTheme === themeOption.key
                ? `3px solid ${themeOption.primary}`
                : `2px solid ${theme.border}`,
              background: theme.surface,
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              textAlign: 'center',
              position: 'relative',
              overflow: 'hidden'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-4px)';
              e.currentTarget.style.boxShadow = '0 12px 35px rgba(0,0,0,0.15)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            {/* Theme Preview */}
            <div style={{
              width: '100%',
              height: '60px',
              borderRadius: '8px',
              background: `linear-gradient(135deg, ${themeOption.primary}, ${themeOption.primaryDark})`,
              marginBottom: '12px',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                color: 'white',
                fontWeight: '600',
                fontSize: '12px',
                textShadow: '0 1px 3px rgba(0,0,0,0.3)'
              }}>
                Preview
              </div>
            </div>

            <h4 style={{
              margin: '0 0 4px 0',
              fontSize: '16px',
              fontWeight: '600',
              color: theme.text
            }}>
              {themeOption.name}
            </h4>

            <div style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '6px',
              marginTop: '8px'
            }}>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: themeOption.primary
              }} />
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: themeOption.primaryDark
              }} />
            </div>

            {currentTheme === themeOption.key && (
              <div style={{
                position: 'absolute',
                top: '12px',
                right: '12px',
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: themeOption.primary,
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '12px',
                fontWeight: '600'
              }}>
                ✓
              </div>
            )}
          </button>
        ))}
      </div>

      <div style={{
        marginTop: '20px',
        padding: '16px',
        borderRadius: '12px',
        background: theme.surfaceDark,
        border: `1px solid ${theme.border}`
      }}>
        <p style={{
          margin: 0,
          fontSize: '13px',
          color: theme.textMuted,
          textAlign: 'center'
        }}>
          🌈 Your theme preference is automatically saved and will persist across sessions
        </p>
      </div>
    </div>
  );
};

export default ThemeSelector;