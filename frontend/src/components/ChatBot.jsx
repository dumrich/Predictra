import React, { useState, useRef, useEffect } from 'react';

const ChatBot = ({ distributionData }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      message: '👋 Hello! I\'m your Distribution Assistant. Ask me about the probability distributions in your dataset!',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const [isInitialized, setIsInitialized] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    // Only auto-scroll after initial load and when new messages are added
    if (isInitialized) {
      scrollToBottom();
    }
  }, [messages, isInitialized]);

  // Mark as initialized after component mounts to prevent initial auto-scroll
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsInitialized(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  // Dummy responses for distribution queries
  const generateResponse = (userMessage) => {
    const message = userMessage.toLowerCase();

    // Sample responses based on keywords
    if (message.includes('mean') || message.includes('average')) {
      return "📊 Great question about the mean! The mean represents the central tendency of your data distribution. For numeric fields, I can help you understand if your data is normally distributed or has any skewness. Which column are you curious about?";
    }

    if (message.includes('standard deviation') || message.includes('std') || message.includes('variance')) {
      return "📈 Standard deviation measures the spread of your data! A low standard deviation means data points are close to the mean, while a high standard deviation indicates more spread out values. This helps identify outliers and data consistency.";
    }

    if (message.includes('distribution') || message.includes('histogram')) {
      return "📊 I can see the distribution patterns in your data! Are you interested in understanding if your data follows a normal distribution, or would you like me to explain any skewness or unusual patterns I observe?";
    }

    if (message.includes('outlier') || message.includes('anomaly')) {
      return "🔍 Outliers are data points that fall significantly outside the typical range. I can help you identify potential outliers by looking at values beyond 2-3 standard deviations from the mean. Would you like me to analyze a specific column?";
    }

    if (message.includes('normal') || message.includes('gaussian')) {
      return "📐 A normal distribution forms the classic bell curve! Most real-world data has some deviation from perfect normality. I can help you assess how close your data is to a normal distribution and what that means for your analysis.";
    }

    if (message.includes('skew') || message.includes('asymmetric')) {
      return "⚖️ Skewness indicates whether your data leans more towards higher or lower values. Positive skew means a longer tail on the right side, while negative skew means a longer tail on the left. This affects which statistical measures are most appropriate.";
    }

    if (message.includes('categorical') || message.includes('category')) {
      return "🏷️ For categorical data, I can help you understand the frequency distribution of different categories. Are there dominant categories? Is the distribution balanced? This affects how representative your sample is.";
    }

    if (message.includes('correlation') || message.includes('relationship')) {
      return "🔗 While I focus on individual distributions, relationships between variables are fascinating! You might want to explore scatter plots or correlation matrices to understand how different variables relate to each other.";
    }

    if (message.includes('predict') || message.includes('model')) {
      return "🤖 Understanding your data distribution is crucial for building good predictive models! Normal distributions work well with linear models, while skewed data might benefit from transformations or tree-based models.";
    }

    if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
      return "👋 Hello! I'm excited to help you explore your data distributions. What would you like to know about your dataset's statistical patterns?";
    }

    if (message.includes('help') || message.includes('what can you do')) {
      return "🚀 I can help you understand:\n• Distribution shapes and patterns\n• Statistical measures (mean, median, std dev)\n• Outlier detection\n• Skewness and normality\n• Categorical frequency analysis\n• Data quality insights\n\nJust ask me about any aspect of your data distribution!";
    }

    // Default responses
    const defaultResponses = [
      "🤔 That's an interesting question about your data distribution! Could you be more specific about which column or statistical aspect you'd like to explore?",
      "📊 I'd love to help you understand that better! Try asking about specific columns, statistical measures, or distribution patterns you're curious about.",
      "🔍 Great question! I can provide insights about means, standard deviations, outliers, skewness, and distribution shapes. What specifically would you like to know?",
      "📈 I'm here to help you decode your data patterns! Ask me about any statistical concepts or specific columns you'd like to understand better.",
      "🎯 Let me help you with that! I specialize in explaining distribution characteristics, statistical measures, and data patterns. What aspect interests you most?"
    ];

    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    // Add user message
    const userMsg = {
      id: Date.now(),
      type: 'user',
      message: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsTyping(true);

    // Simulate typing delay
    setTimeout(() => {
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        message: generateResponse(inputMessage),
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1000); // Random delay between 1-2 seconds
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div style={{
      background: 'white',
      borderRadius: '20px',
      boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
      border: '1px solid rgba(102, 126, 234, 0.2)',
      height: '600px',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        padding: '20px 30px',
        borderBottom: '2px solid #f1f5f9',
        background: 'linear-gradient(135deg, #667eea, #764ba2)',
        color: 'white'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px'
          }}>
            🤖
          </div>
          <div>
            <h3 style={{
              margin: '0 0 4px 0',
              fontSize: '18px',
              fontWeight: '700'
            }}>
              Distribution Assistant
            </h3>
            <p style={{
              margin: 0,
              fontSize: '12px',
              opacity: 0.9
            }}>
              Ask me about probability distributions
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        padding: '20px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              display: 'flex',
              justifyContent: msg.type === 'user' ? 'flex-end' : 'flex-start',
              alignItems: 'flex-start',
              gap: '8px'
            }}
          >
            {msg.type === 'bot' && (
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #667eea, #764ba2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '16px',
                flexShrink: 0
              }}>
                🤖
              </div>
            )}

            <div style={{
              maxWidth: '70%',
              padding: '12px 16px',
              borderRadius: msg.type === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              background: msg.type === 'user'
                ? 'linear-gradient(135deg, #667eea, #764ba2)'
                : '#f8fafc',
              color: msg.type === 'user' ? 'white' : '#2d3748',
              fontSize: '14px',
              lineHeight: '1.5',
              wordBreak: 'break-word',
              whiteSpace: 'pre-wrap'
            }}>
              {msg.message}
              <div style={{
                marginTop: '4px',
                fontSize: '11px',
                opacity: 0.7
              }}>
                {formatTime(msg.timestamp)}
              </div>
            </div>

            {msg.type === 'user' && (
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#e2e8f0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '16px',
                flexShrink: 0
              }}>
                👤
              </div>
            )}
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div style={{
            display: 'flex',
            justifyContent: 'flex-start',
            alignItems: 'flex-start',
            gap: '8px'
          }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '16px'
            }}>
              🤖
            </div>
            <div style={{
              padding: '12px 16px',
              borderRadius: '16px 16px 16px 4px',
              background: '#f8fafc',
              color: '#64748b',
              fontSize: '14px'
            }}>
              <div style={{
                display: 'flex',
                gap: '4px',
                alignItems: 'center'
              }}>
                <div style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: '#64748b',
                  animation: 'typing 1.4s ease-in-out infinite both'
                }}></div>
                <div style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: '#64748b',
                  animation: 'typing 1.4s ease-in-out 0.2s infinite both'
                }}></div>
                <div style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: '#64748b',
                  animation: 'typing 1.4s ease-in-out 0.4s infinite both'
                }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: '20px',
        borderTop: '2px solid #f1f5f9',
        background: '#f8fafc'
      }}>
        <div style={{
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-end'
        }}>
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about your data distribution..."
            style={{
              flex: 1,
              padding: '12px 16px',
              borderRadius: '12px',
              border: '2px solid #e2e8f0',
              fontSize: '14px',
              fontFamily: 'inherit',
              resize: 'none',
              minHeight: '20px',
              maxHeight: '80px',
              outline: 'none',
              transition: 'border-color 0.2s ease'
            }}
            onFocus={(e) => {
              e.target.style.borderColor = '#667eea';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = '#e2e8f0';
            }}
            rows="1"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isTyping}
            style={{
              padding: '12px 16px',
              borderRadius: '12px',
              border: 'none',
              background: (!inputMessage.trim() || isTyping)
                ? '#e2e8f0'
                : 'linear-gradient(135deg, #667eea, #764ba2)',
              color: (!inputMessage.trim() || isTyping) ? '#a0aec0' : 'white',
              fontSize: '14px',
              fontWeight: '600',
              cursor: (!inputMessage.trim() || isTyping) ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
              minWidth: '60px'
            }}
          >
            {isTyping ? '⏳' : '📤'}
          </button>
        </div>
      </div>

      <style jsx>{`
        @keyframes typing {
          0% {
            transform: scale(0.8);
            opacity: 0.5;
          }
          50% {
            transform: scale(1.2);
            opacity: 1;
          }
          100% {
            transform: scale(0.8);
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
};

export default ChatBot;