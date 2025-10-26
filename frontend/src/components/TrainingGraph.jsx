import React, { useEffect, useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrainingGraph = ({ websocketUrl, isTraining, onTrainingComplete }) => {
  const [lossData, setLossData] = useState({
    epochs: [],
    trainLoss: [],
    testLoss: [],
    isAveraged: false,
    intervalSize: 1
  });
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [lastUpdate, setLastUpdate] = useState(null);
  const websocketRef = useRef(null);

  useEffect(() => {
    if (isTraining && websocketUrl) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [isTraining, websocketUrl]);

  const connectWebSocket = () => {
    try {
      // Create WebSocket connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = websocketUrl.replace(/^https?:\/\//, '');
      const wsUrl = `${protocol}//${host}/training-loss`;

      websocketRef.current = new WebSocket(wsUrl);

      websocketRef.current.onopen = () => {
        console.log('WebSocket connected for training loss');
        setConnectionStatus('Connected');
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received WebSocket message:', data);

          if (data.type === 'loss_update') {
            setLossData(prevData => ({
              epochs: [...prevData.epochs, data.epoch],
              trainLoss: [...prevData.trainLoss, data.train_loss],
              testLoss: [...prevData.testLoss, data.test_loss],
              isAveraged: data.is_averaged || false,
              intervalSize: data.interval_size || 1
            }));
            setLastUpdate(new Date().toLocaleTimeString());
          } else if (data.type === 'training_complete') {
            setConnectionStatus('Training Complete');
            if (onTrainingComplete) {
              onTrainingComplete();
            }
          } else if (data.type === 'error') {
            console.error('Training error:', data.message);
            setConnectionStatus(`Error: ${data.message}`);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('Connection Error');
      };

      websocketRef.current.onclose = () => {
        console.log('WebSocket connection closed');
        setConnectionStatus('Disconnected');
      };

    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setConnectionStatus('Connection Failed');
    }
  };

  const disconnectWebSocket = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
  };

  const chartData = {
    labels: lossData.epochs,
    datasets: [
      {
        label: 'Training Loss',
        data: lossData.trainLoss,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
      {
        label: 'Test Loss',
        data: lossData.testLoss,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: lossData.isAveraged
          ? `Training Loss Over Time (Averaged over ${lossData.intervalSize} epochs)`
          : 'Training Loss Over Time',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Epoch'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Loss'
        },
        beginAtZero: false,
      },
    },
    animation: {
      duration: 300, // Reduced duration for real-time updates
    },
  };

  const clearData = () => {
    setLossData({
      epochs: [],
      trainLoss: [],
      testLoss: [],
      isAveraged: false,
      intervalSize: 1
    });
    setLastUpdate(null);
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'Connected':
        return '#22c55e';
      case 'Training Complete':
        return '#3b82f6';
      case 'Disconnected':
        return '#6b7280';
      default:
        return '#ef4444';
    }
  };

  return (
    <div style={{
      background: 'white',
      borderRadius: '20px',
      padding: '30px',
      boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
      border: '1px solid rgba(102, 126, 234, 0.2)',
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        paddingBottom: '15px',
        borderBottom: '2px solid #f1f5f9'
      }}>
        <div>
          <h3 style={{
            margin: '0 0 5px 0',
            fontSize: '20px',
            fontWeight: '700',
            color: '#2d3748'
          }}>
            📈 Real-Time Training Loss
          </h3>
          <p style={{
            margin: 0,
            fontSize: '14px',
            color: '#64748b'
          }}>
            Live visualization of model training progress
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          {/* Connection Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            borderRadius: '20px',
            background: `${getStatusColor()}20`,
            border: `1px solid ${getStatusColor()}40`
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: getStatusColor(),
              animation: connectionStatus === 'Connected' ? 'pulse 2s infinite' : 'none'
            }}></div>
            <span style={{
              fontSize: '12px',
              fontWeight: '600',
              color: getStatusColor()
            }}>
              {connectionStatus}
            </span>
          </div>

          {/* Clear Button */}
          <button
            onClick={clearData}
            style={{
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '6px 12px',
              fontSize: '12px',
              fontWeight: '500',
              color: '#64748b',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#f1f5f9';
              e.currentTarget.style.borderColor = '#cbd5e1';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#f8fafc';
              e.currentTarget.style.borderColor = '#e2e8f0';
            }}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Stats */}
      {lossData.epochs.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '15px',
          marginBottom: '20px'
        }}>
          <div style={{
            background: '#f8fafc',
            borderRadius: '12px',
            padding: '12px',
            textAlign: 'center',
            border: '1px solid #e2e8f0'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              {lossData.epochs.length}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500' }}>
              Epochs
            </div>
          </div>
          <div style={{
            background: 'rgba(75, 192, 192, 0.1)',
            borderRadius: '12px',
            padding: '12px',
            textAlign: 'center',
            border: '1px solid rgba(75, 192, 192, 0.2)'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#059669' }}>
              {lossData.trainLoss.length > 0 ? lossData.trainLoss[lossData.trainLoss.length - 1].toFixed(4) : '-'}
            </div>
            <div style={{ fontSize: '12px', color: '#059669', fontWeight: '500' }}>
              Train Loss
            </div>
          </div>
          <div style={{
            background: 'rgba(255, 99, 132, 0.1)',
            borderRadius: '12px',
            padding: '12px',
            textAlign: 'center',
            border: '1px solid rgba(255, 99, 132, 0.2)'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#dc2626' }}>
              {lossData.testLoss.length > 0 ? lossData.testLoss[lossData.testLoss.length - 1].toFixed(4) : '-'}
            </div>
            <div style={{ fontSize: '12px', color: '#dc2626', fontWeight: '500' }}>
              Test Loss
            </div>
          </div>
          <div style={{
            background: '#f8fafc',
            borderRadius: '12px',
            padding: '12px',
            textAlign: 'center',
            border: '1px solid #e2e8f0'
          }}>
            <div style={{ fontSize: '12px', fontWeight: '700', color: '#1e293b' }}>
              {lastUpdate || '-'}
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', fontWeight: '500' }}>
              Last Update
            </div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div style={{
        flex: 1,
        minHeight: '300px',
        position: 'relative'
      }}>
        {lossData.epochs.length > 0 ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#64748b'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '15px', opacity: 0.5 }}>
              📊
            </div>
            <h4 style={{
              margin: '0 0 8px 0',
              fontSize: '18px',
              fontWeight: '600'
            }}>
              Waiting for Training Data
            </h4>
            <p style={{
              margin: 0,
              fontSize: '14px',
              textAlign: 'center',
              lineHeight: '1.5'
            }}>
              {isTraining
                ? 'Connected to training stream. Data will appear here as training progresses.'
                : 'Start model training to see real-time loss visualization.'
              }
            </p>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default TrainingGraph;