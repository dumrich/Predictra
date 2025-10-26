import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const DataVisualization = ({ distributionData, isLoading }) => {
  const [selectedColumn, setSelectedColumn] = useState(null);

  useEffect(() => {
    if (distributionData && distributionData.columns && distributionData.columns.length > 0) {
      setSelectedColumn(distributionData.columns[0]);
    }
  }, [distributionData]);

  if (isLoading) {
    return (
      <div style={{
        background: 'white',
        borderRadius: '20px',
        padding: '40px',
        boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
        border: '1px solid rgba(102, 126, 234, 0.2)',
        textAlign: 'center'
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          border: '4px solid #e2e8f0',
          borderTop: '4px solid #667eea',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto 16px auto'
        }}></div>
        <h3 style={{
          margin: '0 0 8px 0',
          fontSize: '18px',
          fontWeight: '600',
          color: '#4a5568'
        }}>
          Analyzing Distribution
        </h3>
        <p style={{
          margin: 0,
          fontSize: '14px',
          color: '#718096'
        }}>
          Computing statistical distributions...
        </p>
      </div>
    );
  }

  if (!distributionData || !distributionData.distributions) {
    return (
      <div style={{
        background: 'white',
        borderRadius: '20px',
        padding: '40px',
        boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
        border: '1px solid rgba(102, 126, 234, 0.2)',
        textAlign: 'center',
        color: '#718096'
      }}>
        <div style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.5 }}>
          📊
        </div>
        <h3 style={{
          fontSize: '18px',
          fontWeight: '600',
          margin: '0 0 8px 0'
        }}>
          No Distribution Data
        </h3>
        <p style={{
          margin: 0,
          fontSize: '14px'
        }}>
          Select a dataset to view distributions
        </p>
      </div>
    );
  }

  const currentDistribution = distributionData.distributions[selectedColumn];

  const renderChart = () => {
    if (!currentDistribution) return null;

    if (currentDistribution.type === 'categorical') {
      // Bar chart for categorical data
      const chartData = {
        labels: currentDistribution.categories,
        datasets: [
          {
            label: 'Count',
            data: currentDistribution.counts,
            backgroundColor: 'rgba(102, 126, 234, 0.6)',
            borderColor: 'rgba(102, 126, 234, 1)',
            borderWidth: 1,
          },
        ],
      };

      const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          title: {
            display: true,
            text: `${selectedColumn.replace(/_/g, ' ')} Distribution`,
            font: {
              size: 16,
              weight: 'bold'
            }
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Count'
            }
          },
          x: {
            title: {
              display: true,
              text: selectedColumn.replace(/_/g, ' ')
            }
          }
        },
      };

      return <Bar data={chartData} options={options} />;

    } else if (currentDistribution.type === 'numeric') {
      // Histogram for numeric data
      const chartData = {
        labels: currentDistribution.histogram.bin_centers.map(center =>
          typeof center === 'number' ? center.toFixed(2) : center
        ),
        datasets: [
          {
            label: 'Frequency',
            data: currentDistribution.histogram.counts,
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
          },
        ],
      };

      const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          title: {
            display: true,
            text: `${selectedColumn.replace(/_/g, ' ')} Histogram`,
            font: {
              size: 16,
              weight: 'bold'
            }
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Frequency'
            }
          },
          x: {
            title: {
              display: true,
              text: selectedColumn.replace(/_/g, ' ')
            }
          }
        },
      };

      return <Bar data={chartData} options={options} />;
    }

    return null;
  };

  const renderStats = () => {
    if (!currentDistribution) return null;

    if (currentDistribution.type === 'categorical') {
      return (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '16px'
        }}>
          <div style={{
            background: '#f8fafc',
            padding: '16px',
            borderRadius: '12px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#1e293b' }}>
              {currentDistribution.unique_values}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500' }}>
              Unique Categories
            </div>
          </div>
          <div style={{
            background: '#f8fafc',
            padding: '16px',
            borderRadius: '12px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#1e293b' }}>
              {currentDistribution.total_count}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500' }}>
              Total Count
            </div>
          </div>
        </div>
      );
    } else if (currentDistribution.type === 'numeric') {
      const stats = currentDistribution.statistics;
      return (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '12px'
        }}>
          <div style={{
            background: '#f8fafc',
            padding: '12px',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              {stats.mean.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>
              Mean
            </div>
          </div>
          <div style={{
            background: '#f8fafc',
            padding: '12px',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              {stats.median.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>
              Median
            </div>
          </div>
          <div style={{
            background: '#f8fafc',
            padding: '12px',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              {stats.std.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>
              Std Dev
            </div>
          </div>
          <div style={{
            background: '#f8fafc',
            padding: '12px',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              {stats.min.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>
              Min
            </div>
          </div>
          <div style={{
            background: '#f8fafc',
            padding: '12px',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              {stats.max.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>
              Max
            </div>
          </div>
          <div style={{
            background: '#f8fafc',
            padding: '12px',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              {currentDistribution.non_null_count}
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>
              Valid
            </div>
          </div>
        </div>
      );
    }

    return null;
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
            📊 Dataset Distribution
          </h3>
          <p style={{
            margin: 0,
            fontSize: '14px',
            color: '#64748b'
          }}>
            Statistical analysis and visualization
          </p>
        </div>

        {/* Column Selector */}
        <select
          value={selectedColumn || ''}
          onChange={(e) => setSelectedColumn(e.target.value)}
          style={{
            padding: '8px 12px',
            borderRadius: '8px',
            border: '1px solid #e2e8f0',
            fontSize: '14px',
            fontWeight: '500',
            color: '#2d3748',
            background: 'white',
            cursor: 'pointer',
            minWidth: '120px'
          }}
        >
          {distributionData.columns.map((column) => (
            <option key={column} value={column}>
              {column.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      </div>

      {/* Dataset Summary */}
      <div style={{
        background: '#f8fafc',
        borderRadius: '12px',
        padding: '16px',
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-around',
        alignItems: 'center'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
            {distributionData.total_rows}
          </div>
          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500' }}>
            Total Rows
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
            {distributionData.columns.length}
          </div>
          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500' }}>
            Columns
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
            {currentDistribution?.type || 'N/A'}
          </div>
          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: '500' }}>
            Type
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div style={{ marginBottom: '20px' }}>
        {renderStats()}
      </div>

      {/* Chart */}
      <div style={{
        flex: 1,
        minHeight: '300px',
        position: 'relative'
      }}>
        {renderChart()}
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default DataVisualization;