import React, { useEffect, useRef } from 'react';
import * as FaIcons from 'react-icons/fa';

interface PerformanceMetrics {
  average_response_time: number;
  error_rate: number;
  throughput_cycles_per_minute: number;
  memory_usage_mb: number;
}

interface RealTimeChartProps {
  title: string;
  data: PerformanceMetrics;
  timeRange: string;
}

const RealTimeChart: React.FC<RealTimeChartProps> = ({ title, data, timeRange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    drawChart();
  }, [data]);

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Chart dimensions
    const width = rect.width;
    const height = rect.height;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Sample data points (in a real implementation, this would come from historical data)
    const responseTimeData = [2.1, 1.8, 2.3, 1.9, 2.0, data.average_response_time];
    const throughputData = [45, 52, 48, 55, 50, data.throughput_cycles_per_minute];
    const errorRateData = [0.02, 0.01, 0.03, 0.015, 0.02, data.error_rate];

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Vertical grid lines
    for (let i = 0; i <= 5; i++) {
      const x = padding + (chartWidth * i) / 5;
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
    }

    // Horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = padding + (chartHeight * i) / 4;
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
    }
    ctx.stroke();

    // Draw response time line
    drawLine(ctx, responseTimeData, '#3b82f6', padding, chartWidth, chartHeight, padding);

    // Draw throughput line
    drawLine(ctx, throughputData, '#10b981', padding, chartWidth, chartHeight, padding);

    // Draw error rate line (scaled)
    drawLine(ctx, errorRateData.map(x => x * 100), '#ef4444', padding, chartWidth, chartHeight, padding);

    // Draw legend
    const legendY = padding + 20;
    drawLegendItem(ctx, '#3b82f6', 'Response Time (s)', padding, legendY);
    drawLegendItem(ctx, '#10b981', 'Throughput (cycles/min)', padding, legendY + 20);
    drawLegendItem(ctx, '#ef4444', 'Error Rate (%)', padding, legendY + 40);
  };

  const drawLine = (ctx: CanvasRenderingContext2D, data: number[], color: string, padding: number, chartWidth: number, chartHeight: number, chartPadding: number) => {
    const maxValue = Math.max(...data);
    const minValue = Math.min(...data);
    const range = maxValue - minValue || 1;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((value, index) => {
      const x = padding + (chartWidth * index) / (data.length - 1);
      const y = chartPadding + chartHeight - ((value - minValue) / range) * chartHeight;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw points
    ctx.fillStyle = color;
    data.forEach((value, index) => {
      const x = padding + (chartWidth * index) / (data.length - 1);
      const y = chartPadding + chartHeight - ((value - minValue) / range) * chartHeight;

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const drawLegendItem = (ctx: CanvasRenderingContext2D, color: string, label: string, x: number, y: number) => {
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 8, 12, 12);

    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.fillText(label, x + 16, y + 2);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center">
        <FaIcons.FaChartLine className="mr-2 text-orange-600" />
        {title}
      </h3>

      <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-lg font-bold text-blue-600">
            {data.average_response_time.toFixed(2)}s
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Avg Response Time
          </div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-600">
            {data.throughput_cycles_per_minute}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Cycles/Min
          </div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-red-600">
            {(data.error_rate * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Error Rate
          </div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-purple-600">
            {data.memory_usage_mb.toFixed(1)}MB
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Memory Usage
          </div>
        </div>
      </div>

      <div className="relative">
        <canvas
          ref={canvasRef}
          className="w-full h-64 border border-gray-200 dark:border-gray-600 rounded"
        />
      </div>

      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 text-center">
        Last 6 data points • Time range: {timeRange}
      </div>
    </div>
  );
};

export default RealTimeChart;