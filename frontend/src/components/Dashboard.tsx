import React, { useState, useEffect, useRef } from 'react';
import { FaChartLine, FaTimes, FaCog, FaUsers, FaBrain, FaMemory, FaCalculator } from 'react-icons/fa';
import { DashboardMetrics, DashboardConfig, createMockDashboardMetrics } from 'types/dashboard';
import { getDashboardMetrics, getDashboardHistory, getDashboardCorrelations, createDashboardWebSocket } from 'api/dashboardApi';
import MetricsCard from 'components/MetricsCard';
import RealTimeChart from 'components/RealTimeChart';
import LearningProgress from 'components/LearningProgress';
import AgentActivity from 'components/AgentActivity';
import StatisticalAnalysis from 'components/StatisticalAnalysis';

interface DashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis'>('overview');
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [correlations, setCorrelations] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<DashboardConfig>({
    showRealTime: true,
    showHistorical: false,
    timeRange: '24h',
    refreshInterval: 30,
    selectedMetrics: ['learning', 'memory', 'emergence', 'performance']
  });

  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadMetrics();
      setupWebSocket();
      setupAutoRefresh();
    } else {
      cleanup();
    }

    return () => cleanup();
  }, [isOpen, config.refreshInterval, config.selectedMetrics]);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getDashboardMetrics();
      setMetrics(data);

      // Fetch correlations data if emergence metrics are selected
      if (config.selectedMetrics.includes('emergence')) {
        try {
          const correlationsData = await getDashboardCorrelations(24);
          setCorrelations(correlationsData);
        } catch (corrError) {
          console.warn('Failed to load correlations data:', corrError);
          // Don't fail the whole dashboard for correlations
        }
      }
    } catch (err: any) {
      console.error('Dashboard API error:', err);
      // Fall back to mock data for development
      if (process.env.NODE_ENV === 'development') {
        console.log('Using mock dashboard data for development');
        setMetrics(createMockDashboardMetrics());
        setError(null);
      } else {
        setError(err.message || 'Failed to load dashboard metrics');
      }
    } finally {
      setLoading(false);
    }
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = createDashboardWebSocket(
        (data) => {
          if (data.type === 'update' && data.data) {
            setMetrics(prev => prev ? { ...prev, ...data.data } : data.data);
          }
        },
        (error) => {
          console.error('WebSocket error:', error);
          // Fallback to polling if WebSocket fails
          setupAutoRefresh();
        }
      );
    } catch (err) {
      console.error('Failed to setup WebSocket:', err);
      setupAutoRefresh();
    }
  };

  const setupAutoRefresh = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(loadMetrics, config.refreshInterval * 1000);
  };

  const cleanup = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const toggleMetric = (metric: string) => {
    setConfig(prev => ({
      ...prev,
      selectedMetrics: prev.selectedMetrics.includes(metric)
        ? prev.selectedMetrics.filter(m => m !== metric)
        : [...prev.selectedMetrics, metric]
    }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-7xl h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <FaChartLine className="text-blue-600 text-2xl" />
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                ECA Scientific Dashboard
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Real-time cognitive architecture performance monitoring
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Close Dashboard"
            aria-label="Close Dashboard"
          >
            <FaTimes className="text-gray-500" />
          </button>
        </div>

        {/* Controls */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          {/* Tab Navigation */}
          <div className="flex space-x-1 mb-4">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'overview'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'
              }`}
            >
              <FaChartLine className="inline mr-2" />
              Overview
            </button>
            <button
              onClick={() => setActiveTab('analysis')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'analysis'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'
              }`}
            >
              <FaCalculator className="inline mr-2" />
              Statistical Analysis
            </button>
          </div>

          {/* Controls - only show for overview tab */}
          {activeTab === 'overview' && (
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center space-x-2">
                <label htmlFor="refresh-interval" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Auto-refresh:
                </label>
                <select
                  id="refresh-interval"
                  value={config.refreshInterval}
                  onChange={(e) => setConfig(prev => ({ ...prev, refreshInterval: parseInt(e.target.value) }))}
                  className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
                >
                  <option value={10}>10s</option>
                  <option value={30}>30s</option>
                  <option value={60}>1m</option>
                  <option value={300}>5m</option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                {['learning', 'memory', 'emergence', 'performance'].map(metric => (
                  <label key={metric} className="flex items-center space-x-1 text-sm">
                    <input
                      type="checkbox"
                      checked={config.selectedMetrics.includes(metric)}
                      onChange={() => toggleMetric(metric)}
                      className="rounded border-gray-300 dark:border-gray-600"
                    />
                    <span className="capitalize text-gray-700 dark:text-gray-300">{metric}</span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === 'overview' ? (
            <>
              {loading && !metrics ? (
                <div className="flex items-center justify-center h-64">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                </div>
              ) : error ? (
                <div className="bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-4">
                  <p className="text-red-800 dark:text-red-200">{error}</p>
                  <button
                    onClick={loadMetrics}
                    className="mt-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
                  >
                    Retry
                  </button>
                </div>
              ) : metrics ? (
                <div className="space-y-6">
                  {/* System Overview */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <MetricsCard
                      title="Total Events"
                      value={metrics.summary?.total_events || 0}
                      icon={<FaCog className="text-blue-600" />}
                      trend="+12%"
                      color="blue"
                    />
                    <MetricsCard
                      title="Active Users"
                      value={metrics.summary?.active_users || 0}
                      icon={<FaUsers className="text-green-600" />}
                      trend="+5%"
                      color="green"
                    />
                    <MetricsCard
                      title="Events/Min"
                      value={`${(metrics.summary?.events_per_minute || 0).toFixed(1)}`}
                      icon={<FaBrain className="text-purple-600" />}
                      trend="+8%"
                      color="purple"
                    />
                    <MetricsCard
                      title="Avg Processing Time"
                      value={`${(metrics.summary?.avg_processing_time_ms || 0).toFixed(1)}ms`}
                      icon={<FaChartLine className="text-orange-600" />}
                      trend="-8%"
                      color="orange"
                    />
                  </div>

                  {/* Detailed Metrics */}
                  {config.selectedMetrics.includes('learning') && (
                    <LearningProgress metrics={{
                      skill_improvement_rates: Object.fromEntries(
                        Object.entries(metrics.learning_metrics?.skill_performance || {}).map(([skill, data]) => [
                          skill,
                          [data.current_avg] // Convert to array format expected by component
                        ])
                      ),
                      recent_learning_events: [], // TODO: Add from backend if available
                      average_skill_performance: Object.values(metrics.learning_metrics?.skill_performance || {}).reduce((sum, skill) => sum + skill.current_avg, 0) /
                        Math.max(1, Object.keys(metrics.learning_metrics?.skill_performance || {}).length),
                      total_skills_tracked: Object.keys(metrics.learning_metrics?.skill_performance || {}).length
                    }} />
                  )}

                  {config.selectedMetrics.includes('memory') && (
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center">
                        <FaMemory className="mr-2 text-blue-600" />
                        Memory Performance
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {Object.entries(metrics.memory_metrics?.tier_performance || {}).map(([tier, stats]: [string, any]) => (
                          <MetricsCard
                            key={tier}
                            title={`${tier.toUpperCase()} Hit Rate`}
                            value={`${(stats.hit_rate * 100).toFixed(1)}%`}
                            icon={<FaMemory className="text-blue-600" />}
                            color="blue"
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {config.selectedMetrics.includes('emergence') && (
                    <AgentActivity metrics={{
                      agent_activation_counts: metrics.agent_metrics?.activation_frequencies || {},
                      conflict_resolution_rate: Math.min(1.0, (metrics.conflict_metrics?.coherence_improvement || 0) / 100), // Use coherence as proxy
                      average_coherence_score: metrics.conflict_metrics?.coherence_improvement || 0,
                      novel_behaviors_detected: correlations?.executive_summary?.emergence_indicators?.novel_behaviors_detected || 0,
                      emergence_level: correlations?.executive_summary?.emergence_indicators?.emergence_level
                    }} />
                  )}

                  {config.selectedMetrics.includes('performance') && (
                    <RealTimeChart
                      title="System Performance"
                      data={{
                        average_response_time: metrics.summary?.avg_processing_time_ms || 0,
                        error_rate: 0, // TODO: Add from backend if available
                        throughput_cycles_per_minute: metrics.summary?.events_per_minute || 0,
                        memory_usage_mb: 0 // TODO: Add from backend if available
                      }}
                      timeRange={config.timeRange}
                    />
                  )}
                </div>
              ) : null}
            </>
          ) : (
            <StatisticalAnalysis isVisible={activeTab === 'analysis'} />
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;