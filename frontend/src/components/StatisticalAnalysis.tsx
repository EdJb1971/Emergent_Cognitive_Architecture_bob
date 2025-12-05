import React, { useState, useEffect } from 'react';
import * as FaIcons from 'react-icons/fa';
import { getDashboardStatisticalAnalysis, getLearningCurvesAnalysis, exportDashboardData } from '../api/dashboardApi';

interface StatisticalAnalysisProps {
  isVisible: boolean;
}

interface StatisticalResult {
  sample_size: number;
  descriptive_stats?: {
    mean: number;
    median: number;
    std_dev: number;
    min: number;
    max: number;
  };
  trend_analysis?: {
    slope: number;
    r_squared: number;
    p_value: number;
    direction: string;
    significance: string;
  };
  distribution?: {
    is_normal: boolean | null;
    outliers: {
      count: number;
      values: number[];
    };
  };
  // Learning analysis properties
  power_law_fit?: {
    a: number;
    b: number;
    r_squared: number;
    p_value: number;
    goodness_of_fit: string;
  };
  learning_characteristics?: {
    learning_rate: number;
    efficiency: string;
    convergence_indicated: boolean;
    samples: number;
  };
  performance_trajectory?: {
    initial_performance: number;
    final_performance: number;
    improvement: number;
    improvement_rate: number;
  };
  // Error states
  insufficient_data?: boolean;
  fitting_error?: string;
}

const StatisticalAnalysis: React.FC<StatisticalAnalysisProps> = ({ isVisible }) => {
  const [analysisResults, setAnalysisResults] = useState<Record<string, StatisticalResult>>({});
  const [learningCurves, setLearningCurves] = useState<any>({});
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['user_satisfaction', 'processing_time']);
  const [isLoading, setIsLoading] = useState(false);
  const [exportFormat, setExportFormat] = useState<'csv' | 'json'>('csv');

  const availableMetrics = [
    { value: 'user_satisfaction', label: 'User Satisfaction' },
    { value: 'processing_time', label: 'Processing Time' },
    { value: 'learning_performance', label: 'Learning Performance' }
  ];

  useEffect(() => {
    if (isVisible) {
      loadStatisticalAnalysis();
      loadLearningCurves();
    }
  }, [isVisible, selectedMetrics]);

  const loadStatisticalAnalysis = async () => {
    setIsLoading(true);
    try {
      const results = await getDashboardStatisticalAnalysis(selectedMetrics.join(','));
      setAnalysisResults(results.results || {});
    } catch (error) {
      console.error('Failed to load statistical analysis:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadLearningCurves = async () => {
    try {
      const curves = await getLearningCurvesAnalysis();
      setLearningCurves(curves.results || {});
    } catch (error) {
      console.error('Failed to load learning curves:', error);
    }
  };

  const handleExport = async (format: 'csv' | 'json', dataType: string) => {
    try {
      await exportDashboardData(format, dataType);
    } catch (error) {
      console.error('Failed to export data:', error);
    }
  };

  const formatNumber = (num: number, decimals: number = 3) => {
    return num.toFixed(decimals);
  };

  const getSignificanceColor = (pValue: number) => {
    if (pValue < 0.001) return 'text-red-600 font-bold';
    if (pValue < 0.01) return 'text-orange-600 font-semibold';
    if (pValue < 0.05) return 'text-yellow-600';
    return 'text-gray-500';
  };

  if (!isVisible) return null;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
          <FaIcons.FaChartLine className="mr-2 text-blue-600" />
          Statistical Analysis & Research Tools
        </h3>

        {/* Export Controls */}
        <div className="flex space-x-2">
          <select
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value as 'csv' | 'json')}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
            aria-label="Export format selection"
          >
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
          </select>
          <button
            onClick={() => handleExport(exportFormat, 'dashboard')}
            className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700"
          >
            Export Dashboard
          </button>
          <button
            onClick={() => handleExport(exportFormat, 'scientific')}
            className="px-3 py-1 bg-green-600 text-white rounded-md text-sm hover:bg-green-700"
          >
            Export Scientific
          </button>
        </div>
      </div>

      {/* Metric Selection */}
      <div className="mb-6">
        <h4 className="text-md font-medium mb-3 text-gray-700 dark:text-gray-300">Select Metrics for Analysis</h4>
        <div className="flex flex-wrap gap-2">
          {availableMetrics.map(metric => (
            <label key={metric.value} className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedMetrics.includes(metric.value)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedMetrics([...selectedMetrics, metric.value]);
                  } else {
                    setSelectedMetrics(selectedMetrics.filter(m => m !== metric.value));
                  }
                }}
                className="rounded border-gray-300"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">{metric.label}</span>
            </label>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600 dark:text-gray-400">Performing statistical analysis...</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Statistical Analysis Results */}
          {Object.entries(analysisResults).map(([metricName, result]) => (
            <div key={metricName} className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
              <h4 className="text-md font-semibold mb-3 text-gray-900 dark:text-white capitalize">
                {metricName.replace('_', ' ')} Analysis
              </h4>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* First Column: Descriptive Stats or Learning Analysis */}
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded">
                  {result.descriptive_stats ? (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Descriptive Stats</h5>
                      <div className="space-y-1 text-sm">
                        <div>Mean: {formatNumber(result.descriptive_stats.mean)}</div>
                        <div>Median: {formatNumber(result.descriptive_stats.median)}</div>
                        <div>Std Dev: {formatNumber(result.descriptive_stats.std_dev)}</div>
                        <div>Min: {formatNumber(result.descriptive_stats.min)}</div>
                        <div>Max: {formatNumber(result.descriptive_stats.max)}</div>
                        <div>Sample Size: {result.sample_size}</div>
                      </div>
                    </>
                  ) : result.power_law_fit ? (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Learning Analysis</h5>
                      <div className="space-y-1 text-sm">
                        <div>Learning Rate: {formatNumber(result.learning_characteristics?.learning_rate || 0)}</div>
                        <div>Efficiency: <span className="capitalize">{result.learning_characteristics?.efficiency || 'unknown'}</span></div>
                        <div>Power-law R²: {formatNumber(result.power_law_fit.r_squared)}</div>
                        <div>Improvement: {formatNumber(result.performance_trajectory?.improvement || 0)}</div>
                        <div>Samples: {result.learning_characteristics?.samples || 0}</div>
                      </div>
                    </>
                  ) : (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Analysis Status</h5>
                      <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                        {(result as any).insufficient_data ? (
                          <div>Insufficient data ({result.sample_size} samples)</div>
                        ) : (result as any).fitting_error ? (
                          <div>Fitting error: {(result as any).fitting_error}</div>
                        ) : (
                          <div>Analysis completed</div>
                        )}
                      </div>
                    </>
                  )}
                </div>

                {/* Second Column: Trend Analysis or Learning Trajectory */}
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded">
                  {result.trend_analysis ? (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Trend Analysis</h5>
                      <div className="space-y-1 text-sm">
                        <div>Slope: {formatNumber(result.trend_analysis.slope)}</div>
                        <div>R²: {formatNumber(result.trend_analysis.r_squared)}</div>
                        <div className={getSignificanceColor(result.trend_analysis.p_value)}>
                          p-value: {formatNumber(result.trend_analysis.p_value)}
                        </div>
                        <div>Direction: <span className="capitalize">{result.trend_analysis.direction}</span></div>
                        <div>Significance: <span className="capitalize">{result.trend_analysis.significance}</span></div>
                      </div>
                    </>
                  ) : result.learning_characteristics ? (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Learning Trajectory</h5>
                      <div className="space-y-1 text-sm">
                        <div>Initial: {formatNumber(result.performance_trajectory?.initial_performance || 0)}</div>
                        <div>Final: {formatNumber(result.performance_trajectory?.final_performance || 0)}</div>
                        <div>Improvement Rate: {formatNumber(result.performance_trajectory?.improvement_rate || 0)}</div>
                        <div>Convergence: {result.learning_characteristics.convergence_indicated ? 'Yes' : 'No'}</div>
                        <div>Fit Quality: <span className="capitalize">{result.power_law_fit?.goodness_of_fit || 'unknown'}</span></div>
                      </div>
                    </>
                  ) : (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Status</h5>
                      <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                        <div>No trend analysis available</div>
                      </div>
                    </>
                  )}
                </div>

                {/* Third Column: Distribution Analysis */}
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded">
                  {result.distribution ? (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Distribution</h5>
                      <div className="space-y-1 text-sm">
                        <div>Normal: {result.distribution.is_normal === null ? 'Unknown' : result.distribution.is_normal ? 'Yes' : 'No'}</div>
                        <div>Outliers: {result.distribution.outliers.count}</div>
                        {result.distribution.outliers.count > 0 && (
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            Values: {result.distribution.outliers.values.slice(0, 3).join(', ')}
                            {result.distribution.outliers.values.length > 3 && '...'}
                          </div>
                        )}
                      </div>
                    </>
                  ) : (
                    <>
                      <h5 className="font-medium text-gray-900 dark:text-white mb-2">Distribution</h5>
                      <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                        <div>No distribution analysis available</div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Learning Curves Analysis */}
          {Object.keys(learningCurves).length > 0 && (
            <div className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
              <h4 className="text-md font-semibold mb-3 text-gray-900 dark:text-white">
                Learning Curves (Power-Law Analysis)
              </h4>

              <div className="space-y-3">
                {Object.entries(learningCurves).map(([skillName, analysis]: [string, any]) => (
                  <div key={skillName} className="bg-gray-50 dark:bg-gray-700 p-3 rounded">
                    <h5 className="font-medium text-gray-900 dark:text-white mb-2 capitalize">
                      {skillName.replace('_', ' ')}
                    </h5>

                    {analysis.power_law_fit ? (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                        <div>R²: {formatNumber(analysis.power_law_fit.r_squared)}</div>
                        <div>p-value: {formatNumber(analysis.power_law_fit.p_value)}</div>
                        <div>Learning Rate: {formatNumber(analysis.learning_characteristics.learning_rate)}</div>
                        <div>Efficiency: <span className="capitalize">{analysis.learning_characteristics.efficiency}</span></div>
                      </div>
                    ) : (
                      <div className="text-gray-500 text-sm">Insufficient data for power-law analysis</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Research Tools */}
          <div className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
            <h4 className="text-md font-semibold mb-3 text-gray-900 dark:text-white">
              Research Tools
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button
                onClick={() => handleExport('json', 'scientific')}
                className="p-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                <FaIcons.FaFileExport className="inline mr-2" />
                Generate Research Report
              </button>

              <button
                onClick={() => handleExport('csv', 'historical')}
                className="p-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
              >
                <FaIcons.FaDownload className="inline mr-2" />
                Export Historical Data
              </button>
            </div>

            <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              <p><strong>Research Applications:</strong></p>
              <ul className="list-disc list-inside mt-1 space-y-1">
                <li>Statistical validation of learning improvements</li>
                <li>Power-law analysis of skill acquisition curves</li>
                <li>Emergence detection in multi-agent coordination</li>
                <li>Comparative analysis with baseline systems</li>
                <li>Publication-ready data exports</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StatisticalAnalysis;