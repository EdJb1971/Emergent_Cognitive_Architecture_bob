import React from 'react';
import * as FaIcons from 'react-icons/fa';

interface EmergenceMetrics {
  agent_activation_counts: Record<string, number>;
  conflict_resolution_rate: number;
  average_coherence_score: number;
  novel_behaviors_detected: number;
  emergence_level?: string;
}

interface AgentActivityProps {
  metrics: EmergenceMetrics;
}

const AgentActivity: React.FC<AgentActivityProps> = ({ metrics }) => {
  const getAgentColor = (agent: string) => {
    const colors: Record<string, string> = {
      perception: 'bg-blue-500',
      emotional: 'bg-pink-500',
      memory: 'bg-purple-500',
      planning: 'bg-green-500',
      creative: 'bg-orange-500',
      critic: 'bg-red-500',
      discovery: 'bg-indigo-500',
      web_browsing: 'bg-cyan-500'
    };
    return colors[agent] || 'bg-gray-500';
  };

  const formatAgentName = (agent: string) => {
    return agent.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const maxActivations = Math.max(...Object.values(metrics.agent_activation_counts), 1);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center">
        <FaIcons.FaUsers className="mr-2 text-purple-600" />
        Agent Coordination & Emergence
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Agent Activation Heatmap */}
        <div>
          <h4 className="text-md font-medium mb-3 text-gray-700 dark:text-gray-300">
            Agent Activation Frequency
          </h4>
          <div className="space-y-3">
            {Object.entries(metrics.agent_activation_counts)
              .sort(([,a], [,b]) => b - a)
              .map(([agent, count]) => {
                const percentage = (count / maxActivations) * 100;
                return (
                  <div key={agent} className="flex items-center space-x-3">
                    <div className="w-24 text-sm text-gray-600 dark:text-gray-400 truncate">
                      {formatAgentName(agent)}
                    </div>
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${getAgentColor(agent)} transition-all duration-300`}
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                    <div className="w-12 text-sm font-medium text-gray-900 dark:text-white text-right">
                      {count}
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Emergence Metrics */}
        <div>
          <h4 className="text-md font-medium mb-3 text-gray-700 dark:text-gray-300">
            Emergence Indicators
          </h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-2">
                <FaIcons.FaCheckCircle className="text-green-500" />
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  Conflict Resolution Rate
                </span>
              </div>
              <span className="text-lg font-bold text-green-600">
                {(metrics.conflict_resolution_rate * 100).toFixed(1)}%
              </span>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-2">
                <FaIcons.FaUsers className="text-blue-500" />
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  Average Coherence Score
                </span>
              </div>
              <span className="text-lg font-bold text-blue-600">
                {(metrics.average_coherence_score * 100).toFixed(1)}%
              </span>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-2">
                <FaIcons.FaExclamationTriangle className="text-orange-500" />
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  Novel Behaviors Detected
                </span>
              </div>
              <span className="text-lg font-bold text-orange-600">
                {metrics.novel_behaviors_detected}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Coordination Insights */}
      <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg border border-purple-200 dark:border-purple-700">
        <h4 className="text-md font-medium mb-2 text-gray-900 dark:text-white">
          Coordination Insights
        </h4>
        <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
          <p>
            • <strong>Most Active:</strong> {Object.entries(metrics.agent_activation_counts).sort(([,a], [,b]) => b - a)[0]?.[0] || 'None'}
          </p>
          <p>
            • <strong>Conflict Resolution:</strong> {metrics.conflict_resolution_rate > 0.8 ? 'Excellent' : metrics.conflict_resolution_rate > 0.6 ? 'Good' : 'Needs Improvement'}
          </p>
          <p>
            • <strong>Emergence Level:</strong> {metrics.emergence_level ? metrics.emergence_level.charAt(0).toUpperCase() + metrics.emergence_level.slice(1) : (metrics.novel_behaviors_detected > 10 ? 'High' : metrics.novel_behaviors_detected > 5 ? 'Moderate' : 'Low')}
          </p>
        </div>
      </div>
    </div>
  );
};

export default AgentActivity;