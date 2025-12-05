import React from 'react';
import * as FaIcons from 'react-icons/fa';

interface LearningMetrics {
  skill_improvement_rates: Record<string, number[]>;
  recent_learning_events: LearningEvent[];
  average_skill_performance: number;
  total_skills_tracked: number;
}

interface LearningEvent {
  timestamp: string;
  skill_category: string;
  outcome_score: number;
  confidence_score: number;
  success: boolean;
  error_type?: string;
  improvement_rate: number;
}

interface LearningProgressProps {
  metrics: LearningMetrics;
}

const LearningProgress: React.FC<LearningProgressProps> = ({ metrics }) => {
  const getSkillColor = (skill: string) => {
    const colors = [
      'bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500',
      'bg-red-500', 'bg-indigo-500', 'bg-pink-500', 'bg-teal-500'
    ];
    return colors[skill.length % colors.length];
  };

  const formatSkillName = (skill: string) => {
    return skill.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center">
        <FaIcons.FaGraduationCap className="mr-2 text-blue-600" />
        Learning & Skill Development
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Skill Performance Overview */}
        <div>
          <h4 className="text-md font-medium mb-3 text-gray-700 dark:text-gray-300">
            Skill Performance Overview
          </h4>
          <div className="space-y-3">
            {Object.entries(metrics.skill_improvement_rates).slice(0, 5).map(([skill, rates]) => {
              const latestRate = rates[rates.length - 1] || 0;
              const previousRate = rates[rates.length - 2] || 0;
              const improvement = latestRate - previousRate;

              return (
                <div key={skill} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${getSkillColor(skill)}`}></div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {formatSkillName(skill)}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {(latestRate * 100).toFixed(1)}%
                    </span>
                    {improvement !== 0 && (
                      <span className={`text-xs px-2 py-1 rounded ${
                        improvement > 0
                          ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                          : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                      }`}>
                        {improvement > 0 ? '+' : ''}{(improvement * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Recent Learning Events */}
        <div>
          <h4 className="text-md font-medium mb-3 text-gray-700 dark:text-gray-300">
            Recent Learning Events
          </h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {metrics.recent_learning_events.slice(0, 8).map((event, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded text-sm">
                <div className="flex items-center space-x-2">
                  {event.success ? (
                    <FaIcons.FaTrophy className="text-green-500 text-xs" />
                  ) : (
                    React.createElement(FaIcons.FaChartLine, { className: "text-red-500 text-xs" })
                  )}
                  <span className="text-gray-900 dark:text-white">
                    {formatSkillName(event.skill_category)}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-xs ${
                    event.success
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {(event.outcome_score * 100).toFixed(0)}%
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-200 dark:border-gray-600">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">
            {metrics.total_skills_tracked}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Skills Tracked
          </div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">
            {(metrics.average_skill_performance * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Average Performance
          </div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">
            {metrics.recent_learning_events.filter(e => e.success).length}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Successful Events (Last 24h)
          </div>
        </div>
      </div>
    </div>
  );
};

export default LearningProgress;