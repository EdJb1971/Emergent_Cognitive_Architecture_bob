// Dashboard Types - Scientific Metrics for ECA Evaluation

export interface DashboardMetrics {
  // Real-time metrics
  timestamp: string;
  summary: {
    total_events: number;
    events_per_minute: number;
    active_users: number;
    avg_processing_time_ms: number;
  };

  // Agent metrics
  agent_metrics: {
    activation_frequencies: Record<string, number>;
    total_activations: number;
  };

  // Memory Performance
  memory_metrics: {
    tier_performance: Record<string, {
      hit_rate: number;
      retrieval_time_ms: number;
      access_count: number;
    }>;
  };

  // Learning metrics
  learning_metrics: {
    skill_performance: Record<string, {
      current_avg: number;
      trend: number;
      total_samples: number;
    }>;
  };

  // Conflict resolution
  conflict_metrics: {
    resolution_stats: Record<string, number>;
    coherence_improvement: number;
    total_conflicts: number;
  };

  // User experience
  user_experience: {
    avg_satisfaction: number;
    avg_engagement: number;
  };

  // Recent events
  recent_events: Array<{
    type: string;
    timestamp: string;
    cycle_id?: string;
    user_id?: string;
    summary: string;
  }>;
}

export interface LearningMetrics {
  skill_improvement_rates: Record<string, number[]>;
  recent_learning_events: LearningEvent[];
  average_skill_performance: number;
  total_skills_tracked: number;
}

export interface EmergenceMetrics {
  agent_activation_counts: Record<string, number>;
  conflict_resolution_rate: number;
  average_coherence_score: number;
  novel_behaviors_detected: number;
}

export interface PerformanceMetrics {
  average_response_time: number;
  error_rate: number;
  throughput_cycles_per_minute: number;
  memory_usage_mb: number;
}

export interface LearningEvent {
  timestamp: string;
  skill_category: string;
  outcome_score: number;
  confidence_score: number;
  success: boolean;
  error_type?: string;
  improvement_rate: number;
}

export interface HistoricalData {
  time_range: string;
  data_points: DataPoint[];
  trends: TrendAnalysis;
}

export interface DataPoint {
  timestamp: string;
  metric_name: string;
  value: number;
  metadata?: Record<string, any>;
}

export interface TrendAnalysis {
  direction: 'improving' | 'declining' | 'stable' | 'volatile';
  slope: number;
  confidence: number;
  significant_change: boolean;
}

export interface DashboardConfig {
  showRealTime: boolean;
  showHistorical: boolean;
  timeRange: '1h' | '6h' | '24h' | '7d';
  refreshInterval: number; // seconds
  selectedMetrics: string[];
}

// Mock data for development and testing
export const createMockDashboardMetrics = (): DashboardMetrics => ({
  timestamp: new Date().toISOString(),
  summary: {
    total_events: 1247,
    events_per_minute: 8.5,
    active_users: 3,
    avg_processing_time_ms: 2300
  },

  agent_metrics: {
    activation_frequencies: {
      perception: 1247,
      emotional: 892,
      memory: 756,
      planning: 634,
      creative: 423,
      critic: 892,
      discovery: 234,
      web_browsing: 145
    },
    total_activations: 5223
  },

  memory_metrics: {
    tier_performance: {
      stm: {
        hit_rate: 0.84,
        retrieval_time_ms: 45.2,
        access_count: 156
      },
      ltm: {
        hit_rate: 0.76,
        retrieval_time_ms: 120.5,
        access_count: 89
      }
    }
  },

  learning_metrics: {
    skill_performance: {
      technical_explanation: {
        current_avg: 0.78,
        trend: 0.03,
        total_samples: 25
      },
      emotional_support: {
        current_avg: 0.71,
        trend: 0.02,
        total_samples: 20
      },
      problem_solving: {
        current_avg: 0.79,
        trend: 0.01,
        total_samples: 30
      },
      creative_brainstorming: {
        current_avg: 0.74,
        trend: 0.02,
        total_samples: 18
      }
    }
  },

  conflict_metrics: {
    resolution_stats: {
      resolved: 87,
      escalated: 13
    },
    coherence_improvement: 0.82,
    total_conflicts: 100
  },

  user_experience: {
    avg_satisfaction: 0.85,
    avg_engagement: 0.72
  },

  recent_events: [
    {
      type: 'cognitive_cycle',
      timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
      cycle_id: 'cycle-123',
      user_id: 'user-456',
      summary: 'Cognitive cycle completed successfully'
    },
    {
      type: 'agent_activation',
      timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
      cycle_id: 'cycle-122',
      user_id: 'user-456',
      summary: 'Perception agent activated for image analysis'
    },
    {
      type: 'learning_event',
      timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      cycle_id: 'cycle-121',
      user_id: 'user-456',
      summary: 'Skill improvement detected in technical explanation'
    }
  ]
});