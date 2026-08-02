export type AutonomousTaskType =
  | 'sleep'
  | 'reflection'
  | 'discovery'
  | 'curiosity'
  | 'self_assessment'
  | 'proactive_engagement'
  | 'summary_update'
  | 'stm_flush';

export type AutonomousTaskStatus =
  | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'rejected' | 'duplicate';

export interface AutonomousPolicy {
  task_type: AutonomousTaskType;
  enabled: boolean;
  cooldown_seconds: number;
  timeout_seconds: number;
  max_retries: number;
  max_per_hour: number;
  max_concurrent_per_user: number;
  provider_policy: 'local_only' | 'no_inference';
  cancel_on_user_activity: boolean;
  description: string;
}

export interface AutonomousRuntime {
  master_enabled: boolean;
  max_concurrent_global: number;
  active_count: number;
  queued_count: number;
  policies: Record<AutonomousTaskType, AutonomousPolicy>;
  changed_at: string;
  persistence: string;
}

export interface AutonomousTask {
  request: {
    task_id: string;
    user_id: string;
    task_type: AutonomousTaskType;
    trigger_reason: string;
    signals: Record<string, unknown>;
    payload: Record<string, unknown>;
    deduplication_key: string;
    provider_policy: string;
    priority: number;
    created_at: string;
  };
  status: AutonomousTaskStatus;
  attempt: number;
  max_attempts: number;
  started_at?: string;
  completed_at?: string;
  result: Record<string, unknown>;
  error?: string;
  rejection_reason?: string;
}

export interface AutonomousEvent {
  sequence: number;
  event_id: string;
  event_type: string;
  task_id?: string;
  task_type?: AutonomousTaskType;
  created_at: string;
  payload: Record<string, unknown>;
  event_hash: string;
}

