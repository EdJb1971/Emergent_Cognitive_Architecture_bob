export interface IdentityProfile {
  schema_version: number;
  assistant_name: string;
  user_name: string | null;
  assistant_aliases: string[];
  revision: number;
  updated_at: string;
}

export interface CleanStartStatus {
  pending_restart: boolean;
  preserve_identity: boolean;
  requested_at: string | null;
  message: string;
}
