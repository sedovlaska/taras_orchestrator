export type Health = {
  status: string;
  framework: string;
  model: string;
  ollama_host: string;
  members: string[];
  policy: {
    mode: string;
    allowed_risks: string;
    denied: string;
  };
};

export type ModelsResponse = {
  models: string[];
  default_model: string;
  reachable: boolean;
};

export type AgentStatus = {
  name: string;
  online: boolean;
  framework?: string;
};

export type RunSummary = {
  total_runs?: number;
  by_status?: Array<{ status: string; count: number }>;
  by_agent?: Array<{ agent: string; count: number }>;
  approvals?: Array<{ status: string; count: number }>;
  recent_errors?: Array<Record<string, unknown>>;
};

export type RunListItem = {
  id: string;
  message: string;
  status: string;
  intent: string;
  agents: string[];
  created_at: string;
};

export type TimelineEvent = {
  event: string;
  data: Record<string, any>;
};

export type DiagnosticCheck = {
  id: string;
  status: "ok" | "warn" | "error" | string;
  title: string;
  detail: string;
};

export type Diagnostics = {
  status: string;
  checks: DiagnosticCheck[];
  counts: Record<string, number>;
};

export type EvalResult = {
  id: string;
  passed: boolean;
  reason?: string;
  failures?: string[];
};

export type EvalSuite = {
  pass_rate: number;
  failed: number;
  results: EvalResult[];
};

export type WorkspaceFile = {
  path: string;
  size_bytes: number;
  modified_at?: string;
  line_count?: number;
  matches?: number;
  preview?: string;
};

export type WorkspaceRead = {
  path: string;
  content: string;
  truncated: boolean;
  size_bytes: number;
};

export type ContextPack = {
  id: string;
  name: string;
  description?: string;
  paths?: string[];
  query?: string;
  search_limit?: number;
  max_chars?: number;
  created_at?: string;
};

export type RunbookVariable = {
  name: string;
  label: string;
  description?: string;
  default?: string;
  required?: boolean;
  multiline?: boolean;
};

export type Runbook = {
  id: string;
  title: string;
  category: string;
  description: string;
  variables: RunbookVariable[];
};

export type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant" | "error";
  content: string;
  agents?: string[];
};

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ConversationMessage = {
  id: number;
  conversation_id: string;
  created_at: string;
  role: string;
  content: string;
};
