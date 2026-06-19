import type {
  AgentStatus,
  ApprovalData,
  Conversation,
  ConversationMessage,
  ContextPack,
  Diagnostics,
  EvalSuite,
  Health,
  ModelSettings,
  ModelsResponse,
  RunListItem,
  RunSummary,
  Runbook,
  TimelineEvent,
  WorkspaceFile,
  WorkspaceRead
} from "./types";

async function jsonRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {})
    }
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Request failed: ${response.status}`);
  }
  return data as T;
}

export const api = {
  health: () => jsonRequest<Health>("/health"),
  models: () => jsonRequest<ModelsResponse>("/models"),
  modelSettings: () => jsonRequest<{ settings: ModelSettings }>("/settings/model"),
  updateModelSettings: (payload: {
    provider: string;
    base_url: string;
    model: string;
    api_key?: string | null;
    keep_existing_api_key?: boolean;
  }) =>
    jsonRequest<{ settings: ModelSettings }>("/settings/model", {
      method: "PUT",
      body: JSON.stringify(payload)
    }),
  agents: () => jsonRequest<{ agents: AgentStatus[] }>("/agents/status"),
  runs: () => jsonRequest<{ runs: RunListItem[] }>("/runs?limit=30"),
  run: (runId: string) => jsonRequest<{ run: RunListItem }>(`/runs/${encodeURIComponent(runId)}`),
  runEvents: (runId: string) =>
    jsonRequest<{ events: TimelineEvent[] }>(`/runs/${encodeURIComponent(runId)}/events`),
  runTrace: (runId: string) =>
    jsonRequest<{ trace: { markdown: string } }>(`/runs/${encodeURIComponent(runId)}/trace`),
  summary: () => jsonRequest<{ summary: RunSummary }>("/runs/summary"),
  diagnostics: () => jsonRequest<{ diagnostics: Diagnostics }>("/diagnostics"),
  evals: () => jsonRequest<{ suite: EvalSuite }>("/evals/run", { method: "POST" }),
  workspaceFiles: () => jsonRequest<{ files: WorkspaceFile[] }>("/workspace/files?limit=120"),
  workspaceFile: (path: string) =>
    jsonRequest<{ file: WorkspaceRead }>(`/workspace/file?path=${encodeURIComponent(path)}`),
  workspaceSearch: (query: string) =>
    jsonRequest<{ results: WorkspaceFile[] }>("/workspace/search", {
      method: "POST",
      body: JSON.stringify({ query, limit: 80 })
    }),
  contextBundle: (payload: { paths?: string[]; query?: string }) =>
    jsonRequest<{ bundle: { prompt_context: string } }>("/context/bundle", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  contextPacks: () => jsonRequest<{ packs: ContextPack[] }>("/context/packs?limit=30"),
  createContextPack: (payload: { name: string; paths?: string[]; query?: string }) =>
    jsonRequest<{ pack: ContextPack }>("/context/packs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  applyContextPack: (packId: string) =>
    jsonRequest<{ pack: ContextPack; bundle: { prompt_context: string } }>(
      `/context/packs/${encodeURIComponent(packId)}/bundle`,
      { method: "POST" }
    ),
  deleteContextPack: (packId: string) =>
    jsonRequest<{ deleted: boolean }>(`/context/packs/${encodeURIComponent(packId)}`, {
      method: "DELETE"
    }),
  runbooks: () => jsonRequest<{ runbooks: Runbook[] }>("/runbooks"),
  renderRunbook: (runbookId: string, values: Record<string, string>) =>
    jsonRequest<{ prompt: string }>(`/runbooks/${encodeURIComponent(runbookId)}/render`, {
      method: "POST",
      body: JSON.stringify({ values })
    }),
  resolveApproval: (approvalId: string, action: "approve" | "deny") =>
    jsonRequest<{ approval: { tool_id: string; status: string } }>(
      `/approvals/${encodeURIComponent(approvalId)}/${action}`,
      { method: "POST" }
    ),
  approvals: (status?: string) =>
    jsonRequest<{ approvals: ApprovalData[] }>(
      `/approvals${status ? `?status=${encodeURIComponent(status)}` : ""}`
    ),
  resumeRun: (runId: string) =>
    jsonRequest<{ answer: string; agents_used: string[]; events: TimelineEvent[] }>(
      `/runs/${encodeURIComponent(runId)}/resume`,
      { method: "POST" }
    ),
  conversations: () => jsonRequest<{ conversations: Conversation[] }>("/conversations?limit=50"),
  conversation: (id: string) =>
    jsonRequest<{ conversation: Conversation; messages: ConversationMessage[] }>(
      `/conversations/${encodeURIComponent(id)}`
    ),
  createConversation: (title?: string) =>
    jsonRequest<{ conversation: Conversation }>("/conversations", {
      method: "POST",
      body: JSON.stringify({ title: title || null })
    }),
  deleteConversation: (id: string) =>
    jsonRequest<{ deleted: boolean }>(`/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE"
    })
};
