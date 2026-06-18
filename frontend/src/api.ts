import type {
  AgentStatus,
  Conversation,
  ConversationMessage,
  ContextPack,
  Diagnostics,
  EvalSuite,
  Health,
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

export async function streamChat(
  message: string,
  handlers: {
    onEvent: (event: TimelineEvent) => void;
    onAgents: (agents: string[]) => void;
    onChunk: (content: string) => void;
    onDone: (payload: { answer?: string; agents_used?: string[] }) => void;
  },
  conversationId?: string | null
) {
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation_id: conversationId || null })
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
        continue;
      }
      if (!line.startsWith("data: ")) continue;
      const data = JSON.parse(line.slice(6));
      handlers.onEvent({ event: eventType, data });

      if (eventType === "route" && Array.isArray(data.agents)) {
        handlers.onAgents(data.agents);
      }
      if (eventType === "chunk" || eventType === "token") {
        handlers.onChunk(data.content || data.text || "");
      }
      if (eventType === "done") {
        handlers.onDone(data);
      }
    }
  }
}
