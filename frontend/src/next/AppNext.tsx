import { useChat } from "@ai-sdk/react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable
} from "@tanstack/react-table";
import type { ColumnDef, SortingState } from "@tanstack/react-table";
import { DefaultChatTransport } from "ai";
import type { UIMessage } from "ai";
import {
  Activity,
  ArrowDown,
  ArrowUp,
  BookOpen,
  Bot,
  CheckCircle2,
  ChevronsUpDown,
  Clock3,
  Cpu,
  FileSearch,
  GitBranch,
  HeartPulse,
  History,
  Loader2,
  MessageSquarePlus,
  Package,
  Play,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  Trash2,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";

import {
  Conversation as AiConversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton
} from "@/components/ai-elements/conversation";
import { Loader } from "@/components/ai-elements/loader";
import {
  Message,
  MessageContent,
  MessageResponse
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSelect,
  PromptInputSelectContent,
  PromptInputSelectItem,
  PromptInputSelectTrigger,
  PromptInputSelectValue,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage
} from "@/components/ai-elements/prompt-input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import {
  CodeBlock,
  CodeBlockBody,
  CodeBlockContent,
  CodeBlockCopyButton,
  CodeBlockFilename,
  CodeBlockHeader,
  CodeBlockItem,
  type BundledLanguage
} from "@/components/kibo-ui/code-block";
import {
  TreeExpander,
  TreeIcon,
  TreeLabel,
  TreeNode,
  TreeNodeContent,
  TreeNodeTrigger,
  TreeProvider,
  TreeView
} from "@/components/kibo-ui/tree";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/api";
import { notifyError, notifySuccess } from "@/lib/notify";
import type {
  ApprovalData,
  Conversation,
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
  TraceData,
  WorkspaceFile
} from "@/types";

import { ApprovalCard } from "./ApprovalCard";
import { RunnerTimeline } from "./RunnerTimeline";

const CHAT_API = "/chat/stream";
const DEFAULT_INSPECTOR_WIDTH = 560;
const MIN_INSPECTOR_WIDTH = 460;
const MAX_INSPECTOR_WIDTH = 860;

type WorkspaceTreeNode = {
  id: string;
  name: string;
  path: string;
  children: WorkspaceTreeNode[];
  file?: WorkspaceFile;
};

function lastUserText(messages: UIMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "user") continue;
    return message.parts
      .filter((part): part is { type: "text"; text: string } => part.type === "text")
      .map((part) => part.text)
      .join("");
  }
  return "";
}

function eventToTrace(event: TimelineEvent): TraceData {
  return { event: event.event, ...event.data };
}

function messagesFromConversation(items: Array<{ id: number; role: string; content: string }>): UIMessage[] {
  return items
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map(
      (item) =>
        ({
          id: `conversation-${item.id}`,
          role: item.role,
          parts: [{ type: "text", text: item.content }]
        }) as UIMessage
    );
}

function buildWorkspaceTree(files: WorkspaceFile[]): WorkspaceTreeNode[] {
  const roots: WorkspaceTreeNode[] = [];

  for (const file of files) {
    const parts = file.path.split(/[\\/]+/).filter(Boolean);
    let siblings = roots;
    let currentPath = "";

    parts.forEach((part, index) => {
      currentPath = currentPath ? `${currentPath}/${part}` : part;
      const isFile = index === parts.length - 1;
      let node = siblings.find((candidate) => candidate.name === part);

      if (!node) {
        node = {
          id: `${isFile ? "file" : "folder"}:${currentPath}`,
          name: part,
          path: currentPath,
          children: []
        };
        siblings.push(node);
      }

      if (isFile) {
        node.id = `file:${file.path}`;
        node.path = file.path;
        node.file = file;
      }

      siblings = node.children;
    });
  }

  const sortNodes = (nodes: WorkspaceTreeNode[]): WorkspaceTreeNode[] =>
    nodes
      .sort((left, right) => {
        const leftIsFolder = left.children.length > 0;
        const rightIsFolder = right.children.length > 0;
        if (leftIsFolder !== rightIsFolder) {
          return leftIsFolder ? -1 : 1;
        }
        return left.name.localeCompare(right.name);
      })
      .map((node) => ({ ...node, children: sortNodes(node.children) }));

  return sortNodes(roots);
}

function collectFolderIds(nodes: WorkspaceTreeNode[]): string[] {
  return nodes.flatMap((node) => [
    ...(node.children.length ? [node.id] : []),
    ...collectFolderIds(node.children)
  ]);
}

function detectLanguage(path: string): BundledLanguage | undefined {
  const name = path.toLowerCase();
  const extension = name.split(".").pop();

  if (name === "dockerfile") return "docker";
  if (name.endsWith(".md")) return "markdown";
  if (name.endsWith(".json")) return "json";
  if (name.endsWith(".toml")) return "toml";
  if (name.endsWith(".yml") || name.endsWith(".yaml")) return "yaml";
  if (name.endsWith(".tsx")) return "tsx";
  if (name.endsWith(".ts")) return "typescript";
  if (name.endsWith(".jsx")) return "jsx";
  if (name.endsWith(".js")) return "javascript";
  if (name.endsWith(".py")) return "python";
  if (name.endsWith(".css")) return "css";
  if (name.endsWith(".html")) return "html";
  if (extension === "env" || name.includes(".env")) return "dotenv";
  return undefined;
}

function WorkspaceTreeItems({
  nodes,
  level = 0,
  parentPath = []
}: {
  nodes: WorkspaceTreeNode[];
  level?: number;
  parentPath?: boolean[];
}) {
  return (
    <>
      {nodes.map((node, index) => {
        const hasChildren = node.children.length > 0;
        const isLast = index === nodes.length - 1;

        return (
          <TreeNode
            key={node.id}
            nodeId={node.id}
            level={level}
            isLast={isLast}
            parentPath={parentPath}
          >
            <TreeNodeTrigger className="py-1.5">
              <TreeExpander hasChildren={hasChildren} />
              <TreeIcon hasChildren={hasChildren} />
              <TreeLabel>{node.name}</TreeLabel>
              {node.file?.matches ? (
                <Badge variant="outline" className="ml-2 h-5 px-1.5 text-[10px]">
                  {node.file.matches}
                </Badge>
              ) : null}
            </TreeNodeTrigger>
            <TreeNodeContent hasChildren={hasChildren}>
              <WorkspaceTreeItems
                nodes={node.children}
                level={level + 1}
                parentPath={[...parentPath, isLast]}
              />
            </TreeNodeContent>
          </TreeNode>
        );
      })}
    </>
  );
}

function Stat({
  label,
  value,
  tone = "text-foreground"
}: {
  label: string;
  value: string | number;
  tone?: string;
}) {
  return (
    <div className="rounded-md border bg-card px-3 py-2">
      <div className={`text-lg font-semibold tabular-nums ${tone}`}>{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

function EmptyLine({ children }: { children: string }) {
  return <p className="py-3 text-sm text-muted-foreground">{children}</p>;
}

function RunsTable({
  runs,
  onSelect
}: {
  runs: RunListItem[];
  onSelect: (runId: string) => void;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const columns = useMemo<ColumnDef<RunListItem>[]>(
    () => [
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => (
          <Badge variant={row.original.status === "completed" ? "secondary" : "destructive"}>
            <span className="whitespace-nowrap">{row.original.status}</span>
          </Badge>
        )
      },
      {
        accessorKey: "message",
        header: "Message",
        cell: ({ row }) => (
          <button
            type="button"
            className="line-clamp-2 text-left text-sm font-medium hover:underline"
            onClick={() => onSelect(row.original.id)}
          >
            {row.original.message}
          </button>
        )
      },
      {
        accessorKey: "intent",
        header: "Intent",
        cell: ({ row }) => <span className="text-xs text-muted-foreground">{row.original.intent}</span>
      },
      {
        accessorKey: "latency_ms",
        header: "Latency",
        cell: ({ row }) => (
          <span className="text-xs tabular-nums text-muted-foreground">
            {row.original.latency_ms ? `${row.original.latency_ms}ms` : "-"}
          </span>
        )
      }
    ],
    [onSelect]
  );
  const table = useReactTable({
    data: runs,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel()
  });

  if (runs.length === 0) {
    return <EmptyLine>No recorded runs.</EmptyLine>;
  }

  return (
    <div className="overflow-hidden rounded-md border bg-card">
      <table className="w-full table-auto text-sm">
        <thead className="bg-muted/60 text-xs text-muted-foreground">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  className="px-2 py-2 text-left font-medium"
                  style={{ width: header.column.id === "message" ? "48%" : undefined }}
                >
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 hover:text-foreground"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {header.column.getIsSorted() === "asc" ? (
                      <ArrowUp className="size-3" />
                    ) : header.column.getIsSorted() === "desc" ? (
                      <ArrowDown className="size-3" />
                    ) : (
                      <ChevronsUpDown className="size-3 opacity-50" />
                    )}
                  </button>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-t hover:bg-accent/40">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-2 py-2 align-top">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AppNext() {
  const [input, setInput] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState<string | null>(null);
  const [modelsReachable, setModelsReachable] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [modelSettings, setModelSettings] = useState<ModelSettings | null>(null);
  const [modelForm, setModelForm] = useState({
    provider: "ollama",
    model: "",
    base_url: "",
    api_key: ""
  });
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [summary, setSummary] = useState<RunSummary>({});
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [evalSuite, setEvalSuite] = useState<EvalSuite | null>(null);
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [fileSearch, setFileSearch] = useState("");
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [preview, setPreview] = useState("Select a workspace file to preview it here.");
  const [contextPacks, setContextPacks] = useState<ContextPack[]>([]);
  const [contextPackName, setContextPackName] = useState("");
  const [runbooks, setRunbooks] = useState<Runbook[]>([]);
  const [selectedRunbookId, setSelectedRunbookId] = useState<string | null>(null);
  const [runbookValues, setRunbookValues] = useState<Record<string, string>>({});
  const [approvalsQueue, setApprovalsQueue] = useState<ApprovalData[]>([]);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [modelSettingsSaving, setModelSettingsSaving] = useState(false);
  const [inspectorTab, setInspectorTab] = useState("timeline");
  const [inspectorWidth, setInspectorWidth] = useState(DEFAULT_INSPECTOR_WIDTH);

  const [traces, setTraces] = useState<TraceData[]>([]);
  const [approvals, setApprovals] = useState<ApprovalData[]>([]);
  const [resolved, setResolved] = useState<Record<string, "approve" | "deny">>({});

  const modelRef = useRef<string | null>(null);
  const conversationRef = useRef<string | null>(null);
  modelRef.current = selectedModel;
  conversationRef.current = conversationId;

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: CHAT_API,
        prepareSendMessagesRequest: ({ messages }) => ({
          body: {
            message: lastUserText(messages),
            conversation_id: conversationRef.current,
            ...(modelRef.current ? { model: modelRef.current } : {})
          }
        })
      }),
    []
  );

  const { messages, sendMessage, status, setMessages, error } = useChat({
    transport,
    onData: (part) => {
      if (part.type === "data-trace") {
        setTraces((current) => [...current, part.data as TraceData]);
      } else if (part.type === "data-approval") {
        const data = part.data as ApprovalData;
        setApprovals((current) =>
          current.some((item) => item.approval_id === data.approval_id)
            ? current
            : [...current, data]
        );
      }
    },
    onError: (err) => notifyError(err, "Chat failed")
  });

  const busy = status === "submitted" || status === "streaming";
  const selectedRunbook = runbooks.find((runbook) => runbook.id === selectedRunbookId) || null;
  const runbookGroups = useMemo(
    () =>
      Array.from(
        runbooks.reduce((groups, runbook) => {
          const category = runbook.category || "Runbooks";
          groups.set(category, [...(groups.get(category) || []), runbook]);
          return groups;
        }, new Map<string, Runbook[]>())
      ).map(([category, items]) => ({ category, items })),
    [runbooks]
  );
  const runbookTreeKey = runbookGroups.map((group) => group.category).join("|");
  const selectedRunbookTreeId = selectedRunbookId ? `runbook:${selectedRunbookId}` : undefined;
  const workspaceTree = useMemo(() => buildWorkspaceTree(files), [files]);
  const workspaceTreeKey = files.map((file) => file.path).join("|");
  const workspaceDefaultExpandedIds = useMemo(() => collectFolderIds(workspaceTree), [workspaceTree]);
  const selectedFileTreeId = selectedPath ? `file:${selectedPath}` : undefined;
  const previewData = useMemo(
    () => [
      {
        language: selectedPath || "preview",
        filename: selectedPath || "Workspace preview",
        code: preview
      }
    ],
    [preview, selectedPath]
  );
  const hasModels = modelsReachable && models.length > 0;
  const modelOptions = hasModels ? models : defaultModel ? [defaultModel] : [];
  const lastMessageIsAssistant = messages.at(-1)?.role === "assistant";
  const errorRuns = (summary.by_status || [])
    .filter((row) => ["failed", "denied"].includes(row.status))
    .reduce((total, row) => total + row.count, 0);

  const loadModels = useCallback(async () => {
    try {
      const data: ModelsResponse = await api.models();
      setModels(data.models || []);
      setDefaultModel(data.default_model || null);
      setModelsReachable(data.reachable);
      setSelectedModel((current) =>
        current && (data.models || []).includes(current) ? current : data.default_model || null
      );
    } catch (err) {
      notifyError(err, "Models failed");
    }
  }, []);

  const loadModelSettings = useCallback(async () => {
    try {
      const data = await api.modelSettings();
      setModelSettings(data.settings);
      setModelForm({
        provider: data.settings.provider || "ollama",
        model: data.settings.model || "",
        base_url: data.settings.base_url || "",
        api_key: ""
      });
    } catch (err) {
      notifyError(err, "Model settings failed");
    }
  }, []);

  const refreshDashboard = useCallback(async () => {
    setDashboardLoading(true);
    try {
      let failed = 0;
      const loadPanel = async <T,>(request: Promise<T>, apply: (value: T) => void) => {
        try {
          const value = await request;
          apply(value);
        } catch {
          failed += 1;
        }
      };
      await Promise.allSettled([
        loadPanel(api.health(), setHealth),
        loadPanel(api.runs(), (value) => setRuns(value.runs || [])),
        loadPanel(api.summary(), (value) => setSummary(value.summary || {})),
        loadPanel(api.diagnostics(), (value) => setDiagnostics(value.diagnostics)),
        loadPanel(api.workspaceFiles(), (value) => setFiles(value.files || [])),
        loadPanel(api.contextPacks(), (value) => setContextPacks(value.packs || [])),
        loadPanel(api.runbooks(), (value) => {
          setRunbooks(value.runbooks || []);
          if (!selectedRunbookId && value.runbooks?.length) {
            setSelectedRunbookId(value.runbooks[0].id);
          }
        }),
        loadPanel(api.conversations(), (value) => setConversations(value.conversations || [])),
        loadPanel(api.approvals("pending"), (value) => setApprovalsQueue(value.approvals || []))
      ]);
      if (failed > 0) {
        notifyError(new Error(`${failed} dashboard request${failed === 1 ? "" : "s"} failed`), "Dashboard refresh incomplete");
      }
    } catch (err) {
      notifyError(err, "Dashboard refresh failed");
    } finally {
      setDashboardLoading(false);
    }
  }, [selectedRunbookId]);

  useEffect(() => {
    loadModels();
    loadModelSettings();
    refreshDashboard();
  }, [loadModels, loadModelSettings, refreshDashboard]);

  useEffect(() => {
    if (!selectedRunbook) return;
    setRunbookValues(
      Object.fromEntries(
        selectedRunbook.variables.map((variable) => [variable.name, variable.default || ""])
      )
    );
  }, [selectedRunbook]);

  function addToPrompt(text: string) {
    setInput((current) => (current.trim() ? `${current.trim()}\n\n${text}` : text));
  }

  async function handleSubmit(message: PromptInputMessage) {
    const text = message.text.trim();
    if (!text || busy) return;

    let convo = conversationId;
    if (!convo) {
      try {
        const created = await api.createConversation(text.slice(0, 60));
        convo = created.conversation.id;
        setConversationId(convo);
        conversationRef.current = convo;
        setConversations((current) => [created.conversation, ...current]);
      } catch (err) {
        notifyError(err, "New conversation failed");
        return;
      }
    }

    setTraces([]);
    setApprovals([]);
    setResolved({});
    setInput("");
    sendMessage({ text });
  }

  async function selectConversation(id: string) {
    try {
      const data = await api.conversation(id);
      setConversationId(id);
      conversationRef.current = id;
      setMessages(messagesFromConversation(data.messages));
      setTraces([]);
      setApprovals([]);
      setResolved({});
    } catch (err) {
      notifyError(err, "Conversation load failed");
    }
  }

  async function deleteConversation(id: string) {
    try {
      await api.deleteConversation(id);
      setConversations((current) => current.filter((conversation) => conversation.id !== id));
      if (conversationId === id) {
        newConversation();
      }
    } catch (err) {
      notifyError(err, "Conversation delete failed");
    }
  }

  function newConversation() {
    setConversationId(null);
    conversationRef.current = null;
    setMessages([]);
    setTraces([]);
    setApprovals([]);
    setResolved({});
    setInput("");
  }

  async function resolveApproval(approval: ApprovalData, action: "approve" | "deny") {
    try {
      await api.resolveApproval(approval.approval_id, action);
      setResolved((current) => ({ ...current, [approval.approval_id]: action }));
      setApprovalsQueue((current) =>
        current.filter((item) => item.approval_id !== approval.approval_id)
      );
      setTraces((current) => [
        ...current,
        { event: "approval_resolved", tool_id: approval.tool_id, status: action }
      ]);
      if (action === "approve" && approval.run_id) {
        const data = await api.resumeRun(approval.run_id);
        for (const event of data.events || []) {
          setTraces((current) => [...current, eventToTrace(event)]);
        }
        setMessages((current) => [
          ...current,
          {
            id: `resume-${approval.run_id}`,
            role: "assistant",
            parts: [{ type: "text", text: data.answer || "Done." }]
          } as UIMessage
        ]);
      }
      refreshDashboard();
    } catch (err) {
      notifyError(err, "Approval failed");
    }
  }

  async function selectRun(runId: string) {
    try {
      const data = await api.runEvents(runId);
      setTraces((data.events || []).map(eventToTrace));
    } catch (err) {
      notifyError(err, "Run events failed");
    }
  }

  async function selectFile(path: string) {
    try {
      setSelectedPath(path);
      const data = await api.workspaceFile(path);
      setPreview(data.file.content + (data.file.truncated ? "\n[truncated]" : ""));
    } catch (err) {
      notifyError(err, "File preview failed");
    }
  }

  async function searchWorkspace() {
    try {
      if (fileSearch.trim()) {
        const data = await api.workspaceSearch(fileSearch.trim());
        setFiles(data.results || []);
      } else {
        const data = await api.workspaceFiles();
        setFiles(data.files || []);
      }
    } catch (err) {
      notifyError(err, "Workspace search failed");
    }
  }

  async function attachSelectedFile() {
    if (!selectedPath) return;
    try {
      const data = await api.contextBundle({ paths: [selectedPath] });
      addToPrompt(data.bundle.prompt_context);
      notifySuccess("File attached to prompt");
    } catch (err) {
      notifyError(err, "Attach failed");
    }
  }

  async function saveContextPack() {
    if (!contextPackName.trim()) return;
    try {
      const data = await api.createContextPack({
        name: contextPackName.trim(),
        paths: selectedPath ? [selectedPath] : [],
        query: fileSearch.trim() || undefined
      });
      setContextPacks((current) => [data.pack, ...current]);
      setContextPackName("");
      notifySuccess("Context pack saved");
    } catch (err) {
      notifyError(err, "Context pack save failed");
    }
  }

  async function applyContextPack(packId: string) {
    try {
      const data = await api.applyContextPack(packId);
      addToPrompt(data.bundle.prompt_context);
      notifySuccess("Context pack attached");
    } catch (err) {
      notifyError(err, "Context pack failed");
    }
  }

  async function deleteContextPack(packId: string) {
    try {
      await api.deleteContextPack(packId);
      setContextPacks((current) => current.filter((pack) => pack.id !== packId));
    } catch (err) {
      notifyError(err, "Context pack delete failed");
    }
  }

  const startInspectorResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = inspectorWidth;
    const onPointerMove = (moveEvent: PointerEvent) => {
      const viewportMax = Math.max(MIN_INSPECTOR_WIDTH, window.innerWidth - 560);
      const maxWidth = Math.min(MAX_INSPECTOR_WIDTH, viewportMax);
      const nextWidth = startWidth - (moveEvent.clientX - startX);
      setInspectorWidth(Math.min(maxWidth, Math.max(MIN_INSPECTOR_WIDTH, nextWidth)));
    };
    const onPointerUp = () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }, [inspectorWidth]);

  async function renderRunbook(runNow = false) {
    if (!selectedRunbook) return;
    try {
      const data = await api.renderRunbook(selectedRunbook.id, runbookValues);
      if (runNow) {
        setInput(data.prompt);
        await handleSubmit({ text: data.prompt, files: [] });
      } else {
        addToPrompt(data.prompt);
      }
    } catch (err) {
      notifyError(err, "Runbook failed");
    }
  }

  async function runEvals() {
    try {
      const data = await api.evals();
      setEvalSuite(data.suite);
    } catch (err) {
      notifyError(err, "Eval run failed");
    }
  }

  async function saveModelSettings() {
    setModelSettingsSaving(true);
    try {
      const data = await api.updateModelSettings({
        provider: modelForm.provider,
        model: modelForm.model,
        base_url: modelForm.base_url,
        api_key: modelForm.api_key.trim() ? modelForm.api_key : null,
        keep_existing_api_key: !modelForm.api_key.trim() && Boolean(modelSettings?.api_key_set)
      });
      setModelSettings(data.settings);
      setModelForm({
        provider: data.settings.provider || "ollama",
        model: data.settings.model || "",
        base_url: data.settings.base_url || "",
        api_key: ""
      });
      notifySuccess("Model backend saved");
      await Promise.all([loadModels(), refreshDashboard()]);
    } catch (err) {
      notifyError(err, "Model settings failed");
    } finally {
      setModelSettingsSaving(false);
    }
  }

  return (
    <div
      className="governance-shell dark grid h-screen grid-cols-1 overflow-hidden bg-background text-foreground md:grid-cols-[280px_minmax(0,1fr)]"
      style={{ "--inspector-width": `${inspectorWidth}px` } as CSSProperties}
    >
      <aside className="hidden border-r bg-muted/20 md:flex md:flex-col">
        <div className="flex h-14 items-center gap-2 border-b px-3">
          <span className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Bot className="size-4" />
          </span>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold">AGNO Team Orchestrator</h1>
            <p className="truncate text-xs text-muted-foreground">Governance console</p>
          </div>
        </div>

        <div className="border-b p-3">
          <Button className="w-full justify-start" size="sm" onClick={newConversation}>
            <MessageSquarePlus className="size-4" />
            New conversation
          </Button>
        </div>

        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-4 p-3">
            <section>
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-xs font-medium uppercase text-muted-foreground">Conversations</h2>
                <Badge variant="secondary">{conversations.length}</Badge>
              </div>
              <div className="space-y-1">
                {conversations.length === 0 ? (
                  <EmptyLine>No saved conversations.</EmptyLine>
                ) : (
                  conversations.map((conversation) => (
                    <div
                      key={conversation.id}
                      className={`group flex items-center gap-2 rounded-md border px-2 py-2 text-sm ${
                        conversation.id === conversationId
                          ? "border-ring bg-accent"
                          : "border-transparent hover:bg-accent/60"
                      }`}
                    >
                      <button
                        type="button"
                        className="min-w-0 flex-1 text-left"
                        onClick={() => selectConversation(conversation.id)}
                      >
                        <span className="block truncate font-medium">{conversation.title}</span>
                        <span className="block truncate text-xs text-muted-foreground">
                          {new Date(conversation.updated_at).toLocaleString()}
                        </span>
                      </button>
                      <button
                        type="button"
                        className="rounded p-1 text-muted-foreground opacity-0 hover:text-destructive group-hover:opacity-100"
                        onClick={() => deleteConversation(conversation.id)}
                        aria-label="Delete conversation"
                      >
                        <Trash2 className="size-3.5" />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </section>
          </div>
        </ScrollArea>
      </aside>

      <main className="flex min-w-0 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between gap-2 border-b px-3 sm:px-4">
          <div className="flex min-w-0 items-center gap-2 text-sm text-muted-foreground">
            <GitBranch className="size-4 shrink-0" />
            <span className="hidden sm:inline">{health?.provider || "provider"}</span>
            <span className="hidden sm:inline">·</span>
            <span className="truncate sm:max-w-[220px]">{health?.model || defaultModel || "model"}</span>
          </div>
          <div className="flex min-w-0 items-center gap-2">
            <PromptInputSelect
              value={selectedModel ?? defaultModel ?? ""}
              onValueChange={setSelectedModel}
              disabled={modelOptions.length === 0}
            >
              <PromptInputSelectTrigger
                className="h-8 max-w-[150px] text-xs sm:max-w-[220px]"
                aria-label="Model for next message"
              >
                <Cpu className="size-3.5 shrink-0" />
                <PromptInputSelectValue placeholder={defaultModel ?? "Model"} />
              </PromptInputSelectTrigger>
              <PromptInputSelectContent>
                {modelOptions.map((model) => (
                  <PromptInputSelectItem key={model} value={model}>
                    {model}
                  </PromptInputSelectItem>
                ))}
              </PromptInputSelectContent>
            </PromptInputSelect>
            {!hasModels ? (
              <Button variant="outline" size="sm" onClick={loadModels} aria-label="Retry models">
                <RefreshCw className="size-4" />
                <span className="hidden sm:inline">Retry models</span>
              </Button>
            ) : null}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInspectorTab("settings")}
              aria-label="Model backend settings"
            >
              <Settings className="size-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={refreshDashboard}
              disabled={dashboardLoading}
              aria-label="Refresh dashboard"
            >
              {dashboardLoading ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
              <span className="hidden sm:inline">Refresh</span>
            </Button>
          </div>
        </header>

        <AiConversation>
          <ConversationContent className="mx-auto w-full max-w-3xl">
            {messages.length === 0 ? (
              <ConversationEmptyState
                icon={<Bot className="size-7" />}
                title="Start a governed run"
                description="Routing, policy, approval, runner, and tool events stay visible while the answer streams."
              />
            ) : (
              messages.map((message, index) => {
                const isLastAssistant = message.role === "assistant" && index === messages.length - 1;
                const text = message.parts
                  .filter((part): part is { type: "text"; text: string } => part.type === "text")
                  .map((part) => part.text)
                  .join("");
                return (
                  <div key={message.id} className="space-y-3">
                    {isLastAssistant && traces.length > 0 ? (
                      <RunnerTimeline events={traces} running={busy} />
                    ) : null}
                    {isLastAssistant && approvals.length > 0
                      ? approvals.map((approval) => (
                          <ApprovalCard
                            key={approval.approval_id}
                            approval={approval}
                            resolved={resolved[approval.approval_id] ?? null}
                            onResolve={resolveApproval}
                          />
                        ))
                      : null}
                    <Message from={message.role}>
                      <MessageContent>
                        {message.role === "assistant" ? (
                          text ? (
                            <MessageResponse>{text}</MessageResponse>
                          ) : busy ? (
                            <span className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Loader size={14} /> Thinking...
                            </span>
                          ) : null
                        ) : (
                          <span className="whitespace-pre-wrap">{text}</span>
                        )}
                      </MessageContent>
                    </Message>
                  </div>
                );
              })
            )}

            {!lastMessageIsAssistant && (traces.length > 0 || approvals.length > 0) ? (
              <div className="space-y-3">
                {traces.length > 0 ? <RunnerTimeline events={traces} running={busy} /> : null}
                {approvals.map((approval) => (
                  <ApprovalCard
                    key={approval.approval_id}
                    approval={approval}
                    resolved={resolved[approval.approval_id] ?? null}
                    onResolve={resolveApproval}
                  />
                ))}
              </div>
            ) : null}

            {error ? (
              <div className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                {error.message}
              </div>
            ) : null}
          </ConversationContent>
          <ConversationScrollButton />
        </AiConversation>

        <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-4">
          <PromptInput onSubmit={handleSubmit}>
            <PromptInputBody>
              <PromptInputTextarea
                value={input}
                onChange={(event) => setInput(event.currentTarget.value)}
                placeholder="Ask the orchestrator..."
              />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <span className="text-xs text-muted-foreground">
                  {conversationId ? "Threaded" : "New thread"}
                </span>
              </PromptInputTools>
              <PromptInputSubmit disabled={busy || !input.trim()} status={status} />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </main>

      <aside className="relative hidden min-w-0 border-l bg-muted/20 lg:flex lg:flex-col">
        <div
          className="absolute -left-1 top-0 z-10 h-full w-2 cursor-col-resize touch-none hover:bg-ring/30"
          onPointerDown={startInspectorResize}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize inspector"
        />
        <Tabs value={inspectorTab} onValueChange={setInspectorTab} className="flex min-h-0 flex-1 flex-col">
          <div className="border-b p-3">
            <TabsList className="grid w-full grid-cols-7">
              <TabsTrigger value="timeline" aria-label="Timeline">
                <Activity className="size-4" />
              </TabsTrigger>
              <TabsTrigger value="history" aria-label="History">
                <History className="size-4" />
              </TabsTrigger>
              <TabsTrigger value="workspace" aria-label="Workspace">
                <FileSearch className="size-4" />
              </TabsTrigger>
              <TabsTrigger value="runbooks" aria-label="Runbooks">
                <BookOpen className="size-4" />
              </TabsTrigger>
              <TabsTrigger value="health" aria-label="Health">
                <HeartPulse className="size-4" />
              </TabsTrigger>
              <TabsTrigger value="settings" aria-label="Settings">
                <Settings className="size-4" />
              </TabsTrigger>
              <TabsTrigger value="approvals" aria-label="Approvals">
                <ShieldCheck className="size-4" />
              </TabsTrigger>
            </TabsList>
          </div>

          <ScrollArea className="min-h-0 flex-1">
            <TabsContent value="timeline" className="m-0 space-y-3 p-3">
              <RunnerTimeline events={traces} running={busy} />
              {traces.length === 0 ? <EmptyLine>No active run trace.</EmptyLine> : null}
            </TabsContent>

            <TabsContent value="history" className="m-0 space-y-3 p-3">
              <div className="grid grid-cols-3 gap-2">
                <Stat label="runs" value={summary.total_runs || 0} />
                <Stat label="errors" value={errorRuns} tone={errorRuns ? "text-destructive" : "text-emerald-500"} />
                <Stat label="avg ms" value={summary.avg_latency_ms ?? "-"} />
              </div>
              <RunsTable runs={runs} onSelect={selectRun} />
            </TabsContent>

            <TabsContent value="workspace" className="m-0 space-y-3 p-3">
              <div className="flex gap-2">
                <Input
                  value={fileSearch}
                  onChange={(event) => setFileSearch(event.currentTarget.value)}
                  placeholder="Search workspace"
                />
                <Button variant="outline" size="icon" onClick={searchWorkspace} aria-label="Search workspace">
                  <Search className="size-4" />
                </Button>
              </div>
              <div className="grid min-h-[520px] grid-cols-[minmax(240px,0.9fr)_minmax(360px,1.5fr)] gap-3">
                <div className="min-h-0 overflow-hidden rounded-md border bg-card">
                  <TreeProvider
                    key={workspaceTreeKey}
                    defaultExpandedIds={workspaceDefaultExpandedIds}
                    selectedIds={selectedFileTreeId ? [selectedFileTreeId] : []}
                    onSelectionChange={(ids) => {
                      const path = ids.find((id) => id.startsWith("file:"))?.slice("file:".length);
                      if (path) {
                        selectFile(path);
                      }
                    }}
                    indent={18}
                    className="h-full"
                  >
                    <TreeView className="max-h-[520px] overflow-auto p-1.5">
                      {workspaceTree.length ? (
                        <WorkspaceTreeItems nodes={workspaceTree} />
                      ) : (
                        <EmptyLine>No workspace files found.</EmptyLine>
                      )}
                    </TreeView>
                  </TreeProvider>
                </div>
                <CodeBlock
                  data={previewData}
                  value={previewData[0].language}
                  className="min-h-[520px] bg-background"
                >
                  <CodeBlockHeader>
                    <CodeBlockFilename value={previewData[0].language}>
                      {previewData[0].filename}
                    </CodeBlockFilename>
                    <CodeBlockCopyButton />
                  </CodeBlockHeader>
                  <CodeBlockBody className="h-[calc(520px-38px)] overflow-auto">
                    {(item) => (
                      <CodeBlockItem key={item.language} value={item.language}>
                        <CodeBlockContent
                          language={detectLanguage(item.filename)}
                          syntaxHighlighting={Boolean(selectedPath)}
                        >
                          {item.code}
                        </CodeBlockContent>
                      </CodeBlockItem>
                    )}
                  </CodeBlockBody>
                </CodeBlock>
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={attachSelectedFile} disabled={!selectedPath}>
                  Attach file
                </Button>
                <Input
                  value={contextPackName}
                  onChange={(event) => setContextPackName(event.currentTarget.value)}
                  placeholder="Pack name"
                />
                <Button size="sm" onClick={saveContextPack}>
                  <Package className="size-4" />
                  Save
                </Button>
              </div>
              <div className="space-y-1">
                {contextPacks.map((pack) => (
                  <div key={pack.id} className="flex items-center gap-2 rounded-md border px-2 py-1.5 text-sm">
                    <span className="min-w-0 flex-1 truncate">{pack.name}</span>
                    <Button size="sm" variant="ghost" onClick={() => applyContextPack(pack.id)}>
                      Apply
                    </Button>
                    <Button size="icon" variant="ghost" onClick={() => deleteContextPack(pack.id)}>
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="runbooks" className="m-0 space-y-3 p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="text-sm font-semibold">Runbooks</h2>
                  <p className="text-xs text-muted-foreground">
                    Prepare reusable prompts, then attach them to the composer or run immediately.
                  </p>
                </div>
                <Badge variant="secondary">{runbooks.length}</Badge>
              </div>
              <div className="grid min-h-[420px] grid-cols-[minmax(160px,0.8fr)_minmax(240px,1.2fr)] gap-3">
                <div className="min-h-0 overflow-hidden rounded-md border bg-card">
                  <TreeProvider
                    key={runbookTreeKey}
                    defaultExpandedIds={runbookGroups.map((group) => `category:${group.category}`)}
                    selectedIds={selectedRunbookTreeId ? [selectedRunbookTreeId] : []}
                    onSelectionChange={(ids) => {
                      const runbookId = ids.find((id) => id.startsWith("runbook:"))?.slice("runbook:".length);
                      if (runbookId) {
                        setSelectedRunbookId(runbookId);
                      }
                    }}
                    indent={18}
                    className="h-full"
                  >
                    <TreeView className="max-h-[420px] overflow-auto p-1.5">
                      {runbookGroups.map((group, groupIndex) => (
                        <TreeNode
                          key={group.category}
                          nodeId={`category:${group.category}`}
                          isLast={groupIndex === runbookGroups.length - 1}
                        >
                          <TreeNodeTrigger className="py-1.5">
                            <TreeExpander hasChildren />
                            <TreeIcon hasChildren />
                            <TreeLabel className="text-xs uppercase tracking-wide text-muted-foreground">
                              {group.category}
                            </TreeLabel>
                          </TreeNodeTrigger>
                          <TreeNodeContent hasChildren>
                            {group.items.map((runbook, itemIndex) => (
                              <TreeNode
                                key={runbook.id}
                                nodeId={`runbook:${runbook.id}`}
                                level={1}
                                isLast={itemIndex === group.items.length - 1}
                                parentPath={[groupIndex === runbookGroups.length - 1]}
                              >
                                <TreeNodeTrigger className="py-1.5">
                                  <TreeExpander />
                                  <TreeIcon />
                                  <TreeLabel>{runbook.title}</TreeLabel>
                                </TreeNodeTrigger>
                              </TreeNode>
                            ))}
                          </TreeNodeContent>
                        </TreeNode>
                      ))}
                    </TreeView>
                  </TreeProvider>
                </div>

                <div className="min-w-0 rounded-md border bg-card">
                  {selectedRunbook ? (
                    <div className="flex h-full min-h-[420px] flex-col">
                      <div className="border-b p-3">
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <p className="truncate text-sm font-semibold">{selectedRunbook.title}</p>
                            <p className="mt-1 text-xs leading-5 text-muted-foreground">
                              {selectedRunbook.description}
                            </p>
                          </div>
                          <Badge variant="outline">{selectedRunbook.variables.length}</Badge>
                        </div>
                      </div>
                      <div className="min-h-0 flex-1 space-y-3 overflow-auto p-3">
                        {selectedRunbook.variables.map((variable) => (
                          <label key={variable.name} className="block space-y-1.5">
                            <span className="flex items-center justify-between gap-2 text-xs font-medium text-muted-foreground">
                              <span>{variable.label}</span>
                              {variable.required ? <span>Required</span> : null}
                            </span>
                            <Textarea
                              value={runbookValues[variable.name] || ""}
                              onChange={(event) =>
                                setRunbookValues((current) => ({
                                  ...current,
                                  [variable.name]: event.currentTarget.value
                                }))
                              }
                              rows={variable.multiline ? 4 : 2}
                              className="min-h-[72px] resize-y text-sm leading-5"
                            />
                            {variable.description ? (
                              <span className="block text-xs leading-5 text-muted-foreground">
                                {variable.description}
                              </span>
                            ) : null}
                          </label>
                        ))}
                      </div>
                      <div className="flex justify-end gap-2 border-t p-3">
                        <Button size="sm" variant="outline" onClick={() => renderRunbook(false)}>
                          Attach
                        </Button>
                        <Button size="sm" onClick={() => renderRunbook(true)}>
                          <Play className="size-3.5" />
                          Run
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="p-3">
                      <EmptyLine>No runbook selected.</EmptyLine>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="settings" className="m-0 space-y-3 p-3">
              <Card className="rounded-md shadow-none">
                <CardHeader className="p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <CardTitle className="text-sm">Model backend</CardTitle>
                      <CardDescription>
                        Switch between local Ollama and an OpenAI-compatible endpoint.
                      </CardDescription>
                    </div>
                    <Button
                      size="sm"
                      onClick={saveModelSettings}
                      disabled={modelSettingsSaving || !modelForm.model.trim()}
                    >
                      {modelSettingsSaving ? <Loader2 className="size-4 animate-spin" /> : <Settings className="size-4" />}
                      Save
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3 p-3 pt-0">
                  <label className="block space-y-1">
                    <span className="text-xs text-muted-foreground">Provider</span>
                    <Select
                      value={modelForm.provider}
                      onValueChange={(provider) => setModelForm((current) => ({ ...current, provider }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ollama">Ollama</SelectItem>
                        <SelectItem value="openai">OpenAI-compatible</SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                  <label className="block space-y-1">
                    <span className="text-xs text-muted-foreground">Model</span>
                    <Input
                      value={modelForm.model}
                      onChange={(event) =>
                        setModelForm((current) => ({ ...current, model: event.currentTarget.value }))
                      }
                      placeholder={modelForm.provider === "openai" ? "openai/gpt-4o-mini" : "qwen3:1.7b"}
                    />
                  </label>
                  <label className="block space-y-1">
                    <span className="text-xs text-muted-foreground">Base URL</span>
                    <Input
                      value={modelForm.base_url}
                      onChange={(event) =>
                        setModelForm((current) => ({ ...current, base_url: event.currentTarget.value }))
                      }
                      disabled={modelForm.provider !== "openai"}
                      placeholder="https://openrouter.ai/api/v1"
                    />
                  </label>
                  <label className="block space-y-1">
                    <span className="text-xs text-muted-foreground">API key</span>
                    <Input
                      type="password"
                      value={modelForm.api_key}
                      onChange={(event) =>
                        setModelForm((current) => ({ ...current, api_key: event.currentTarget.value }))
                      }
                      disabled={modelForm.provider !== "openai"}
                      placeholder={modelSettings?.api_key_masked || "sk-..."}
                    />
                  </label>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="health" className="m-0 space-y-3 p-3">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="provider" value={health?.provider || "-"} />
                <Stat label="status" value={health?.status || "-"} />
              </div>
              <Card className="rounded-md shadow-none">
                <CardHeader className="p-3">
                  <CardTitle className="text-sm">Diagnostics</CardTitle>
                  <CardDescription>
                    {diagnostics?.counts?.error || 0} errors · {diagnostics?.counts?.warn || 0} warnings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 p-3 pt-0">
                  {(diagnostics?.checks || []).map((check) => (
                    <div key={check.id} className="flex items-start gap-2 text-sm">
                      {check.status === "ok" ? (
                        <CheckCircle2 className="mt-0.5 size-4 text-emerald-500" />
                      ) : check.status === "warn" ? (
                        <Clock3 className="mt-0.5 size-4 text-amber-500" />
                      ) : (
                        <XCircle className="mt-0.5 size-4 text-destructive" />
                      )}
                      <div>
                        <div className="font-medium">{check.title}</div>
                        <div className="text-xs text-muted-foreground">{check.detail}</div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
              <Card className="rounded-md shadow-none">
                <CardHeader className="p-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">Evals</CardTitle>
                    <Button size="sm" variant="outline" onClick={runEvals}>
                      <TerminalSquare className="size-4" />
                      Run
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="p-3 pt-0">
                  {evalSuite ? (
                    <div className="grid grid-cols-2 gap-2">
                      <Stat label="pass rate" value={`${Math.round(evalSuite.pass_rate * 100)}%`} />
                      <Stat label="failed" value={evalSuite.failed} tone={evalSuite.failed ? "text-destructive" : "text-emerald-500"} />
                    </div>
                  ) : (
                    <EmptyLine>Evals have not run in this session.</EmptyLine>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="approvals" className="m-0 space-y-3 p-3">
              {approvalsQueue.length === 0 ? (
                <EmptyLine>No pending approvals.</EmptyLine>
              ) : (
                approvalsQueue.map((approval) => (
                  <ApprovalCard
                    key={approval.approval_id}
                    approval={approval}
                    resolved={resolved[approval.approval_id] ?? null}
                    onResolve={resolveApproval}
                  />
                ))
              )}
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </aside>
    </div>
  );
}
