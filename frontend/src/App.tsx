import {
  ActionIcon,
  AppShell,
  Badge,
  Box,
  Button,
  Card,
  Code,
  Divider,
  Group,
  Indicator,
  Loader,
  NavLink,
  Paper,
  Popover,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Tabs,
  Text,
  TextInput,
  Textarea,
  ThemeIcon,
  Title,
  Tooltip,
  rem
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import {
  IconActivity,
  IconBook2,
  IconBrandDocker,
  IconChartBar,
  IconCheck,
  IconClipboardText,
  IconCode,
  IconCopy,
  IconDatabase,
  IconFileSearch,
  IconHistory,
  IconMessage,
  IconMessagePlus,
  IconPlayerPlay,
  IconRefresh,
  IconRobot,
  IconSearch,
  IconSend,
  IconSettings,
  IconShieldCheck,
  IconTerminal2,
  IconTrash,
  IconX
} from "@tabler/icons-react";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { api, streamChat } from "./api";
import type {
  AgentStatus,
  ChatMessage,
  Conversation,
  ContextPack,
  Diagnostics,
  EvalSuite,
  Health,
  RunListItem,
  RunSummary,
  Runbook,
  TimelineEvent,
  WorkspaceFile
} from "./types";

const agentIcons: Record<string, typeof IconRobot> = {
  code: IconCode,
  db: IconDatabase,
  devops: IconTerminal2,
  docs: IconBook2,
  system: IconSettings,
  docker: IconBrandDocker
};

const agentColors: Record<string, string> = {
  code: "blue",
  db: "orange",
  devops: "red",
  docs: "violet",
  system: "teal",
  docker: "cyan"
};

const makeId = () => Math.random().toString(36).slice(2);

const WELCOME_MESSAGE =
  "UI is live without Ollama. Browse workspace files, run diagnostics, run local evals, inspect history, and prepare prompts.";

function welcomeMessages(): ChatMessage[] {
  return [{ id: makeId(), role: "system", content: WELCOME_MESSAGE }];
}

function notifyError(error: unknown, title = "Request failed") {
  notifications.show({
    color: "red",
    title,
    message: error instanceof Error ? error.message : String(error)
  });
}

function formatDate(value?: string) {
  if (!value) return "";
  return new Date(value).toLocaleString();
}

function relativeTime(value?: string) {
  if (!value) return "";
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(value).toLocaleDateString();
}

function eventTone(event: TimelineEvent) {
  if (event.event.includes("error")) return "red";
  if (event.event.includes("approval")) return "orange";
  if (event.event.includes("policy")) return event.data.allowed ? "green" : "red";
  if (event.event === "done") return "green";
  return "blue";
}

function eventTitle(event: TimelineEvent) {
  const data = event.data || {};
  if (event.event === "route") return `Route: ${(data.agents || []).join(", ") || "orchestrator"}`;
  if (event.event === "policy_decision") return data.tool_id || "Policy decision";
  if (event.event === "agent_start") return `${data.agent || "Agent"} selected`;
  if (event.event === "runner_error") return `${data.runner || "Runner"} failed`;
  if (event.event === "approval_required") return `${data.tool_id || "Tool"} needs approval`;
  if (event.event === "done") return data.intent || "Run complete";
  return event.event.replaceAll("_", " ");
}

function eventDetail(event: TimelineEvent) {
  const data = event.data || {};
  if (event.event === "route") {
    return `${Math.round((data.confidence || 0) * 100)}% confidence. ${data.reason || ""}`;
  }
  if (event.event === "policy_decision") {
    return `${data.reason || ""} · ${data.risk || "unknown"} risk`;
  }
  if (data.message) return data.message;
  if (data.status) return data.status;
  if (data.runner) return data.runner;
  return "";
}

// Walk the rendered <pre> subtree to recover the raw text for copying, and
// sniff the language from the inner <code class="language-xxx">.
function extractCode(node: unknown): { text: string; language: string | null } {
  let language: string | null = null;

  function text(child: any): string {
    if (child == null || typeof child === "boolean") return "";
    if (typeof child === "string" || typeof child === "number") return String(child);
    if (Array.isArray(child)) return child.map(text).join("");
    const props = child?.props;
    if (props) {
      if (typeof props.className === "string") {
        const match = /language-(\w+)/.exec(props.className);
        if (match) language = match[1];
      }
      return text(props.children);
    }
    return "";
  }

  const raw = text(node).replace(/\n$/, "");
  return { text: raw, language };
}

function CodeBlock({ children }: { children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const { text, language } = extractCode(children);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      // clipboard unavailable; ignore
    }
  }

  return (
    <div className="md-codeblock">
      {language ? <span className="md-codeblock-lang">{language}</span> : null}
      <Tooltip label={copied ? "Copied" : "Copy"} withArrow position="left">
        <ActionIcon
          size="sm"
          variant="subtle"
          color={copied ? "green" : "gray"}
          onClick={copy}
          aria-label="Copy code"
          className="md-copy-btn"
        >
          {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
        </ActionIcon>
      </Tooltip>
      <pre className="md-pre">{children}</pre>
    </div>
  );
}

const markdownComponents: Components = {
  pre({ children }) {
    return <CodeBlock>{children}</CodeBlock>;
  },
  code({ className, children, ...rest }) {
    // Block code keeps the hljs/language class (rehype-highlight target);
    // inline code (no language class) gets the lightweight pill style.
    const isBlock = /\bhljs\b|language-/.test(className || "");
    return (
      <code className={isBlock ? className : "md-inline-code"} {...rest}>
        {children}
      </code>
    );
  },
  a({ children, href }) {
    return (
      <a href={href} target="_blank" rel="noreferrer noopener">
        {children}
      </a>
    );
  },
  table({ children }) {
    return (
      <div className="md-table-wrap">
        <table>{children}</table>
      </div>
    );
  }
};

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { ignoreMissing: true, detect: true }]]}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function AgentBadge({ name, online = true }: { name: string; online?: boolean }) {
  const Icon = agentIcons[name] || IconRobot;
  return (
    <Badge
      leftSection={<Icon size={12} />}
      color={online ? agentColors[name] || "gray" : "gray"}
      variant={online ? "light" : "outline"}
      radius="sm"
    >
      {name}
    </Badge>
  );
}

function Metric({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <Paper withBorder p="sm" radius="sm">
      <Text fw={800} size="xl" c={color}>
        {value}
      </Text>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
    </Paper>
  );
}

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [activeAgents, setActiveAgents] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>(welcomeMessages);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [loadingConversationId, setLoadingConversationId] = useState<string | null>(null);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loadingChat, setLoadingChat] = useState(false);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
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

  const onlineCount = agents.filter((agent) => agent.online).length;
  const selectedRunbook = runbooks.find((runbook) => runbook.id === selectedRunbookId) || null;
  const errorRuns = useMemo(() => {
    const statusErrors = (summary.by_status || [])
      .filter((row) => ["failed", "denied"].includes(row.status))
      .reduce((total, row) => total + row.count, 0);
    return statusErrors + (summary.recent_errors?.length || 0);
  }, [summary]);

  async function refreshAll() {
    try {
      const [healthData, agentData, runData, summaryData, packsData, runbooksData, conversationsData] =
        await Promise.all([
          api.health(),
          api.agents(),
          api.runs(),
          api.summary(),
          api.contextPacks(),
          api.runbooks(),
          api.conversations()
        ]);
      setHealth(healthData);
      setAgents(agentData.agents || []);
      setRuns(runData.runs || []);
      setSummary(summaryData.summary || {});
      setContextPacks(packsData.packs || []);
      setRunbooks(runbooksData.runbooks || []);
      setConversations(conversationsData.conversations || []);
      if (!selectedRunbookId && runbooksData.runbooks?.length) {
        setSelectedRunbookId(runbooksData.runbooks[0].id);
      }
    } catch (error) {
      notifyError(error, "Refresh failed");
    }
  }

  async function refreshDiagnostics() {
    try {
      const data = await api.diagnostics();
      setDiagnostics(data.diagnostics);
    } catch (error) {
      notifyError(error, "Diagnostics failed");
    }
  }

  async function runEvals() {
    try {
      const data = await api.evals();
      setEvalSuite(data.suite);
    } catch (error) {
      notifyError(error, "Evals failed");
    }
  }

  async function loadWorkspace() {
    try {
      const data = fileSearch.trim()
        ? await api.workspaceSearch(fileSearch.trim())
        : await api.workspaceFiles();
      setFiles((data as { files?: WorkspaceFile[]; results?: WorkspaceFile[] }).files || (data as any).results || []);
    } catch (error) {
      notifyError(error, "Workspace failed");
    }
  }

  async function openFile(path: string) {
    try {
      const data = await api.workspaceFile(path);
      setSelectedPath(path);
      setPreview(data.file.content);
    } catch (error) {
      notifyError(error, "File preview failed");
    }
  }

  function appendPrompt(text: string) {
    setInput((current) => (current.trim() ? `${current.trim()}\n\n${text}` : text));
  }

  async function attachFile() {
    if (!selectedPath) return;
    try {
      const data = await api.contextBundle({ paths: [selectedPath] });
      appendPrompt(data.bundle.prompt_context);
      notifications.show({ color: "green", title: "Context attached", message: selectedPath });
    } catch (error) {
      notifyError(error, "Context failed");
    }
  }

  async function attachSearch() {
    if (!fileSearch.trim()) return;
    try {
      const data = await api.contextBundle({ query: fileSearch.trim() });
      appendPrompt(data.bundle.prompt_context);
      notifications.show({ color: "green", title: "Search attached", message: fileSearch });
    } catch (error) {
      notifyError(error, "Context failed");
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
      setContextPacks((items) => [data.pack, ...items]);
      setContextPackName("");
    } catch (error) {
      notifyError(error, "Save pack failed");
    }
  }

  async function applyPack(packId: string) {
    try {
      const data = await api.applyContextPack(packId);
      appendPrompt(data.bundle.prompt_context);
    } catch (error) {
      notifyError(error, "Apply pack failed");
    }
  }

  async function deletePack(packId: string) {
    try {
      await api.deleteContextPack(packId);
      setContextPacks((items) => items.filter((item) => item.id !== packId));
    } catch (error) {
      notifyError(error, "Delete pack failed");
    }
  }

  async function renderRunbook() {
    if (!selectedRunbook) return "";
    const data = await api.renderRunbook(selectedRunbook.id, runbookValues);
    return data.prompt;
  }

  async function useRunbook(runNow = false) {
    try {
      const prompt = await renderRunbook();
      if (runNow) {
        await sendMessage(prompt);
      } else {
        setInput(prompt);
      }
    } catch (error) {
      notifyError(error, "Runbook failed");
    }
  }

  async function selectRun(runId: string) {
    try {
      const [runData, eventsData] = await Promise.all([api.run(runId), api.runEvents(runId)]);
      setTimeline(eventsData.events || []);
      setMessages((current) => [
        ...current,
        { id: makeId(), role: "user", content: runData.run.message },
        { id: makeId(), role: "system", content: `Loaded run ${runId}` }
      ]);
    } catch (error) {
      notifyError(error, "Run load failed");
    }
  }

  async function exportTrace(runId: string) {
    try {
      const data = await api.runTrace(runId);
      setPreview(data.trace.markdown);
      notifications.show({ color: "blue", title: "Trace loaded", message: "Trace markdown is in Workspace preview." });
    } catch (error) {
      notifyError(error, "Trace failed");
    }
  }

  async function resolveApproval(event: TimelineEvent, action: "approve" | "deny") {
    try {
      await api.resolveApproval(event.data.approval_id, action);
      setTimeline((items) => [
        ...items,
        { event: "approval_resolved", data: { tool_id: event.data.tool_id, status: action } }
      ]);
      if (action === "approve" && event.data.run_id) {
        const data = await api.resumeRun(event.data.run_id);
        setTimeline((items) => [...items, ...(data.events || [])]);
        setMessages((items) => [
          ...items,
          { id: makeId(), role: "assistant", content: data.answer, agents: data.agents_used || [] }
        ]);
      }
      refreshAll();
    } catch (error) {
      notifyError(error, "Approval failed");
    }
  }

  async function refreshConversations() {
    try {
      const data = await api.conversations();
      setConversations(data.conversations || []);
    } catch (error) {
      notifyError(error, "Conversations failed");
    }
  }

  function startNewConversation() {
    setActiveConversationId(null);
    setMessages(welcomeMessages());
    setTimeline([]);
    setActiveAgents([]);
    setInput("");
  }

  async function openConversation(id: string) {
    if (id === activeConversationId || loadingConversationId) return;
    setLoadingConversationId(id);
    try {
      const data = await api.conversation(id);
      setActiveConversationId(id);
      setTimeline([]);
      setActiveAgents([]);
      const loaded: ChatMessage[] = data.messages.map((message) => ({
        id: makeId(),
        role: message.role === "assistant" ? "assistant" : message.role === "user" ? "user" : "system",
        content: message.content
      }));
      setMessages(
        loaded.length
          ? loaded
          : [{ id: makeId(), role: "system", content: "Empty conversation. Send a message to begin." }]
      );
    } catch (error) {
      notifyError(error, "Conversation load failed");
    } finally {
      setLoadingConversationId(null);
    }
  }

  async function confirmDeleteConversation(id: string) {
    setPendingDeleteId(null);
    try {
      await api.deleteConversation(id);
      setConversations((items) => items.filter((item) => item.id !== id));
      if (id === activeConversationId) startNewConversation();
    } catch (error) {
      notifyError(error, "Delete conversation failed");
    }
  }

  async function sendMessage(override?: string) {
    const text = (override ?? input).trim();
    if (!text || loadingChat) return;

    setLoadingChat(true);
    setInput("");
    setTimeline([]);
    setActiveAgents([]);

    // Auto-create a conversation on first send so every thread is persisted
    // and shows up in the sidebar without a separate "empty thread" state.
    let conversationId = activeConversationId;
    if (!conversationId) {
      try {
        const created = await api.createConversation(text.slice(0, 60));
        conversationId = created.conversation.id;
        setActiveConversationId(conversationId);
        setMessages([]);
      } catch (error) {
        notifyError(error, "New conversation failed");
        setLoadingChat(false);
        return;
      }
    }

    setMessages((items) => [...items, { id: makeId(), role: "user", content: text }]);

    let assistantId = makeId();
    let answer = "";
    setMessages((items) => [...items, { id: assistantId, role: "assistant", content: "", agents: [] }]);

    try {
      await streamChat(text, {
        onEvent: (event) => setTimeline((items) => [...items, event]),
        onAgents: (names) => setActiveAgents(names),
        onChunk: (chunk) => {
          answer += chunk;
          setMessages((items) =>
            items.map((item) => (item.id === assistantId ? { ...item, content: answer } : item))
          );
        },
        onDone: (payload) => {
          answer = payload.answer || answer;
          setMessages((items) =>
            items.map((item) =>
              item.id === assistantId
                ? { ...item, content: answer || "Done.", agents: payload.agents_used || item.agents }
                : item
            )
          );
        }
      }, conversationId);
      refreshAll();
    } catch (error) {
      setMessages((items) =>
        items.map((item) =>
          item.id === assistantId
            ? { ...item, role: "error", content: error instanceof Error ? error.message : String(error) }
            : item
        )
      );
    } finally {
      setLoadingChat(false);
      window.setTimeout(() => setActiveAgents([]), 1200);
    }
  }

  useEffect(() => {
    refreshAll();
    refreshDiagnostics();
    loadWorkspace();
  }, []);

  useEffect(() => {
    if (!selectedRunbook) return;
    const defaults = Object.fromEntries(
      selectedRunbook.variables.map((variable) => [variable.name, variable.default || ""])
    );
    setRunbookValues(defaults);
  }, [selectedRunbookId]);

  return (
    <AppShell
      navbar={{ width: 320, breakpoint: "md" }}
      aside={{ width: 460, breakpoint: "lg" }}
      header={{ height: 58 }}
      padding="md"
      className="shell"
    >
      <AppShell.Header className="topbar">
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="sm" wrap="nowrap">
            <ThemeIcon color="dark" radius="sm" size="lg">
              <IconRobot size={20} />
            </ThemeIcon>
            <Box>
              <Title order={4}>AGNO Team Orchestrator</Title>
              <Text size="xs" c="dimmed">
                Local routing, runbooks, traces, workspace context
              </Text>
            </Box>
          </Group>
          <Group gap="xs" wrap="nowrap">
            <Badge color={health?.status === "ok" ? "green" : "gray"} variant="light">
              {health?.status || "loading"}
            </Badge>
            <Tooltip label="Refresh dashboard data">
              <ActionIcon variant="subtle" onClick={refreshAll} aria-label="Refresh">
                <IconRefresh size={18} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        <ScrollArea h="100%">
          <Stack gap="md">
            <Card withBorder radius="sm" p="md">
              <Group justify="space-between" align="start">
                <Box>
                  <Text size="xs" c="dimmed" fw={700} tt="uppercase">
                    Team
                  </Text>
                  <Title order={2}>{onlineCount}/{agents.length || 6}</Title>
                </Box>
                <ThemeIcon color="green" variant="light">
                  <IconShieldCheck size={20} />
                </ThemeIcon>
              </Group>
              <Text size="sm" c="dimmed" mt={4}>
                Ollama is optional for UI exploration. Local APIs are online.
              </Text>
            </Card>

            <Stack gap="xs">
              <Text size="xs" c="dimmed" fw={700} tt="uppercase">
                Conversations
              </Text>
              <Button
                variant={activeConversationId ? "default" : "light"}
                leftSection={<IconMessagePlus size={16} />}
                onClick={startNewConversation}
                fullWidth
                justify="flex-start"
              >
                New conversation
              </Button>
              <Stack gap={4}>
                {conversations.length === 0 ? (
                  <Text size="sm" c="dimmed" ta="center" py="sm">
                    Your conversations will appear here. Start one above or just send a message.
                  </Text>
                ) : (
                  conversations.map((conversation) => {
                    const active = conversation.id === activeConversationId;
                    return (
                      <NavLink
                        key={conversation.id}
                        active={active}
                        label={conversation.title}
                        description={relativeTime(conversation.updated_at)}
                        onClick={() => openConversation(conversation.id)}
                        leftSection={
                          loadingConversationId === conversation.id ? (
                            <Loader size={14} />
                          ) : (
                            <IconMessage size={16} />
                          )
                        }
                        rightSection={
                          <Popover
                            opened={pendingDeleteId === conversation.id}
                            onChange={(opened) => !opened && setPendingDeleteId(null)}
                            position="bottom-end"
                            withArrow
                            shadow="md"
                            width={220}
                          >
                            <Popover.Target>
                              <ActionIcon
                                component="div"
                                role="button"
                                color="red"
                                variant="subtle"
                                size="sm"
                                aria-label={`Delete ${conversation.title}`}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setPendingDeleteId(
                                    pendingDeleteId === conversation.id ? null : conversation.id
                                  );
                                }}
                              >
                                <IconTrash size={14} />
                              </ActionIcon>
                            </Popover.Target>
                            <Popover.Dropdown onClick={(event) => event.stopPropagation()}>
                              <Text size="sm" fw={600}>
                                Delete this conversation?
                              </Text>
                              <Text size="xs" c="dimmed" mt={2}>
                                This permanently removes its messages.
                              </Text>
                              <Group justify="flex-end" gap="xs" mt="sm">
                                <Button
                                  size="xs"
                                  variant="default"
                                  onClick={() => setPendingDeleteId(null)}
                                >
                                  Cancel
                                </Button>
                                <Button
                                  size="xs"
                                  color="red"
                                  onClick={() => confirmDeleteConversation(conversation.id)}
                                >
                                  Delete
                                </Button>
                              </Group>
                            </Popover.Dropdown>
                          </Popover>
                        }
                        variant="light"
                        className="conversation-link"
                      />
                    );
                  })
                )}
              </Stack>
            </Stack>

            <Divider />

            <Stack gap={6}>
              {["code", "db", "devops", "docs", "system", "docker"].map((name) => {
                const agent = agents.find((item) => item.name === name);
                const active = activeAgents.includes(name);
                const Icon = agentIcons[name] || IconRobot;
                return (
                  <NavLink
                    key={name}
                    active={active}
                    label={name}
                    description={active ? "running" : agent?.online === false ? "offline" : "agno member"}
                    leftSection={
                      <Indicator disabled={!active} color={agentColors[name]} size={8}>
                        <ThemeIcon color={agentColors[name]} variant="light" size="sm">
                          <Icon size={16} />
                        </ThemeIcon>
                      </Indicator>
                    }
                    rightSection={
                      <Badge size="xs" color={agent?.online === false ? "red" : "green"} variant="dot">
                        {agent?.online === false ? "off" : "on"}
                      </Badge>
                    }
                    variant="light"
                    className="agent-link"
                  />
                );
              })}
            </Stack>

            <Divider />

            <Stack gap="xs">
              <Group justify="space-between">
                <Text size="xs" c="dimmed" fw={700} tt="uppercase">
                  Runbook
                </Text>
                <IconClipboardText size={16} color="var(--mantine-color-dimmed)" />
              </Group>
              <Select
                data={runbooks.map((runbook) => ({ value: runbook.id, label: runbook.title }))}
                value={selectedRunbookId}
                onChange={setSelectedRunbookId}
                placeholder="Choose runbook"
              />
              {selectedRunbook && (
                <>
                  <Text size="sm" c="dimmed">
                    {selectedRunbook.category} · {selectedRunbook.description}
                  </Text>
                  <Stack gap="xs">
                    {selectedRunbook.variables.map((variable) =>
                      variable.multiline ? (
                        <Textarea
                          key={variable.name}
                          label={variable.label}
                          description={variable.description}
                          value={runbookValues[variable.name] || ""}
                          minRows={3}
                          onChange={(event) =>
                            setRunbookValues((values) => ({
                              ...values,
                              [variable.name]: event.currentTarget.value
                            }))
                          }
                        />
                      ) : (
                        <TextInput
                          key={variable.name}
                          label={variable.label}
                          description={variable.description}
                          value={runbookValues[variable.name] || ""}
                          onChange={(event) =>
                            setRunbookValues((values) => ({
                              ...values,
                              [variable.name]: event.currentTarget.value
                            }))
                          }
                        />
                      )
                    )}
                  </Stack>
                  <Group grow>
                    <Button variant="default" onClick={() => useRunbook(false)}>
                      Use
                    </Button>
                    <Button leftSection={<IconPlayerPlay size={16} />} onClick={() => useRunbook(true)}>
                      Run
                    </Button>
                  </Group>
                </>
              )}
            </Stack>
          </Stack>
        </ScrollArea>
      </AppShell.Navbar>

      <AppShell.Main className="main">
        <Stack h="calc(100vh - 90px)" gap="md">
          <ScrollArea className="chat-scroll" offsetScrollbars>
            <Stack gap="sm" p="xs">
              {messages.map((message) => (
                <Paper
                  key={message.id}
                  withBorder={message.role !== "user"}
                  radius="sm"
                  p="md"
                  className={`message message-${message.role}`}
                >
                  {message.agents?.length ? (
                    <Group gap={4} mb={6}>
                      {message.agents.map((agent) => (
                        <AgentBadge key={agent} name={agent} />
                      ))}
                    </Group>
                  ) : null}
                  {message.role === "assistant" ? (
                    message.content ? (
                      <MarkdownMessage content={message.content} />
                    ) : (
                      <Text size="sm" c="dimmed" className="message-text">
                        {loadingChat ? "Thinking..." : ""}
                      </Text>
                    )
                  ) : (
                    <Text size="sm" component="pre" className="message-text">
                      {message.content}
                    </Text>
                  )}
                </Paper>
              ))}
            </Stack>
          </ScrollArea>

          <Paper withBorder radius="sm" p="sm" className="composer">
            <Group align="end" gap="sm" wrap="nowrap">
              <Textarea
                autosize
                minRows={2}
                maxRows={6}
                placeholder="Ask, paste context, or run a prepared runbook..."
                value={input}
                disabled={loadingChat}
                onChange={(event) => setInput(event.currentTarget.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                    event.preventDefault();
                    sendMessage();
                  }
                }}
                className="composer-input"
              />
              <Button
                h={54}
                disabled={loadingChat || !input.trim()}
                loading={loadingChat}
                rightSection={<IconSend size={16} />}
                onClick={() => sendMessage()}
              >
                Send
              </Button>
            </Group>
          </Paper>
        </Stack>
      </AppShell.Main>

      <AppShell.Aside p="md">
        <Tabs defaultValue="timeline" keepMounted={false} h="100%" className="inspector">
          <Tabs.List grow>
            <Tabs.Tab value="timeline" leftSection={<IconActivity size={16} />}>
              Timeline
            </Tabs.Tab>
            <Tabs.Tab value="history" leftSection={<IconHistory size={16} />}>
              History
            </Tabs.Tab>
            <Tabs.Tab value="workspace" leftSection={<IconFileSearch size={16} />}>
              Workspace
            </Tabs.Tab>
            <Tabs.Tab value="health" leftSection={<IconChartBar size={16} />}>
              Health
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="timeline" pt="md">
            <ScrollArea h="calc(100vh - 150px)" offsetScrollbars>
              <Stack gap="xs">
                {timeline.length === 0 ? (
                  <Text size="sm" c="dimmed">
                    Routing, policy, runner, approval, and tool events appear here.
                  </Text>
                ) : (
                  timeline.map((event, index) => (
                    <Card key={`${event.event}-${index}`} withBorder radius="sm" p="sm">
                      <Group justify="space-between" gap="xs">
                        <Badge color={eventTone(event)} variant="light">
                          {event.event}
                        </Badge>
                        {event.event === "approval_required" ? (
                          <Group gap={4}>
                            <ActionIcon
                              color="green"
                              variant="light"
                              onClick={() => resolveApproval(event, "approve")}
                              aria-label="Approve"
                            >
                              <IconCheck size={16} />
                            </ActionIcon>
                            <ActionIcon
                              color="red"
                              variant="light"
                              onClick={() => resolveApproval(event, "deny")}
                              aria-label="Deny"
                            >
                              <IconX size={16} />
                            </ActionIcon>
                          </Group>
                        ) : null}
                      </Group>
                      <Text fw={700} size="sm" mt={6}>
                        {eventTitle(event)}
                      </Text>
                      {eventDetail(event) ? (
                        <Text size="xs" c="dimmed" mt={2}>
                          {eventDetail(event)}
                        </Text>
                      ) : null}
                    </Card>
                  ))
                )}
              </Stack>
            </ScrollArea>
          </Tabs.Panel>

          <Tabs.Panel value="history" pt="md">
            <ScrollArea h="calc(100vh - 150px)" offsetScrollbars>
              <Stack gap="md">
                <SimpleGrid cols={2}>
                  <Metric label="runs" value={summary.total_runs || 0} />
                  <Metric label="errors" value={errorRuns} color={errorRuns ? "red" : "green"} />
                </SimpleGrid>
                <Stack gap="xs">
                  {runs.length === 0 ? (
                    <Text size="sm" c="dimmed">
                      No recorded runs yet.
                    </Text>
                  ) : (
                    runs.map((run) => (
                      <Card key={run.id} withBorder radius="sm" p="sm">
                        <Text fw={700} size="sm" lineClamp={2}>
                          {run.message}
                        </Text>
                        <Group gap={6} mt={8}>
                          <Badge size="sm" variant="light">
                            {run.status}
                          </Badge>
                          {(run.agents || []).slice(0, 3).map((agent) => (
                            <AgentBadge key={agent} name={agent} />
                          ))}
                        </Group>
                        <Text size="xs" c="dimmed" mt={6}>
                          {formatDate(run.created_at)}
                        </Text>
                        <Group mt="sm" grow>
                          <Button variant="default" size="xs" onClick={() => selectRun(run.id)}>
                            Open
                          </Button>
                          <Button variant="light" size="xs" onClick={() => exportTrace(run.id)}>
                            Trace
                          </Button>
                        </Group>
                      </Card>
                    ))
                  )}
                </Stack>
              </Stack>
            </ScrollArea>
          </Tabs.Panel>

          <Tabs.Panel value="workspace" pt="md">
            <ScrollArea h="calc(100vh - 150px)" offsetScrollbars>
              <Stack gap="md">
                <Group gap="xs" wrap="nowrap">
                  <TextInput
                    placeholder="Search files"
                    value={fileSearch}
                    onChange={(event) => setFileSearch(event.currentTarget.value)}
                    leftSection={<IconSearch size={16} />}
                    className="grow"
                  />
                  <Button variant="default" onClick={loadWorkspace}>
                    Search
                  </Button>
                </Group>

                <Stack gap="xs">
                  {files.map((file) => (
                    <Card
                      key={file.path}
                      withBorder
                      radius="sm"
                      p="sm"
                      className={selectedPath === file.path ? "selected-card" : undefined}
                      onClick={() => openFile(file.path)}
                    >
                      <Text fw={700} size="sm">
                        {file.path}
                      </Text>
                      <Text size="xs" c="dimmed">
                        {file.matches ? `${file.matches} matches · ` : ""}
                        {Math.round(file.size_bytes / 1024)} KB
                      </Text>
                    </Card>
                  ))}
                </Stack>

                <Textarea
                  label="Preview"
                  value={preview}
                  autosize
                  minRows={8}
                  maxRows={14}
                  readOnly
                  className="preview"
                />

                <Group grow>
                  <Button variant="default" disabled={!selectedPath} onClick={attachFile}>
                    Attach file
                  </Button>
                  <Button variant="light" disabled={!fileSearch.trim()} onClick={attachSearch}>
                    Attach search
                  </Button>
                </Group>

                <Divider label="Context packs" labelPosition="center" />
                <Group gap="xs" wrap="nowrap">
                  <TextInput
                    placeholder="Pack name"
                    value={contextPackName}
                    onChange={(event) => setContextPackName(event.currentTarget.value)}
                    className="grow"
                  />
                  <Button variant="default" onClick={saveContextPack}>
                    Save
                  </Button>
                </Group>
                <Stack gap="xs">
                  {contextPacks.map((pack) => (
                    <Card key={pack.id} withBorder radius="sm" p="sm">
                      <Group justify="space-between" wrap="nowrap">
                        <Box>
                          <Text fw={700} size="sm">
                            {pack.name}
                          </Text>
                          <Text size="xs" c="dimmed">
                            {(pack.paths || []).length} files {pack.query ? `· ${pack.query}` : ""}
                          </Text>
                        </Box>
                        <Group gap={4} wrap="nowrap">
                          <ActionIcon variant="light" onClick={() => applyPack(pack.id)} aria-label="Apply">
                            <IconClipboardText size={16} />
                          </ActionIcon>
                          <ActionIcon
                            color="red"
                            variant="subtle"
                            onClick={() => deletePack(pack.id)}
                            aria-label="Delete"
                          >
                            <IconTrash size={16} />
                          </ActionIcon>
                        </Group>
                      </Group>
                    </Card>
                  ))}
                </Stack>
              </Stack>
            </ScrollArea>
          </Tabs.Panel>

          <Tabs.Panel value="health" pt="md">
            <ScrollArea h="calc(100vh - 150px)" offsetScrollbars>
              <Stack gap="md">
                <SimpleGrid cols={2}>
                  <Metric label="diagnostics" value={diagnostics?.status || "-"} color={diagnostics?.status === "ok" ? "green" : "orange"} />
                  <Metric
                    label="eval pass"
                    value={evalSuite ? `${Math.round((evalSuite.pass_rate || 0) * 100)}%` : "-"}
                    color={evalSuite?.failed ? "red" : "green"}
                  />
                </SimpleGrid>
                <Group grow>
                  <Button variant="default" onClick={refreshDiagnostics}>
                    Check diagnostics
                  </Button>
                  <Button leftSection={<IconPlayerPlay size={16} />} onClick={runEvals}>
                    Run evals
                  </Button>
                </Group>

                <Stack gap="xs">
                  {(diagnostics?.checks || []).map((check) => (
                    <Card key={check.id} withBorder radius="sm" p="sm">
                      <Group justify="space-between">
                        <Text fw={700} size="sm">
                          {check.title}
                        </Text>
                        <Badge color={check.status === "ok" ? "green" : check.status === "warn" ? "orange" : "red"}>
                          {check.status}
                        </Badge>
                      </Group>
                      <Text size="xs" c="dimmed" mt={4}>
                        {check.detail}
                      </Text>
                    </Card>
                  ))}
                </Stack>

                {evalSuite ? (
                  <Stack gap="xs">
                    <Divider label="Eval suite" labelPosition="center" />
                    {evalSuite.results.map((result) => (
                      <Card key={result.id} withBorder radius="sm" p="sm">
                        <Group justify="space-between">
                          <Code>{result.id}</Code>
                          <Badge color={result.passed ? "green" : "red"}>
                            {result.passed ? "pass" : "fail"}
                          </Badge>
                        </Group>
                        {result.failures?.length || result.reason ? (
                          <Text size="xs" c="dimmed" mt={4}>
                            {result.failures?.join("; ") || result.reason}
                          </Text>
                        ) : null}
                      </Card>
                    ))}
                  </Stack>
                ) : null}
              </Stack>
            </ScrollArea>
          </Tabs.Panel>
        </Tabs>
      </AppShell.Aside>

      {loadingChat ? (
        <Box className="loading-strip">
          <Loader size={rem(14)} />
          <Text size="xs" fw={700}>
            Running orchestrator
          </Text>
        </Box>
      ) : null}
    </AppShell>
  );
}
