import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import type { UIMessage } from "ai";
import { Bot, Cpu, RefreshCw, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  Conversation,
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
import { Button } from "@/components/ui/button";
import { api } from "@/api";
import { notifyError } from "@/lib/notify";
import type { ApprovalData, ModelsResponse, TraceData } from "@/types";

import { ApprovalCard } from "./ApprovalCard";
import { RunnerTimeline } from "./RunnerTimeline";

// The orchestrator endpoint emits the AI SDK v5 UI-message-stream when asked
// with ?protocol=ai-sdk (Phase 1). We let the SDK own the transcript + transport
// and translate the outgoing body to the server's ChatRequest shape, since the
// orchestrator expects { message, conversation_id, model } rather than the full
// message array.
const CHAT_API = "/chat/stream?protocol=ai-sdk";

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

export function AppNext() {
  const [input, setInput] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState<string | null>(null);
  const [modelsReachable, setModelsReachable] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);

  // Governance telemetry for the in-flight turn. `traces` is the chronological
  // rail; `approvals` is keyed by approval id (PERSISTENT parts) with the
  // resolved action tracked so the affordance settles after approve/deny.
  const [traces, setTraces] = useState<TraceData[]>([]);
  const [approvals, setApprovals] = useState<ApprovalData[]>([]);
  const [resolved, setResolved] = useState<Record<string, "approve" | "deny">>({});

  // The transport reads these refs so a model/conversation change applies to the
  // very next send without re-instantiating the transport.
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

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  async function handleSubmit(message: PromptInputMessage) {
    const text = message.text.trim();
    if (!text || busy) return;

    // Auto-create a conversation on first send so the thread is persisted and
    // threading (conversation_id) is carried on every subsequent turn.
    let convo = conversationId;
    if (!convo) {
      try {
        const created = await api.createConversation(text.slice(0, 60));
        convo = created.conversation.id;
        setConversationId(convo);
        conversationRef.current = convo;
      } catch (err) {
        notifyError(err, "New conversation failed");
        return;
      }
    }

    // Each turn owns its own telemetry: clear the rail/approvals before sending.
    setTraces([]);
    setApprovals([]);
    setResolved({});
    setInput("");
    sendMessage({ text });
  }

  async function resolveApproval(approval: ApprovalData, action: "approve" | "deny") {
    try {
      await api.resolveApproval(approval.approval_id, action);
      setResolved((current) => ({ ...current, [approval.approval_id]: action }));
      setTraces((current) => [
        ...current,
        { event: "approval_resolved", tool_id: approval.tool_id, status: action }
      ]);
      if (action === "approve" && approval.run_id) {
        // Resume the gated run via a fresh turn against the UNCHANGED REST
        // endpoint, then surface the answer as a new assistant message.
        const data = await api.resumeRun(approval.run_id);
        for (const event of data.events || []) {
          setTraces((current) => [...current, { event: event.event, ...event.data }]);
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
    } catch (err) {
      notifyError(err, "Approval failed");
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

  const hasModels = modelsReachable && models.length > 0;
  const modelOptions = hasModels ? models : defaultModel ? [defaultModel] : [];
  const lastMessageIsAssistant = messages.at(-1)?.role === "assistant";

  return (
    <div className="dark flex h-screen flex-col bg-background text-foreground">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Bot className="size-5" />
          </span>
          <div className="leading-tight">
            <h1 className="text-sm font-semibold">AGNO Team Orchestrator</h1>
            <p className="text-xs text-muted-foreground">AI SDK Elements chat surface</p>
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={newConversation}>
          <Sparkles className="size-4" />
          New chat
        </Button>
      </header>

      <Conversation>
        <ConversationContent className="mx-auto w-full max-w-3xl">
          {messages.length === 0 ? (
            <ConversationEmptyState
              icon={<Bot className="size-7" />}
              title="Start a conversation"
              description="Ask the orchestrator anything. Routing, policy, and tool steps appear inline as it works."
            />
          ) : (
            messages.map((message, index) => {
              const isLastAssistant =
                message.role === "assistant" && index === messages.length - 1;
              const text = message.parts
                .filter((part): part is { type: "text"; text: string } => part.type === "text")
                .map((part) => part.text)
                .join("");
              return (
                <div key={message.id} className="space-y-3">
                  {/* The runner rail + any approvals belong to the assistant turn
                      they produced; pin them above the latest assistant bubble. */}
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
                            <Loader size={14} /> Thinking…
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

          {/* Streaming run with no assistant bubble yet (e.g. approval gate
              before any token): still show the rail + approvals. */}
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
            <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
              {error.message}
            </div>
          ) : null}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-4">
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputBody>
            <PromptInputTextarea
              value={input}
              onChange={(event) => setInput(event.currentTarget.value)}
              placeholder="Ask the orchestrator…"
            />
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputSelect
                  value={selectedModel ?? defaultModel ?? undefined}
                  onValueChange={setSelectedModel}
                  disabled={modelOptions.length === 0}
                >
                  <PromptInputSelectTrigger className="text-xs" aria-label="Model for next message">
                    <Cpu className="size-3.5" />
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
                  <button
                    type="button"
                    onClick={loadModels}
                    className="flex items-center gap-1 rounded-md px-1 text-xs text-amber-500 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    title="Ollama unreachable — using the configured default model. Click to retry."
                  >
                    <RefreshCw className="size-3" /> default model
                  </button>
                ) : null}
              </PromptInputTools>
              {/* No client-side abort is wired to this stream yet, so the
                  submit stays disabled while a turn is in flight rather than
                  offering a stop affordance that wouldn't stop anything. */}
              <PromptInputSubmit disabled={busy || !input.trim()} status={status} />
            </PromptInputFooter>
          </PromptInputBody>
        </PromptInput>
      </div>
    </div>
  );
}
