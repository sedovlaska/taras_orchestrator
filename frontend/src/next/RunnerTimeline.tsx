import {
  Activity,
  AlertTriangle,
  Check,
  ChevronDown,
  Cpu,
  GitBranch,
  Route,
  ShieldCheck,
  ShieldAlert,
  Wrench
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { Loader } from "@/components/ai-elements/loader";
import { cn } from "@/lib/utils";
import type { TraceData } from "@/types";

/**
 * A live, chronological rail of orchestrator steps rendered beside the
 * in-progress assistant turn. Ported from the Mantine `RunnerTimeline` into
 * shadcn's design language: while streaming it stays expanded with a pulsing
 * head; once the run settles (`done`/error/needs-approval) it collapses into a
 * single quiet summary line that can be re-opened to audit the run.
 *
 * Steps are derived from the TRANSIENT `data-trace` parts collected via the
 * `useChat({ onData })` callback (route, classify, policy_decision, the
 * runner_/tool_ families, and done).
 */

function iconFor(event: string): LucideIcon {
  if (event === "route" || event === "classify") return Route;
  if (event === "policy_decision") return ShieldAlert;
  if (event === "agent_start") return GitBranch;
  if (event === "approval_required" || event === "approval_resolved") return ShieldCheck;
  if (event === "tool_start" || event === "tool_result" || event === "tool_gated") return Wrench;
  if (event === "runner_start") return Cpu;
  if (event === "runner_result") return Check;
  if (event === "runner_error" || event.includes("error")) return AlertTriangle;
  if (event === "done") return Check;
  return Activity;
}

// Semantic tone for a step marker, reusing the shadcn token palette.
function toneFor(event: TraceData): "muted" | "primary" | "destructive" | "success" {
  if (event.event.includes("error")) return "destructive";
  if (event.event === "policy_decision") return event.allowed ? "success" : "destructive";
  if (event.event === "approval_required") return "primary";
  if (event.event === "done" || event.event === "runner_result") return "success";
  return "primary";
}

function labelFor(event: TraceData): string {
  switch (event.event) {
    case "route":
      return `Routed to ${(event.agents || []).join(", ") || "orchestrator"}`;
    case "classify":
      return `Classified as ${event.intent || "request"}`;
    case "policy_decision":
      return `${event.allowed ? "Allowed" : "Denied"} ${event.tool_id || "tool"}`;
    case "agent_start":
      return `${event.agent || "Agent"} engaged`;
    case "approval_required":
      return `${event.tool_id || "Tool"} needs approval`;
    case "approval_resolved":
      return `${event.tool_id || "Tool"} ${event.status || "resolved"}`;
    case "tool_start":
      return `Calling ${event.tool_id || "tool"}`;
    case "tool_result":
      return `${event.tool_id || "Tool"} returned`;
    case "tool_gated":
      return `${event.tool_id || "Tool"} gated`;
    case "runner_start":
      return `${event.runner || "Runner"} trying`;
    case "runner_result":
      return `${event.runner || "Runner"} answered`;
    case "runner_error":
      return `${event.runner || "Runner"} fell back`;
    case "done":
      return "Response ready";
    default:
      return event.event.replaceAll("_", " ");
  }
}

function detailFor(event: TraceData): string {
  if (event.event === "route") {
    return `${Math.round((event.confidence || 0) * 100)}% confidence${
      event.reason ? ` · ${event.reason}` : ""
    }`;
  }
  if (event.event === "policy_decision") {
    return `${event.reason || ""}${event.risk ? ` · ${event.risk} risk` : ""}`.trim();
  }
  if (event.message) return String(event.message);
  if (event.status) return String(event.status);
  if (event.runner) return String(event.runner);
  return "";
}

const markerTone: Record<string, string> = {
  muted: "bg-muted text-muted-foreground",
  primary: "bg-primary/10 text-primary",
  success: "bg-emerald-500/15 text-emerald-500",
  destructive: "bg-destructive/15 text-destructive"
};

export function RunnerTimeline({
  events,
  running
}: {
  events: TraceData[];
  running: boolean;
}) {
  const [open, setOpen] = useState(true);

  // Auto-collapse when the run settles; re-expand on a fresh run.
  useEffect(() => {
    setOpen(running);
  }, [running]);

  if (events.length === 0) return null;

  const failed = events.some((event) => event.event.includes("error"));
  const needsApproval = events.some((event) => event.event === "approval_required");
  const settledText = failed
    ? "Completed with fallbacks"
    : needsApproval
      ? "Waiting on approval"
      : "Completed";
  const stepCount = `${events.length} ${events.length === 1 ? "step" : "steps"}`;
  const headTone = failed || needsApproval ? "text-amber-500" : "text-emerald-500";

  return (
    <div className="overflow-hidden rounded-lg border bg-card/60">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label={running ? "Orchestrator working — toggle step detail" : `Run trace, ${stepCount} — toggle detail`}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition-colors hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
      >
        <span className="flex min-w-0 items-center gap-2">
          {running ? (
            <Loader size={14} className="text-primary" />
          ) : (
            <span
              className={cn(
                "flex size-5 items-center justify-center rounded-full",
                failed || needsApproval ? "bg-amber-500/15 text-amber-500" : "bg-emerald-500/15 text-emerald-500"
              )}
            >
              {failed ? <AlertTriangle className="size-3" /> : <Check className="size-3" />}
            </span>
          )}
          {running ? (
            <span className="text-xs font-bold uppercase tracking-wide text-primary">Working</span>
          ) : (
            <span className={cn("text-sm font-semibold", (failed || needsApproval) && headTone)}>
              {settledText}
            </span>
          )}
          <span className="truncate text-xs text-muted-foreground">
            {running ? labelFor(events[events.length - 1]) : `· ${stepCount}`}
          </span>
        </span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform motion-reduce:transition-none",
            open && "rotate-180"
          )}
        />
      </button>

      {open ? (
        <div className="px-3 pb-3 pt-1">
          {events.map((event, index) => {
            const Icon = iconFor(event.event);
            const tone = toneFor(event);
            const detail = detailFor(event);
            const isLast = index === events.length - 1;
            const pending = running && isLast;
            return (
              <div key={`${event.event}-${index}`} className="flex gap-2.5">
                <div className="flex flex-col items-center">
                  <span
                    className={cn(
                      "flex size-6 items-center justify-center rounded-full",
                      markerTone[tone],
                      pending && "ring-2 ring-primary/40"
                    )}
                  >
                    <Icon className="size-3" />
                  </span>
                  {!isLast ? <span className="my-0.5 w-px flex-1 bg-border" /> : null}
                </div>
                <div className={cn("min-w-0 flex-1", isLast ? "pb-0" : "pb-2.5")}>
                  <p className="truncate text-sm font-medium text-foreground">{labelFor(event)}</p>
                  {detail ? (
                    <p className="line-clamp-2 text-xs text-muted-foreground">{detail}</p>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
