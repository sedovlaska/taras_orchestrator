import { Check, ShieldAlert, X } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ApprovalData } from "@/types";

/**
 * Approve/deny affordance for a PERSISTENT `data-approval` part. It hits the
 * UNCHANGED REST endpoints `/approvals/{id}/approve|deny` (via the shared `api`
 * object) and, on approval, resumes the owning run with a fresh turn — the
 * resume answer is then surfaced by the host view.
 */
export function ApprovalCard({
  approval,
  resolved,
  onResolve
}: {
  approval: ApprovalData;
  resolved?: "approve" | "deny" | null;
  onResolve: (approval: ApprovalData, action: "approve" | "deny") => Promise<void> | void;
}) {
  const [busy, setBusy] = useState<"approve" | "deny" | null>(null);
  const settled = resolved ?? null;

  async function handle(action: "approve" | "deny") {
    if (busy || settled) return;
    setBusy(action);
    try {
      await onResolve(approval, action);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-amber-500/15 text-amber-500">
          <ShieldAlert className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-semibold text-foreground">Tool approval required</p>
            <Badge variant="secondary" className="font-mono text-xs">
              {approval.tool_id}
            </Badge>
            {approval.risk ? (
              <Badge variant="outline" className="text-xs">
                {approval.risk} risk
              </Badge>
            ) : null}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {approval.agent ? `${approval.agent} · ` : ""}
            This call is gated by policy. Approve to run it and continue, or deny to stop the run.
          </p>

          {settled ? (
            <p className="mt-2.5 text-xs font-medium text-muted-foreground">
              {settled === "approve" ? "Approved — resuming." : "Denied — run stopped."}
            </p>
          ) : (
            <div className="mt-2.5 flex gap-2">
              <Button
                size="sm"
                onClick={() => handle("approve")}
                disabled={busy !== null}
                className="bg-emerald-600 text-white hover:bg-emerald-600/90"
              >
                <Check className="size-4" />
                {busy === "approve" ? "Approving…" : "Approve"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handle("deny")}
                disabled={busy !== null}
              >
                <X className="size-4" />
                {busy === "deny" ? "Denying…" : "Deny"}
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
