# Product

## Register

product

## Users

Developers and maintainers using a trusted local orchestration console to route AI-assisted work across code, docs, DevOps, system, Docker, and database agents. They are usually debugging, reviewing, preparing prompts, inspecting runs, or controlling tool approvals from a desktop browser.

## Product Purpose

AGNO Team Orchestrator is a local multi-agent assistant with a FastAPI backend and browser console. It records runs, traces routing and policy decisions, exposes diagnostics, manages workspace context, supports reusable runbooks, and gates risky tools through explicit approvals. Success means users can see what the orchestrator is doing, recover or audit past runs, and safely control model/tool behavior without leaving the local workflow.

## Brand Personality

Practical, transparent, controlled. The interface should feel like an operational tool for repeated work, not a marketing surface or decorative AI demo.

## Anti-references

Avoid landing-page composition, decorative AI gradients, generic chat-only shells, hidden policy behavior, oversized empty cards, and playful styling that makes local tool execution feel less serious than it is.

## Design Principles

- Keep governance visible: route, policy, approval, runner, and tool state should be inspectable without digging through logs.
- Prefer dense but calm operations UI: repeated workflows should be scannable, compact, and predictable.
- Preserve local trust boundaries: secrets, approvals, subprocesses, and workspace access need explicit states and clear feedback.
- Reuse existing API contracts and trace vocabulary so UI polish does not fork product behavior.
- Make degraded states useful: missing models, unreachable runtimes, empty history, and failed runs should point to the next concrete action.

## Accessibility & Inclusion

Use standard keyboard-operable controls, visible focus states, semantic form labels, readable contrast, and reduced-motion-safe transitions. The default target is WCAG AA for product UI.
