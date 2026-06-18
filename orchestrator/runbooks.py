from __future__ import annotations

import re
from dataclasses import dataclass, field

PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


@dataclass(frozen=True)
class RunbookVariable:
    name: str
    label: str
    description: str
    required: bool = True
    default: str = ""
    multiline: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "multiline": self.multiline,
        }


@dataclass(frozen=True)
class Runbook:
    id: str
    title: str
    description: str
    category: str
    suggested_agent: str
    template: str
    variables: tuple[RunbookVariable, ...] = field(default_factory=tuple)

    def as_dict(self, include_template: bool = False) -> dict[str, object]:
        payload = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "suggested_agent": self.suggested_agent,
            "variables": [variable.as_dict() for variable in self.variables],
        }
        if include_template:
            payload["template"] = self.template
        return payload

    def render(self, values: dict[str, str] | None = None) -> str:
        values = values or {}
        resolved: dict[str, str] = {}
        for variable in self.variables:
            value = str(values.get(variable.name, variable.default)).strip()
            if variable.required and not value:
                raise ValueError(f"missing required runbook field: {variable.name}")
            resolved[variable.name] = value

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in resolved:
                raise ValueError(f"unknown runbook placeholder: {name}")
            return resolved[name]

        return PLACEHOLDER_RE.sub(replace, self.template).strip()


RUNBOOKS: tuple[Runbook, ...] = (
    Runbook(
        id="repo_onboarding",
        title="Repo onboarding",
        description="Map the project structure, runtime, extension points, and first contribution path.",
        category="research",
        suggested_agent="code",
        variables=(
            RunbookVariable(
                name="focus",
                label="Focus",
                description="Subsystem or question to emphasize.",
                default="architecture, setup, extension points, and contribution path",
            ),
            RunbookVariable(
                name="files",
                label="Files",
                description="Files or folders the agent should inspect first.",
                default="README.md, pyproject.toml, orchestrator/server.py, static/index.html",
            ),
        ),
        template=(
            "Onboard me to this repository with focus on {{focus}}.\n"
            "Start by inspecting these files with workspace tools: {{files}}.\n"
            "Return a practical map of the codebase, the main runtime flow, the highest-leverage "
            "extension points, and one concrete next contribution."
        ),
    ),
    Runbook(
        id="feature_plan",
        title="Feature plan",
        description="Turn a feature idea into a scoped implementation plan tied to existing files.",
        category="planning",
        suggested_agent="code",
        variables=(
            RunbookVariable(
                name="feature",
                label="Feature",
                description="Feature or capability to design.",
                default="",
            ),
            RunbookVariable(
                name="constraints",
                label="Constraints",
                description="Non-negotiable boundaries or risks.",
                default="keep changes small, preserve existing APIs, add tests",
                required=False,
                multiline=True,
            ),
            RunbookVariable(
                name="files",
                label="Files",
                description="Files or folders likely to change.",
                default="orchestrator/, shared/, static/index.html, tests/",
            ),
        ),
        template=(
            "Plan this feature: {{feature}}\n"
            "Constraints: {{constraints}}\n"
            "Inspect likely touchpoints first: {{files}}.\n"
            "Return a concrete implementation plan, data model/API changes, UI impact, tests, "
            "and the smallest safe PR boundary."
        ),
    ),
    Runbook(
        id="code_review",
        title="Code review",
        description="Review a branch, PR, file set, or diff with findings-first output.",
        category="review",
        suggested_agent="code",
        variables=(
            RunbookVariable(
                name="scope",
                label="Scope",
                description="Branch, PR, files, or diff to review.",
                default="current branch diff",
            ),
            RunbookVariable(
                name="focus",
                label="Focus",
                description="Risk area to prioritize.",
                default="bugs, regressions, missing tests, and unsafe behavior",
            ),
        ),
        template=(
            "Review {{scope}}. Prioritize {{focus}}.\n"
            "Use findings-first output with file and line references where possible. "
            "Keep summaries brief and separate from actionable findings."
        ),
    ),
    Runbook(
        id="test_failure_triage",
        title="Test failure triage",
        description="Analyze a failing command or log and propose the next discriminating fix.",
        category="debugging",
        suggested_agent="devops",
        variables=(
            RunbookVariable(
                name="command",
                label="Command",
                description="Command that failed.",
                default="python -m pytest",
            ),
            RunbookVariable(
                name="failure_log",
                label="Failure log",
                description="Relevant failure output.",
                default="",
                multiline=True,
            ),
        ),
        template=(
            "Triage this failing command: {{command}}\n\n"
            "Failure log:\n{{failure_log}}\n\n"
            "Identify the most likely root cause, the smallest fix, and one verification command."
        ),
    ),
    Runbook(
        id="docs_update",
        title="Docs update",
        description="Generate a documentation update plan from code and API changes.",
        category="documentation",
        suggested_agent="docs",
        variables=(
            RunbookVariable(
                name="topic",
                label="Topic",
                description="Feature or workflow to document.",
                default="",
            ),
            RunbookVariable(
                name="files",
                label="Files",
                description="Source files or docs to inspect.",
                default="README.md, .env.example, orchestrator/server.py",
            ),
        ),
        template=(
            "Update documentation for {{topic}}.\n"
            "Inspect these files first: {{files}}.\n"
            "Return the exact README/API/config text changes needed, plus any caveats users need."
        ),
    ),
)


def list_runbooks() -> list[dict[str, object]]:
    return [runbook.as_dict() for runbook in RUNBOOKS]


def get_runbook(runbook_id: str) -> Runbook:
    for runbook in RUNBOOKS:
        if runbook.id == runbook_id:
            return runbook
    raise KeyError(runbook_id)


def render_runbook(runbook_id: str, values: dict[str, str] | None = None) -> dict[str, object]:
    runbook = get_runbook(runbook_id)
    return {
        "runbook": runbook.as_dict(),
        "prompt": runbook.render(values),
    }
