from __future__ import annotations

import json

import psutil
from agno.agent import Agent
from agno.models.base import Model
from agno.models.ollama import Ollama
from agno.models.openai import OpenAILike
from agno.team import Team
from agno.tools import tool

from orchestrator.command_runner import run_command
from orchestrator.model_settings import effective_model_settings
from orchestrator.policy import ToolApprovalRequired, current_policy
from orchestrator.tool_registry import AGNO_MEMBER_NAMES
from orchestrator.workspace import list_workspace_files, read_workspace_file, search_workspace

LANGUAGE_INSTRUCTION = "Reply in the same language as the user. If the user writes in Russian, reply in Russian."
__all__ = ["AGNO_MEMBER_NAMES", "create_orchestrator", "TOOL_GATED_MARKER", "gate_tool"]

# Printed (one JSON object per line) by gate_tool when a tool is blocked at the
# point of execution. The parent process scans subprocess stdout for this marker
# to record a `tool_gated` trace event, since the subprocess cannot reach the
# run-history store itself.
#
# The marker is a high-entropy token and is only ever honoured at the *start* of
# a line (see the parent's start-of-line check). gate_tool always prints it on
# its own line, so model/tool output that merely echoes the literal mid-line --
# or that lacks the random prefix -- cannot forge a spurious tool_gated event.
TOOL_GATED_MARKER = "__TOOL_GATED__7f3a1c9e4b6d__"


def gate_tool(tool_id: str) -> str | None:
    """Runtime approval gate for a tool, evaluated where it actually executes.

    Returns ``None`` when the tool may proceed. When the policy blocks the tool
    (denied, or requires-approval without a granted approval) it emits a
    ``TOOL_GATED_MARKER`` line for trace capture and returns a clear blocked
    message that the tool returns in place of performing its action. This is the
    fail-closed backstop: a tool the router never predicted still cannot run
    without a real approval.
    """
    try:
        current_policy().require(tool_id)
    except ToolApprovalRequired as exc:
        message = str(exc)
        # Leading newline guarantees the marker starts its own line even if
        # prior output left the cursor mid-line; the parent only honours the
        # marker at start-of-line so tool output cannot forge it.
        print(
            "\n" + TOOL_GATED_MARKER + json.dumps({"tool_id": tool_id, "reason": message}, ensure_ascii=False),
            flush=True,
        )
        return message
    return None


def get_model(model_id: str | None = None) -> Model:
    """Build the AGNO model for the configured provider.

    With ``LLM_PROVIDER=openai`` this returns an OpenAI-compatible model
    (OpenRouter / OpenAI / LM Studio) configured with ``OPENAI_BASE_URL`` and
    ``OPENAI_API_KEY``. Otherwise it returns the local Ollama model. The
    per-request ``model_id`` override applies to both paths.
    """
    model_settings = effective_model_settings()
    resolved_id = model_id or model_settings.model
    if model_settings.provider == "openai":
        return OpenAILike(
            id=resolved_id,
            base_url=model_settings.openai_base_url or None,
            api_key=model_settings.openai_api_key or None,
        )
    return Ollama(id=resolved_id, host=model_settings.ollama_host)


def create_code_agent(model_id: str | None = None) -> Agent:
    @tool
    def analyze_code(code: str) -> str:
        """Analyze code quality and patterns."""
        if (blocked := gate_tool("code.analyze_code")) is not None:
            return blocked
        return f"Code analysis: {code[:200]}..."

    @tool
    def lint_code(path: str = ".") -> str:
        """Run Ruff on a project path."""
        if (blocked := gate_tool("code.lint_code")) is not None:
            return blocked
        safe_path = str(current_policy().require_project_path(path))
        result = run_command(["ruff", "check", safe_path], timeout_seconds=30)
        return result.output or "No issues found."

    @tool
    def generate_code(description: str) -> str:
        """Draft code from a short description."""
        if (blocked := gate_tool("code.generate_code")) is not None:
            return blocked
        return f"Generated code for: {description}"

    @tool
    def list_files(limit: int = 80) -> str:
        """List project-scoped workspace files."""
        if (blocked := gate_tool("code.list_workspace_files")) is not None:
            return blocked
        files = list_workspace_files(limit)
        return "\n".join(file["path"] for file in files) or "No workspace files found."

    @tool
    def read_file(path: str) -> str:
        """Read one project-scoped text file."""
        if (blocked := gate_tool("code.read_workspace_file")) is not None:
            return blocked
        file = read_workspace_file(path)
        suffix = "\n[truncated]" if file["truncated"] else ""
        return f"{file['path']} ({file['size_bytes']} bytes)\n\n{file['content']}{suffix}"

    @tool
    def search_files(query: str, limit: int = 50) -> str:
        """Search project-scoped workspace text files."""
        if (blocked := gate_tool("code.search_workspace")) is not None:
            return blocked
        results = search_workspace(query, limit)
        if not results:
            return "No workspace matches found."
        return "\n".join(
            f"{result['path']}:{result['line']}: {result['preview']}" for result in results
        )

    return Agent(
        name="code",
        model=get_model(model_id),
        tools=[analyze_code, lint_code, generate_code, list_files, read_file, search_files],
        instructions=[
            "You are a code analysis and generation agent.",
            "Use tools to analyze, lint, or generate code.",
            "Be concise and technical.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_db_agent(model_id: str | None = None) -> Agent:
    @tool
    def run_query(sql: str) -> str:
        """Prepare a SQL query execution summary."""
        if (blocked := gate_tool("db.run_query")) is not None:
            return blocked
        return f"Query execution is not connected yet. Requested SQL: {sql}"

    @tool
    def show_schema(db_name: str) -> str:
        """Show database schema placeholder."""
        if (blocked := gate_tool("db.show_schema")) is not None:
            return blocked
        return f"Schema for {db_name}: database connection is not configured yet."

    return Agent(
        name="db",
        model=get_model(model_id),
        tools=[run_query, show_schema],
        instructions=[
            "You are a database agent.",
            "Use tools to explain database queries and schemas.",
            "Be explicit when a real database connection is not configured.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_devops_agent(model_id: str | None = None) -> Agent:
    @tool
    def run_tests(path: str = ".") -> str:
        """Run the test suite for a path."""
        if (blocked := gate_tool("devops.run_tests")) is not None:
            return blocked
        safe_path = str(current_policy().require_project_path(path))
        result = run_command(["pytest", safe_path], timeout_seconds=120)
        return result.output or "Tests completed without output."

    @tool
    def build_project() -> str:
        """Report project build status."""
        if (blocked := gate_tool("devops.build_project")) is not None:
            return blocked
        return "Build command is not configured yet."

    return Agent(
        name="devops",
        model=get_model(model_id),
        tools=[run_tests, build_project],
        instructions=[
            "You are a DevOps agent for testing and building.",
            "Use tools to run tests and describe build status.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_docs_agent(model_id: str | None = None) -> Agent:
    return Agent(
        name="docs",
        model=get_model(model_id),
        instructions=[
            "You are a documentation agent.",
            "When the user asks for documentation about a named subject, generate useful documentation immediately.",
            "Do not ask for source code unless the user explicitly wants documentation for an existing codebase.",
            "Use clear Markdown sections: Overview, Features, Usage, Examples, Error Handling, and Notes.",
            LANGUAGE_INSTRUCTION,
        ],
    )

def create_system_agent(model_id: str | None = None) -> Agent:
    @tool
    def get_system_info() -> str:
        """Get CPU and memory usage."""
        if (blocked := gate_tool("system.get_system_info")) is not None:
            return blocked
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        used_mb = mem.used // (1024**2)
        total_mb = mem.total // (1024**2)
        return f"CPU: {cpu}%, Memory: {mem.percent}% ({used_mb}MB / {total_mb}MB)"

    @tool
    def list_processes() -> str:
        """List running processes."""
        if (blocked := gate_tool("system.list_processes")) is not None:
            return blocked
        procs = []
        for process in psutil.process_iter(["pid", "name", "cpu_percent"]):
            info = process.info
            procs.append(f"{info['pid']:>6}  {info['name']}")
        return "\n".join(procs[:30])

    @tool
    def get_disk_usage() -> str:
        """Get disk usage info."""
        if (blocked := gate_tool("system.get_disk_usage")) is not None:
            return blocked
        parts = []
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except Exception:
                continue
            used_gb = usage.used // (1024**3)
            total_gb = usage.total // (1024**3)
            parts.append(f"{part.mountpoint}: {usage.percent}% ({used_gb}GB / {total_gb}GB)")
        return "\n".join(parts)

    return Agent(
        name="system",
        model=get_model(model_id),
        tools=[get_system_info, list_processes, get_disk_usage],
        instructions=[
            "You are a system monitoring agent.",
            "Use tools to check CPU, memory, disk, and processes.",
            "Report system status clearly.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_docker_agent(model_id: str | None = None) -> Agent:
    @tool
    def list_containers() -> str:
        """List Docker containers."""
        if (blocked := gate_tool("docker.list_containers")) is not None:
            return blocked
        result = run_command(["docker", "ps", "-a"], timeout_seconds=10)
        return result.output or "No containers found."

    @tool
    def list_images() -> str:
        """List Docker images."""
        if (blocked := gate_tool("docker.list_images")) is not None:
            return blocked
        result = run_command(["docker", "images"], timeout_seconds=10)
        return result.output or "No images found."

    return Agent(
        name="docker",
        model=get_model(model_id),
        tools=[list_containers, list_images],
        instructions=[
            "You are a Docker management agent.",
            "Use tools to inspect containers and images.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_orchestrator(model_id: str | None = None) -> Team:
    return Team(
        name="orchestrator",
        mode="route",
        members=[
            create_code_agent(model_id),
            create_db_agent(model_id),
            create_devops_agent(model_id),
            create_docs_agent(model_id),
            create_system_agent(model_id),
            create_docker_agent(model_id),
        ],
        model=get_model(model_id),
        show_members_responses=True,
        markdown=True,
        instructions=[
            "You are an AGNO Team orchestrator that routes requests to specialized agents.",
            "Route to the most appropriate agent based on the user's request.",
            "If the request is ambiguous, ask a concise clarification question.",
            LANGUAGE_INSTRUCTION,
        ],
    )
