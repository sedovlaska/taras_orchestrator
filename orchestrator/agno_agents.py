from __future__ import annotations

import subprocess

import psutil
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.team import Team
from agno.tools import tool

from shared.config import settings

LANGUAGE_INSTRUCTION = "Reply in the same language as the user. If the user writes in Russian, reply in Russian."

AGNO_MEMBER_NAMES = ["code", "db", "devops", "docs", "system", "docker"]


def get_model() -> Ollama:
    return Ollama(id=settings.llm_model, host=settings.ollama_host)


def create_code_agent() -> Agent:
    @tool
    def analyze_code(code: str) -> str:
        """Analyze code quality and patterns."""
        return f"Code analysis: {code[:200]}..."

    @tool
    def lint_code(path: str = ".") -> str:
        """Run Ruff on a project path."""
        result = subprocess.run(["ruff", "check", path], capture_output=True, text=True, timeout=30)
        return result.stdout or result.stderr or "No issues found."

    @tool
    def generate_code(description: str) -> str:
        """Draft code from a short description."""
        return f"Generated code for: {description}"

    return Agent(
        name="code",
        model=get_model(),
        tools=[analyze_code, lint_code, generate_code],
        instructions=[
            "You are a code analysis and generation agent.",
            "Use tools to analyze, lint, or generate code.",
            "Be concise and technical.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_db_agent() -> Agent:
    @tool
    def run_query(sql: str) -> str:
        """Prepare a SQL query execution summary."""
        return f"Query execution is not connected yet. Requested SQL: {sql}"

    @tool
    def show_schema(db_name: str) -> str:
        """Show database schema placeholder."""
        return f"Schema for {db_name}: database connection is not configured yet."

    return Agent(
        name="db",
        model=get_model(),
        tools=[run_query, show_schema],
        instructions=[
            "You are a database agent.",
            "Use tools to explain database queries and schemas.",
            "Be explicit when a real database connection is not configured.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_devops_agent() -> Agent:
    @tool
    def run_tests(path: str = ".") -> str:
        """Run the test suite for a path."""
        result = subprocess.run(["pytest", path], capture_output=True, text=True, timeout=120)
        return result.stdout or result.stderr or "Tests completed without output."

    @tool
    def build_project() -> str:
        """Report project build status."""
        return "Build command is not configured yet."

    return Agent(
        name="devops",
        model=get_model(),
        tools=[run_tests, build_project],
        instructions=[
            "You are a DevOps agent for testing and building.",
            "Use tools to run tests and describe build status.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_docs_agent() -> Agent:
    return Agent(
        name="docs",
        model=get_model(),
        instructions=[
            "You are a documentation agent.",
            "When the user asks for documentation about a named subject, generate useful documentation immediately.",
            "Do not ask for source code unless the user explicitly wants documentation for an existing codebase.",
            "Use clear Markdown sections: Overview, Features, Usage, Examples, Error Handling, and Notes.",
            LANGUAGE_INSTRUCTION,
        ],
    )

def create_system_agent() -> Agent:
    @tool
    def get_system_info() -> str:
        """Get CPU and memory usage."""
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        used_mb = mem.used // (1024**2)
        total_mb = mem.total // (1024**2)
        return f"CPU: {cpu}%, Memory: {mem.percent}% ({used_mb}MB / {total_mb}MB)"

    @tool
    def list_processes() -> str:
        """List running processes."""
        procs = []
        for process in psutil.process_iter(["pid", "name", "cpu_percent"]):
            info = process.info
            procs.append(f"{info['pid']:>6}  {info['name']}")
        return "\n".join(procs[:30])

    @tool
    def get_disk_usage() -> str:
        """Get disk usage info."""
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
        model=get_model(),
        tools=[get_system_info, list_processes, get_disk_usage],
        instructions=[
            "You are a system monitoring agent.",
            "Use tools to check CPU, memory, disk, and processes.",
            "Report system status clearly.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_docker_agent() -> Agent:
    @tool
    def list_containers() -> str:
        """List Docker containers."""
        result = subprocess.run(["docker", "ps", "-a"], capture_output=True, text=True, timeout=10)
        return result.stdout or result.stderr or "No containers found."

    @tool
    def list_images() -> str:
        """List Docker images."""
        result = subprocess.run(["docker", "images"], capture_output=True, text=True, timeout=10)
        return result.stdout or result.stderr or "No images found."

    return Agent(
        name="docker",
        model=get_model(),
        tools=[list_containers, list_images],
        instructions=[
            "You are a Docker management agent.",
            "Use tools to inspect containers and images.",
            LANGUAGE_INSTRUCTION,
        ],
    )


def create_orchestrator() -> Team:
    return Team(
        name="orchestrator",
        mode="route",
        members=[
            create_code_agent(),
            create_db_agent(),
            create_devops_agent(),
            create_docs_agent(),
            create_system_agent(),
            create_docker_agent(),
        ],
        model=get_model(),
        show_members_responses=True,
        markdown=True,
        instructions=[
            "You are an AGNO Team orchestrator that routes requests to specialized agents.",
            "Route to the most appropriate agent based on the user's request.",
            "If the request is ambiguous, ask a concise clarification question.",
            LANGUAGE_INSTRUCTION,
        ],
    )
