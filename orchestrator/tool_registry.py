from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ToolRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    agent: str
    description: str
    risk: ToolRisk
    timeout_seconds: int | None = None
    args: dict[str, str] = field(default_factory=dict)
    touches_filesystem: bool = False
    touches_processes: bool = False
    touches_network: bool = False
    touches_docker: bool = False
    command: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.agent}.{self.name}"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "agent": self.agent,
            "description": self.description,
            "risk": self.risk.value,
            "timeout_seconds": self.timeout_seconds,
            "args": self.args,
            "touches_filesystem": self.touches_filesystem,
            "touches_processes": self.touches_processes,
            "touches_network": self.touches_network,
            "touches_docker": self.touches_docker,
            "command": list(self.command),
        }


AGNO_MEMBER_NAMES = ["code", "db", "devops", "docs", "system", "docker"]

TOOLS = {
    "code.analyze_code": ToolSpec(
        name="analyze_code",
        agent="code",
        description="Analyze pasted code quality and patterns.",
        risk=ToolRisk.LOW,
        args={"code": "Source text to inspect."},
    ),
    "code.lint_code": ToolSpec(
        name="lint_code",
        agent="code",
        description="Run Ruff checks against a project path.",
        risk=ToolRisk.MEDIUM,
        timeout_seconds=30,
        args={"path": "Project-relative path to lint."},
        touches_filesystem=True,
        touches_processes=True,
        command=("ruff", "check"),
    ),
    "code.generate_code": ToolSpec(
        name="generate_code",
        agent="code",
        description="Draft code from a short description.",
        risk=ToolRisk.LOW,
        args={"description": "Requested implementation summary."},
    ),
    "db.run_query": ToolSpec(
        name="run_query",
        agent="db",
        description="Prepare a SQL query execution summary. No database is connected.",
        risk=ToolRisk.LOW,
        args={"sql": "SQL text to explain."},
    ),
    "db.show_schema": ToolSpec(
        name="show_schema",
        agent="db",
        description="Show database schema placeholder. No database is connected.",
        risk=ToolRisk.LOW,
        args={"db_name": "Logical database name."},
    ),
    "devops.run_tests": ToolSpec(
        name="run_tests",
        agent="devops",
        description="Run pytest for a project path.",
        risk=ToolRisk.MEDIUM,
        timeout_seconds=120,
        args={"path": "Project-relative path to test."},
        touches_filesystem=True,
        touches_processes=True,
        command=("pytest",),
    ),
    "devops.build_project": ToolSpec(
        name="build_project",
        agent="devops",
        description="Report project build status placeholder.",
        risk=ToolRisk.LOW,
    ),
    "system.get_system_info": ToolSpec(
        name="get_system_info",
        agent="system",
        description="Read CPU and memory usage.",
        risk=ToolRisk.LOW,
        timeout_seconds=5,
    ),
    "system.list_processes": ToolSpec(
        name="list_processes",
        agent="system",
        description="List local process IDs and names.",
        risk=ToolRisk.HIGH,
        timeout_seconds=10,
        touches_processes=True,
    ),
    "system.get_disk_usage": ToolSpec(
        name="get_disk_usage",
        agent="system",
        description="Read local disk usage.",
        risk=ToolRisk.LOW,
        timeout_seconds=10,
        touches_filesystem=True,
    ),
    "docker.list_containers": ToolSpec(
        name="list_containers",
        agent="docker",
        description="Run docker ps -a.",
        risk=ToolRisk.HIGH,
        timeout_seconds=10,
        touches_processes=True,
        touches_docker=True,
        command=("docker", "ps", "-a"),
    ),
    "docker.list_images": ToolSpec(
        name="list_images",
        agent="docker",
        description="Run docker images.",
        risk=ToolRisk.HIGH,
        timeout_seconds=10,
        touches_processes=True,
        touches_docker=True,
        command=("docker", "images"),
    ),
}


def get_tool(tool_id: str) -> ToolSpec:
    return TOOLS[tool_id]


def tools_for_agent(agent: str) -> list[ToolSpec]:
    return [tool for tool in TOOLS.values() if tool.agent == agent]


def tool_inventory() -> list[dict]:
    return [tool.as_dict() for tool in TOOLS.values()]
