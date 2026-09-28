# models.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Incident(BaseModel):
    """Definition of a production incident."""
    id: str = Field(description="Unique identifier for the incident")
    description: str = Field(description="Detailed description of what went wrong")
    severity: str = Field(description="Severity level, e.g., P1, P2")


class ResponseTask(BaseModel):
    """A single task in the incident response plan."""
    id: str = Field(description="Unique identifier for the task")
    description: str = Field(description="Actionable step to remediate the incident")
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of task IDs that must be completed before this task can start"
    )


class ResponsePlan(BaseModel):
    """Complete response plan for an incident."""
    tasks: List[ResponseTask] = Field(
        description="List of tasks to execute in dependency order"
    )


class TaskOutcome(BaseModel):
    """Result of executing a single response task."""
    task_id: str = Field(description="ID of the task that was attempted")
    status: Literal["completed", "blocked"] = Field(
        description="Outcome status: 'completed' or 'blocked'"
    )
    reason: str = Field(description="Explanation of why it completed or got blocked")


class Evaluation(BaseModel):
    """Evaluation of the overall incident resolution."""
    status: Literal["resolved", "mitigated", "escalated", "unresolved"] = Field(
        description="Final resolution status"
    )
    reason: str = Field(description="Reasoning for the evaluation status")


class IncidentReport(BaseModel):
    """Final incident response report."""
    incident_id: str
    completed_tasks: List[ResponseTask] = Field(default_factory=list)
    blocked_tasks: List[ResponseTask] = Field(default_factory=list)
    evaluation: Optional[Evaluation] = None

    # Pydantic v2 configuration: ignore extra fields returned by the LLM
    # to prevent validation errors caused by unexpected fields.
    model_config = {"extra": "ignore"}