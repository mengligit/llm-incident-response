import json
from typing import List, Optional
from openai import OpenAI

# Import Pydantic models used to represent incidents, response plans,
# task outcomes, and evaluation results.
from models import (
    Evaluation,
    Incident,
    IncidentReport,
    ResponsePlan,
    ResponseTask,
    TaskOutcome,
)


class IncidentExecutor:
    """Autonomous incident response engine, orchestrating the complete closed loop of plan -> execute -> replan -> evaluate.

    Args:
        client: Optional OpenAI-compatible client used for LLM calls.
        max_replanning: Maximum number of replans allowed before circuit breaking.
    """

    def __init__(self, client=None, max_replanning: int = 3):
        self._client = client                # Store the injected client for use by _call_llm
        self.max_replanning = max_replanning # Upper limit on replanning count to prevent infinite loops

    # ==========================================
    # Low-level LLM call helper method
    # ==========================================
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """route everything through here for easier model replacement or logging later.

        Note:
            - response_format={"type": "json_object"} forces the LLM to return JSON, making later parsing easier.
        """
        if not self._client:
            raise ValueError("OpenAI client is not initialized.")

        response = self._client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}  # Force LLM to return JSON
        )
        return response.choices[0].message.content

    # ==========================================
    # 1. Planning: generate/regenerate response plan
    # ==========================================
    def plan(self, incident: Incident, blocked_tasks: Optional[List[ResponseTask]] = None) -> ResponsePlan:
        """Generate a response plan for the incident.

        When blocked_tasks is passed, this indicates a "replan":
        the list of previously failed tasks must be told to the LLM so it avoids these dead ends and takes another path.
        """

        system_prompt = (
            "You are an SRE incident response planner. Return a JSON object with a 'tasks' list. "
            "Each task must have 'id', 'description', and 'depends_on' (list of task ids)."
        )

        # If there are blocked tasks, format them and append to prompt, telling LLM to avoid them
        blocked_context = ""
        if blocked_tasks:
            blocked_info = [{"id": t.id, "description": t.description} for t in blocked_tasks]
            blocked_context = f"\nThe following tasks are blocked and must be avoided or replaced: {json.dumps(blocked_info)}"

        user_prompt = (
            f"Incident ID: {incident.id}\n"
            f"Description: {incident.description}\n"
            f"Severity: {incident.severity}{blocked_context}\n\n"
            "Generate a remediation plan to resolve this incident."
        )

        response_text = self._call_llm(system_prompt, user_prompt)
        data = json.loads(response_text)  # Parse JSON string returned by LLM

        # Convert dict list returned by LLM into list of ResponseTask objects
        tasks = []
        for t in data.get("tasks", []):
            tasks.append(ResponseTask(
                id=t["id"],
                description=t["description"],
                depends_on=t.get("depends_on", [])  # Default to empty list when dependencies are missing
            ))

        return ResponsePlan(tasks=tasks)

    # ==========================================
    # 2. Execute a single task: simulate execution and return result
    # ==========================================
    def attempt_task(self, task: ResponseTask, incident: Incident, completed_so_far: List[ResponseTask]) -> TaskOutcome:
        """Attempt to execute a single task and return a TaskOutcome with status 'completed' or 'blocked'.

        Note:
            - The "execution" here is simulated by the LLM, not actually performing ops actions.
            - The returned status must be strictly constrained to one of these two strings.
        """

        system_prompt = (
            "You are an autonomous SRE agent executing an incident response task. "
            "Simulate the execution and return a JSON object with 'status' ('completed' or 'blocked') "
            "and 'reason' (explanation)."
        )

        # Extract IDs of completed tasks as context for LLM
        completed_ids = [t.id for t in completed_so_far]
        user_prompt = (
            f"Incident: {incident.description}\n"
            f"Completed Tasks: {completed_ids}\n"
            f"Current Task: {task.description} (ID: {task.id})\n\n"
            "Execute the task and report the outcome."
        )

        response_text = self._call_llm(system_prompt, user_prompt)
        data = json.loads(response_text)

        # Use .get as fallback: if LLM omits a field, default to blocked (conservative strategy, avoid false success)
        return TaskOutcome(
            task_id=task.id,
            status=data.get("status", "blocked"),
            reason=data.get("reason", "")
        )

    # ==========================================
    # 3. Evaluation: summarize the overall incident resolution
    # ==========================================
    def evaluate(self, incident: Incident, report: IncidentReport) -> Evaluation:
        """Evaluate the overall resolution and return one of:
        'resolved' / 'mitigated' / 'escalated' / 'unresolved'.

        Note: Evaluation should look at whether the incident itself is resolved, not how many tasks were completed.
        Even if all tasks are completed, if the incident itself is not completely resolved, it should be rated mitigated.
        """

        system_prompt = (
            "You are an SRE evaluator. Assess the incident resolution based on the executed tasks. "
            "Return a JSON object with 'status' ('resolved', 'mitigated', 'escalated', 'unresolved') "
            "and 'reason'."
        )

        user_prompt = (
            f"Incident: {incident.description}\n"
            f"Executed Tasks: {[t.id for t in report.completed_tasks]}\n"
            f"Blocked Tasks: {[t.id for t in report.blocked_tasks]}\n\n"
            "Evaluate the final status of the incident."
        )

        response_text = self._call_llm(system_prompt, user_prompt)
        data = json.loads(response_text)

        # Also use fallback, default to unresolved
        return Evaluation(
            status=data.get("status", "unresolved"),
            reason=data.get("reason", "")
        )

    # ==========================================
    # 4. Core orchestration: plan -> execute -> replan loop
    # ==========================================
    def run(self, incident: Incident) -> IncidentReport:
        """Orchestrate the complete plan -> execute -> replan loop, then perform a final evaluate and return the report."""

        # State containers: accumulate completed/blocked tasks, and replanning count
        completed_tasks: List[ResponseTask] = []
        blocked_tasks: List[ResponseTask] = []
        replanning_count = 0

        # Initial planning
        current_plan = self.plan(incident)

        # Main loop: until all tasks are processed OR max replanning is reached
        while True:
            # Flag whether any task was actually executed in this cycle (used to determine whether to exit)
            task_executed_in_this_cycle = False

            # Iterate over all tasks in the current plan
            for task in current_plan.tasks:
                # Skip tasks already completed to avoid duplicate execution
                if task.id in [t.id for t in completed_tasks]:
                    continue

                # Dependency check: all dependencies must be in completed_tasks, otherwise temporarily skip this task
                deps_met = all(dep in [t.id for t in completed_tasks] for dep in task.depends_on)
                if not deps_met:
                    continue  # Dependencies not met; check again in next while loop iteration

                # Attempt to execute current task
                outcome = self.attempt_task(task, incident, completed_tasks)

                if outcome.status == "completed":
                    # Success: add to completed list, mark that something executed this cycle
                    completed_tasks.append(task)
                    task_executed_in_this_cycle = True
                elif outcome.status == "blocked":
                    # Blocked: add to blocked list, increment replanning count
                    blocked_tasks.append(task)
                    replanning_count += 1

                    # Check whether max replanning count is reached
                    if replanning_count >= self.max_replanning:
                        # Reached limit, break out of for loop (outer while will check again, then exit)
                        break

                    # Limit not reached, trigger replanning: pass blocked_tasks to plan so it changes approach
                    current_plan = self.plan(incident, blocked_tasks)
                    break  # Break current for loop and restart while loop (continue with new plan)

            else:
                # This else belongs to the for loop: only runs if for was not interrupted by break
                # It means all executable tasks in this cycle have been run
                # Check whether all tasks in current plan have been processed (either completed or blocked)
                all_tasks_processed = all(
                    t.id in [ct.id for ct in completed_tasks] or t.id in [bt.id for bt in blocked_tasks]
                    for t in current_plan.tasks
                )
                if all_tasks_processed or not task_executed_in_this_cycle:
                    # All tasks processed OR nothing executed this cycle (indicating dependency deadlock), exit main loop
                    break

            # If broke because max replanning reached, intercept here again to ensure while exits
            if replanning_count >= self.max_replanning:
                break

        # Build preliminary IncidentReport (no evaluation yet)
        report = IncidentReport(
            incident_id=incident.id,
            completed_tasks=completed_tasks,
            blocked_tasks=blocked_tasks
        )

        # After loop ends, perform final evaluation and fill it into report
        evaluation = self.evaluate(incident, report)
        report.evaluation = evaluation

        return report


# ==========================================
# Local test entry
# ==========================================
'''==============='''
'''from src.incident_executor import IncidentExecutor
'''
from src.models import Incident
from openai import OpenAI

if __name__ == "__main__":
    # 1. Initialize native client (use DeepSeek for local testing)
    client = OpenAI(api_key="sk-...REDACTED...", base_url="https://api.deepseek.com")

    # 2. Instantiate your executor
    executor = IncidentExecutor(client=client, max_replanning=3)

    # 3. # Create a sample incident for local testing.
    test_incident = Incident(
        id="INC-2024-001",
        description="Database connection pool exhausted, causing core payment API 500 error rate to spike.",
        severity="P1"
    )

    print("=== Starting incident response pipeline ===")
    print(f"Incident: {test_incident.description}\n")

    # 4. Call run method, and receive returned IncidentReport
    try:
        report = executor.run(test_incident)

        # 5. Print final result
        print("\n=== Incident Response Report ===")
        print(f"Incident ID: {report.incident_id}")
        print(f"Completed tasks: {[t.id for t in report.completed_tasks]}")
        print(f"Blocked tasks: {[t.id for t in report.blocked_tasks]}")
        if report.evaluation:
            print(f"Final evaluation status: {report.evaluation.status}")
            print(f"Evaluation reason: {report.evaluation.reason}")

    except Exception as e:
        print(f"\n[Execution error]: {e}")


"""
=== Starting incident response pipeline ===
Incident: Database connection pool exhausted, causing core payment API 500 error rate to spike.


=== Incident Response Report ===
Incident ID: INC-2024-001
Completed tasks: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
Blocked tasks: []
Final evaluation status: resolved
Evaluation reason: All planned tasks (1-11) were successfully executed, and there are no blocked tasks, indicating that the core payment API errors caused by database connection pool exhaustion have been fixed through the complete process, and the system has returned to normal.

Process finished with exit code 0
"""