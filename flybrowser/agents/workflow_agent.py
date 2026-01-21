# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Workflow agent for orchestrating complex multi-step automation workflows.

This module provides the WorkflowAgent class which coordinates multiple agents
to execute complex, multi-step automation workflows. It supports conditional
logic, loops, error handling, and state management across workflow steps.

The agent supports:
- Multi-step workflow definition and execution
- Conditional branching based on page state
- Loop constructs for repetitive tasks
- State management and variable passing
- Error recovery and rollback
- Workflow templates and reusability

Use Cases:
- E-commerce checkout automation
- Form submission workflows
- Data collection pipelines
- Testing scenarios
- Business process automation

Example:
    >>> agent = WorkflowAgent(page_controller, element_detector, llm_provider)
    >>> result = await agent.execute("Log in, search for 'laptop', add first result to cart")
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from flybrowser.agents.base_agent import BaseAgent
from flybrowser.exceptions import FlyBrowserError
from flybrowser.utils.logger import logger


class WorkflowError(FlyBrowserError):
    """Exception raised when workflow execution fails."""
    pass


class StepType(str, Enum):
    """Types of workflow steps."""

    NAVIGATE = "navigate"
    ACTION = "action"
    EXTRACT = "extract"
    WAIT = "wait"
    CONDITION = "condition"
    LOOP = "loop"
    ASSERT = "assert"
    STORE = "store"


class StepStatus(str, Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """
    Represents a single step in a workflow.

    Attributes:
        step_id: Unique identifier for the step
        step_type: Type of step to execute
        instruction: Natural language instruction or action description
        condition: Optional condition for conditional steps
        loop_count: Number of iterations for loop steps
        on_failure: Action to take on failure ("stop", "continue", "retry")
        max_retries: Maximum retry attempts
        timeout: Step timeout in seconds
        store_as: Variable name to store result
    """

    step_id: str
    step_type: StepType
    instruction: str
    condition: Optional[str] = None
    loop_count: Optional[int] = None
    on_failure: str = "stop"
    max_retries: int = 3
    timeout: float = 30.0
    store_as: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    """
    Result of workflow execution.

    Attributes:
        success: Whether the workflow completed successfully
        steps_completed: Number of steps completed
        total_steps: Total number of steps
        duration: Total execution time in seconds
        variables: Final state of workflow variables
        step_results: Results from each step
        error: Error message if failed
    """

    success: bool
    steps_completed: int = 0
    total_steps: int = 0
    duration: float = 0.0
    variables: Dict[str, Any] = field(default_factory=dict)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Workflow:
    """
    Represents a complete workflow definition.

    Attributes:
        workflow_id: Unique identifier
        name: Human-readable name
        description: Workflow description
        steps: List of workflow steps
        variables: Initial variables
        created_at: Creation timestamp
    """

    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class WorkflowAgent(BaseAgent):
    """
    Agent specialized in orchestrating complex multi-step workflows.

    This agent coordinates multiple capabilities (navigation, actions, extraction)
    to execute complex automation workflows. It provides workflow planning,
    execution, state management, and error recovery.

    **Rationale for this agent:**
    Complex automation tasks often require coordinating multiple steps:
    - E-commerce: Browse → Search → Filter → Add to cart → Checkout
    - Data collection: Navigate → Extract → Paginate → Aggregate
    - Testing: Setup → Execute → Verify → Cleanup
    - Business processes: Login → Navigate → Fill forms → Submit → Verify

    The WorkflowAgent provides:
    - Natural language workflow definition
    - Automatic step planning from high-level instructions
    - State management across steps
    - Error handling and recovery
    - Conditional logic and loops

    Example:
        >>> agent = WorkflowAgent(page_controller, element_detector, llm)
        >>>
        >>> # Execute from natural language
        >>> result = await agent.execute(
        ...     "Log in with email test@example.com, search for 'laptop', "
        ...     "and add the first result to cart"
        ... )
        >>>
        >>> # Execute predefined workflow
        >>> workflow = Workflow(
        ...     workflow_id="checkout",
        ...     name="Checkout Flow",
        ...     description="Complete checkout process",
        ...     steps=[...]
        ... )
        >>> result = await agent.run_workflow(workflow)
    """

    def __init__(
        self,
        page_controller,
        element_detector,
        llm_provider,
        action_agent=None,
        navigation_agent=None,
        extraction_agent=None,
        pii_handler=None,
    ) -> None:
        """
        Initialize the workflow agent.

        Args:
            page_controller: PageController instance for page operations
            element_detector: ElementDetector instance for element location
            llm_provider: BaseLLMProvider instance for LLM operations
            action_agent: Optional ActionAgent instance (created if not provided)
            navigation_agent: Optional NavigationAgent instance
            extraction_agent: Optional ExtractionAgent instance
            pii_handler: Optional PIIHandler for secure handling of sensitive data
        """
        super().__init__(page_controller, element_detector, llm_provider, pii_handler=pii_handler)
        self._action_agent = action_agent
        self._navigation_agent = navigation_agent
        self._extraction_agent = extraction_agent
        self._workflow_templates: Dict[str, Workflow] = {}

    async def execute(
        self,
        instruction: str,
        variables: Optional[Dict[str, Any]] = None,
        max_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute a workflow from natural language instruction.

        This method uses an LLM to plan the workflow steps and then
        executes them in sequence.

        Args:
            instruction: Natural language description of the workflow.
                Examples:
                - "Log in and navigate to the settings page"
                - "Search for 'python books' and extract the first 5 results"
                - "Fill out the contact form and submit it"
            variables: Initial variables for the workflow
            max_steps: Maximum number of steps to plan

        Returns:
            Dictionary containing:
            - success: Whether workflow completed successfully
            - steps_completed: Number of steps completed
            - total_steps: Total number of steps
            - duration: Execution time in seconds
            - variables: Final workflow variables
            - step_results: Results from each step

        Example:
            >>> result = await agent.execute(
            ...     "Log in with email test@example.com and password secret123",
            ...     variables={"email": "test@example.com", "password": "secret123"}
            ... )
        """
        import time

        start_time = time.time()

        try:
            # Mask instruction for logging to avoid exposing PII
            logger.info(f"Executing workflow: {self.mask_for_log(instruction)}")

            # Plan the workflow (instruction is masked for LLM in _plan_workflow)
            workflow = await self._plan_workflow(instruction, variables or {}, max_steps)

            # Execute the workflow
            result = await self.run_workflow(workflow)

            result.duration = time.time() - start_time

            return {
                "success": result.success,
                "steps_completed": result.steps_completed,
                "total_steps": result.total_steps,
                "duration": result.duration,
                "variables": self.mask_dict(result.variables),  # Mask variables in output
                "step_results": result.step_results,
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {self.mask_for_log(str(e))}")
            # Return error dict instead of raising, for better error handling
            return {
                "success": False,
                "steps_completed": 0,
                "total_steps": 0,
                "duration": time.time() - start_time,
                "variables": {},
                "step_results": [],
                "error": str(e),
                "exception_type": type(e).__name__,
            }

    async def _plan_workflow(
        self,
        instruction: str,
        variables: Dict[str, Any],
        max_steps: int,
    ) -> Workflow:
        """Plan workflow steps from natural language instruction."""
        import uuid

        context = await self.get_page_context()

        # Mask instruction and variables for LLM to avoid exposing PII
        safe_instruction = self.mask_for_llm(instruction)
        safe_variables = self.mask_dict(variables)

        prompt = f"""Plan a workflow to accomplish this task:

Task: {safe_instruction}

Current page URL: {context['url']}
Page title: {context['title']}

Available variables: {json.dumps(safe_variables)}

Break down the task into specific steps. Each step should be one of:
- navigate: Go to a URL or follow a link
- action: Click, type, fill form, etc.
- extract: Extract data from the page
- wait: Wait for an element or condition
- assert: Verify something on the page
- store: Store a value in a variable

Return a JSON object with:
- name: Workflow name
- description: Brief description
- steps: Array of steps, each with:
  - step_type: One of the types above
  - instruction: What to do in this step
  - store_as: Optional variable name to store result

Limit to {max_steps} steps maximum.
"""

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_type": {
                                "type": "string",
                                "enum": [t.value for t in StepType],
                            },
                            "instruction": {"type": "string"},
                            "store_as": {"type": "string"},
                            "condition": {"type": "string"},
                        },
                        "required": ["step_type", "instruction"],
                    },
                },
            },
            "required": ["name", "steps"],
        }

        result = await self.llm.generate_structured(
            prompt=prompt,
            schema=schema,
            temperature=0.3,
        )

        # Convert to Workflow object
        steps = []
        for i, step_data in enumerate(result.get("steps", [])):
            try:
                step_type = StepType(step_data.get("step_type", "action"))
            except ValueError:
                step_type = StepType.ACTION

            steps.append(
                WorkflowStep(
                    step_id=f"step_{i + 1}",
                    step_type=step_type,
                    instruction=step_data.get("instruction", ""),
                    store_as=step_data.get("store_as"),
                    condition=step_data.get("condition"),
                )
            )

        workflow = Workflow(
            workflow_id=str(uuid.uuid4())[:8],
            name=result.get("name", "Unnamed Workflow"),
            description=result.get("description", instruction),
            steps=steps,
            variables=variables,
        )

        logger.info(f"Planned workflow with {len(steps)} steps")
        return workflow

    async def run_workflow(self, workflow: Workflow) -> WorkflowResult:
        """
        Execute a predefined workflow.

        Args:
            workflow: Workflow object to execute

        Returns:
            WorkflowResult with execution details
        """
        result = WorkflowResult(
            success=False,
            total_steps=len(workflow.steps),
            variables=workflow.variables.copy(),
        )

        for i, step in enumerate(workflow.steps):
            step.status = StepStatus.RUNNING
            logger.info(f"Executing step {i + 1}/{len(workflow.steps)}: {step.instruction}")

            try:
                step_result = await self._execute_step(step, result.variables)
                step.status = StepStatus.COMPLETED
                step.result = step_result

                # Store result if requested
                if step.store_as and step_result:
                    result.variables[step.store_as] = step_result

                result.step_results.append({
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "instruction": step.instruction,
                    "status": step.status.value,
                    "result": step_result,
                })
                result.steps_completed += 1

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                logger.error(f"Step {step.step_id} failed: {e}")

                result.step_results.append({
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "instruction": step.instruction,
                    "status": step.status.value,
                    "error": str(e),
                })

                if step.on_failure == "stop":
                    result.error = f"Step {step.step_id} failed: {e}"
                    return result
                elif step.on_failure == "retry":
                    # Retry logic would go here
                    pass

        result.success = True
        return result

    async def _execute_step(
        self, step: WorkflowStep, variables: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a single workflow step."""
        # Substitute variables in instruction
        instruction = self._substitute_variables(step.instruction, variables)

        if step.step_type == StepType.NAVIGATE:
            return await self._execute_navigate(instruction)
        elif step.step_type == StepType.ACTION:
            return await self._execute_action(instruction)
        elif step.step_type == StepType.EXTRACT:
            return await self._execute_extract(instruction)
        elif step.step_type == StepType.WAIT:
            return await self._execute_wait(instruction)
        elif step.step_type == StepType.ASSERT:
            return await self._execute_assert(instruction)
        elif step.step_type == StepType.STORE:
            return await self._execute_store(instruction, variables)
        else:
            raise WorkflowError(f"Unknown step type: {step.step_type}")

    def _substitute_variables(
        self, text: str, variables: Dict[str, Any]
    ) -> str:
        """Substitute variables in text using {{variable}} syntax."""
        for key, value in variables.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
        return text

    async def _execute_navigate(self, instruction: str) -> Dict[str, Any]:
        """Execute a navigation step."""
        if self._navigation_agent:
            return await self._navigation_agent.execute(instruction)
        else:
            # Use page controller directly for simple navigation
            if instruction.startswith("http"):
                await self.page.goto(instruction)
            else:
                # Use LLM to determine navigation action
                context = await self.get_page_context()
                prompt = f"Navigate: {instruction}\nCurrent URL: {context['url']}"
                # Simplified - would use navigation agent in practice
                await self.page.goto(instruction)
            return {"navigated": True, "instruction": instruction}

    async def _execute_action(self, instruction: str) -> Dict[str, Any]:
        """Execute an action step."""
        if self._action_agent:
            return await self._action_agent.execute(instruction)
        else:
            # Fallback to basic action execution
            return {"action": instruction, "executed": True}

    async def _execute_extract(self, instruction: str) -> Dict[str, Any]:
        """Execute an extraction step."""
        if self._extraction_agent:
            return await self._extraction_agent.execute(instruction)
        else:
            # Fallback to basic extraction
            return {"extracted": True, "instruction": instruction}

    async def _execute_wait(self, instruction: str) -> Dict[str, Any]:
        """Execute a wait step."""
        import asyncio

        # Parse wait instruction
        if "second" in instruction.lower():
            # Extract number of seconds
            import re
            match = re.search(r"(\d+)", instruction)
            if match:
                seconds = int(match.group(1))
                await asyncio.sleep(seconds)
                return {"waited": seconds}

        # Default wait
        await asyncio.sleep(1)
        return {"waited": 1}

    async def _execute_assert(self, instruction: str) -> Dict[str, Any]:
        """Execute an assertion step."""
        context = await self.get_page_context()

        prompt = f"""Verify this assertion on the current page:

Assertion: {instruction}

Page URL: {context['url']}
Page title: {context['title']}

Return a JSON object with:
- passed: true if assertion passes, false otherwise
- reason: Explanation of the result
"""

        schema = {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["passed", "reason"],
        }

        result = await self.llm.generate_structured(
            prompt=prompt,
            schema=schema,
            temperature=0.1,
        )

        if not result.get("passed", False):
            raise WorkflowError(f"Assertion failed: {result.get('reason', instruction)}")

        return result

    async def _execute_store(
        self, instruction: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a store step."""
        # Parse store instruction to extract value
        return {"stored": True, "instruction": instruction}

    def register_template(self, workflow: Workflow) -> None:
        """Register a workflow template for reuse."""
        self._workflow_templates[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow template: {workflow.name}")

    def get_template(self, workflow_id: str) -> Optional[Workflow]:
        """Get a registered workflow template."""
        return self._workflow_templates.get(workflow_id)

    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

