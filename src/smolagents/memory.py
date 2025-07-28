import inspect
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Type

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


__all__ = ["AgentMemory"]


logger = getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }

    @classmethod
    def from_dict(cls, d):
        # Handle both dataclass-style and .dict() style
        if "function" in d:
            return cls(
                name=d["function"]["name"],
                arguments=d["function"]["arguments"],
                id=d["id"],
            )
        # fallback to dataclass-style
        return cls(
            name=d["name"],
            arguments=d["arguments"],
            id=d["id"],
        )


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    step_number: int
    timing: Timing
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | list[dict[str, Any]] | None = None
    code_action: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": [msg.dict() for msg in self.model_input_messages] if self.model_input_messages else None,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": self.model_output_message.dict() if self.model_output_message else None,
            "model_output": self.model_output,
            "code_action": self.code_action,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages

    @classmethod
    def from_dict(cls, d):
        # Properly reconstruct nested fields from dicts
        def timing_from_dict(timing_dict):
            if isinstance(timing_dict, dict):
                # Only keep keys that Timing accepts
                allowed = {"start_time", "end_time"}
                filtered = {k: v for k, v in timing_dict.items() if k in allowed}
                return Timing(**filtered)
            return timing_dict

        def token_usage_from_dict(token_usage_dict):
            if isinstance(token_usage_dict, dict):
                allowed = {"input_tokens", "output_tokens"}
                filtered = {k: v for k, v in token_usage_dict.items() if k in allowed}
                return TokenUsage(**filtered)
            return token_usage_dict

        timing = timing_from_dict(d["timing"])
        model_input_messages = [ChatMessage.from_dict(msg) if isinstance(msg, dict) else msg for msg in d.get("model_input_messages") or []]
        # Use ToolCall.from_dict for each tool call
        tool_calls = [ToolCall.from_dict(tc) if isinstance(tc, dict) else tc for tc in d.get("tool_calls") or []]
        error = AgentError(**d["error"]) if d.get("error") and isinstance(d["error"], dict) else d.get("error")
        model_output_message = ChatMessage.from_dict(d["model_output_message"]) if d.get("model_output_message") and isinstance(d["model_output_message"], dict) else d.get("model_output_message")
        token_usage = token_usage_from_dict(d["token_usage"]) if d.get("token_usage") else d.get("token_usage")
        # observations_images and action_output are left as-is
        return cls(
            step_number=d["step_number"],
            timing=timing,
            model_input_messages=model_input_messages if model_input_messages else None,
            tool_calls=tool_calls if tool_calls else None,
            error=error,
            model_output_message=model_output_message,
            model_output=d.get("model_output"),
            code_action=d.get("code_action"),
            observations=d.get("observations"),
            observations_images=d.get("observations_images"),
            action_output=d.get("action_output"),
            token_usage=token_usage,
            is_final_answer=d.get("is_final_answer", False),
        )


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]

    @classmethod
    def from_dict(cls, d):
        def timing_from_dict(timing_dict):
            if isinstance(timing_dict, dict):
                allowed = {"start_time", "end_time"}
                filtered = {k: v for k, v in timing_dict.items() if k in allowed}
                return Timing(**filtered)
            return timing_dict

        def token_usage_from_dict(token_usage_dict):
            if isinstance(token_usage_dict, dict):
                allowed = {"input_tokens", "output_tokens"}
                filtered = {k: v for k, v in token_usage_dict.items() if k in allowed}
                return TokenUsage(**filtered)
            return token_usage_dict

        model_input_messages = [ChatMessage.from_dict(msg) if isinstance(msg, dict) else msg for msg in d["model_input_messages"]]
        model_output_message = ChatMessage.from_dict(d["model_output_message"]) if isinstance(d["model_output_message"], dict) else d["model_output_message"]
        timing = timing_from_dict(d["timing"])
        token_usage = token_usage_from_dict(d["token_usage"]) if d.get("token_usage") else d.get("token_usage")
        return cls(
            model_input_messages=model_input_messages,
            model_output_message=model_output_message,
            plan=d["plan"],
            timing=timing,
            token_usage=token_usage,
        )


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]

    @classmethod
    def from_dict(cls, d):
        return cls(
            task=d["task"],
            task_images=d.get("task_images"),
        )


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]

    @classmethod
    def from_dict(cls, d):
        return cls(system_prompt=d["system_prompt"])


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any


@dataclass
class UserResponseStep(MemoryStep):
    user_message: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": self.user_message}])]

    @classmethod
    def from_dict(cls, d):
        return cls(user_message=d["user_message"])


class AgentMemory:
    """Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tasks, actions, and planning steps.
    It allows for resetting the memory, retrieving succinct or full step information, and replaying the agent's steps.

    Args:
        system_prompt (`str`): System prompt for the agent, which sets the context and instructions for the agent's behavior.

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- System prompt step for the agent.
        - **steps** (`list[TaskStep | ActionStep | PlanningStep]`) -- List of steps taken by the agent, which can include tasks, actions, and planning steps.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """Return a succinct representation of the agent's steps, excluding model input messages."""
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """Return a full representation of the agent's steps, including model input messages."""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (`AgentLogger`): The logger to print replay logs to.
            detailed (`bool`, default `False`): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)

    def return_full_code(self) -> str:
        """Returns all code actions from the agent's steps, concatenated as a single script."""
        return "\n\n".join(
            [step.code_action for step in self.steps if isinstance(step, ActionStep) and step.code_action is not None]
        )
    
    def dump_to_dict(self) -> dict:
        """Returns a dictionary representation of the agent's memory, including system prompt and steps."""
        return {
            "system_prompt": self.system_prompt.dict(),
            "steps": [{"step_type": step.__class__.__name__, "step_data": step.dict()} for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls(system_prompt=data["system_prompt"]["system_prompt"])
        obj.system_prompt = SystemPromptStep.from_dict(data["system_prompt"])
        step_classes = {
            "ActionStep": ActionStep,
            "PlanningStep": PlanningStep,
            "TaskStep": TaskStep,
            "UserResponseStep": UserResponseStep,
        }
        obj.steps = [
            step_classes[step["step_type"]].from_dict(step["step_data"])
            for step in data["steps"]
        ]
        return obj


class CallbackRegistry:
    """Registry for callbacks that are called at each step of the agent's execution.

    Callbacks are registered by passing a step class and a callback function.
    """

    def __init__(self):
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}

    def register(self, step_cls: Type[MemoryStep], callback: Callable):
        """Register a callback for a step class.

        Args:
            step_cls (Type[MemoryStep]): Step class to register the callback for.
            callback (Callable): Callback function to register.
        """
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step, **kwargs):
        """Call callbacks registered for a step type.

        Args:
            memory_step (MemoryStep): Step to call the callbacks for.
            **kwargs: Additional arguments to pass to callbacks that accept them.
                Typically, includes the agent instance.

        Notes:
            For backwards compatibility, callbacks with a single parameter signature
            receive only the memory_step, while callbacks with multiple parameters
            receive both the memory_step and any additional kwargs.
        """
        # For compatibility with old callbacks that only take the step as an argument
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(memory_step) if len(inspect.signature(cb).parameters) == 1 else cb(memory_step, **kwargs)
