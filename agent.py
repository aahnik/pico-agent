# pyright: reportExplicitAny=false
# pyright: reportAny=false
#
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import os

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Standard return type for tool execution.
    Must contain at least a 'result' key with textual output for the LLM.
    Can contain arbitrary additional keys for debugging/tracing.
    """

    result: str = Field(description="Textual result suitable for language model")

    class Config:
        extra: str = "allow"  # Allow arbitrary additional fields


type CacheDict = dict[str, ToolResult]


# the tool class defines the prompt format, before and after
# before means, when tool execution engine feeds tool context to the model, for deciding tool
# or for generating args for calling the tool. it needs to give details about the tool to the model.
# the tool class must define those details
#
# after means, once the tool has given its result, the result needs to be feed back for the model for the next step
# the tool function might produce intermediate details/some metadata, for logging to the trace
# but it might expect to feed the model with only relevant details, in a specific format
# will the tool be allowed to edit the history of the context ? for now NO
#
#
#
# a tool call can be made in 2 hops OR 1 hop
# model first selects the tool, and then outputs the args for tool call
# the model can also select tool AND output the args for tool call, in a single call
class Tool(ABC):
    # a way to define a tool
    # defines the actual function, which will be executed when the tool is called
    # defines the signature of the tool
    # supports tool caching: uselful for experiments,
    # the results for tool call with certain set of inputs can be cached to disk
    # pre-warming: the cache can be loaded to memory, during tool initialization
    #
    #

    def __init__(self):
        self.name: str = self.__class__.__name__
        self.cache: CacheDict = {}

    @property
    def description(self) -> str:
        """Override this to provide tool description for the model"""
        return ""

    @property
    def parameters_schema(self) -> type[BaseModel]:
        """Override this to define tool parameters schema using Pydantic BaseModel"""
        return BaseModel

    def call(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments and return validated ToolResult"""
        # Validate input using Pydantic schema
        validated_params = self.parameters_schema(**kwargs)

        # Check cache first
        cache_key = json.dumps(validated_params.model_dump(), sort_keys=True)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Execute and validate result
        result = self.execute(**validated_params.model_dump())

        self.cache[cache_key] = result
        return result

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Override this with actual tool implementation.
        Must return a dict (wrap with ToolResult) with at least a 'result' key containing textual output for LLM.
        Can include arbitrary additional keys for debugging/tracing.
        """
        pass

    def format_for_model(self) -> str:
        """Format tool info for model context"""
        schema = self.parameters_schema.model_json_schema()
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters: {json.dumps(schema)}"

    # def load_cache(self, cache_dict: dict[]):
    #     """Pre-warm cache with saved results"""
    #     self.cache.update(cache_dict)
    #
    # TODO: implement load_cache/prewarm and dump_cache (syncing cache obj to disk for persistent storage)


@dataclass
class ToolCallingModelResponse:
    """Represents a typical model's response, which is capable of both tool calling, and generating final answer"""

    # see the model might be capable of calling multiple tools at once,
    # in that case, a different structure might be used to represent it
    # may be call it MultiToolCallingResponse

    # NOTE: for now we are assuming the modality of the model's response includes only text
    # content is what the model actually produces
    # but then we parse it, and determine, if any tool call was intended
    # NOTE: for Model class inheritors, you

    # actual response
    content: str

    # inferred
    is_final: bool = False
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


class ToolEngine(ABC):
    def __init__(self, tools: list[Tool]):
        self.tools: list[Tool] = tools

    def get_tools_context(self, tools: list[Tool]) -> str:
        return "\n\n".join([tool.format_for_model() for tool in tools])

    @abstractmethod
    def loop(
        self, question: str, image: str | None = None, max_calls: int = 5
    ) -> Trace:
        pass


class GreedyToolEngine(ToolEngine):
    def __init__(self, tools: list[Tool], model: "ToolCallingModel"):
        super().__init__(tools)
        self.model = model

    def loop(
        self, question: str, image: str | None = None, max_calls: int = 5
    ) -> Trace:
        trace = Trace()

        tools_context = self.get_tools_context(self.tools)
        context = f"{tools_context}\n\nQuestion: {question}"

        if image:
            context = f"Image: {image}\n\n{context}"

        calls = 0

        response = self.model.generate(context)
        trace.push(
            [
                ModelCall(
                    model=self.model.__class__.__name__,
                    input=context,
                    images=[image] if image else [],
                    output=response.content,
                )
            ]
        )

        while not response.is_final and calls < max_calls:
            if response.tool_name and response.tool_args:
                tool = next(
                    (t for t in self.tools if t.name == response.tool_name), None
                )

                if tool is None:
                    context += f"\n\nError: Tool '{response.tool_name}' not found. Please provide final answer."
                    response = self.model.generate(context)
                    trace.push(
                        [
                            ModelCall(
                                model=self.model.__class__.__name__,
                                input=context,
                                images=[],
                                output=response.content,
                            )
                        ]
                    )
                    break

                tool_output = tool.call(**response.tool_args)

                trace.push(
                    [
                        ToolCall(
                            tool=response.tool_name,
                            input=response.tool_args,
                            output=tool_output.model_dump(),
                        )
                    ]
                )

                context += f"\n\nTool: {response.tool_name}\nInput: {json.dumps(response.tool_args)}\nOutput: {json.dumps(tool_output.model_dump())}"

                response = self.model.generate(context)
                trace.push(
                    [
                        ModelCall(
                            model=self.model.__class__.__name__,
                            input=context,
                            images=[],
                            output=response.content,
                        )
                    ]
                )

                calls += 1
            else:
                break

        if calls >= max_calls and not response.is_final:
            context += (
                "\n\nMax tool calls reached. Please provide your final answer now."
            )
            response = self.model.generate(context)
            trace.push(
                [
                    ModelCall(
                        model=self.model.__class__.__name__,
                        input=context,
                        images=[],
                        output=response.content,
                    )
                ]
            )

        return trace


class Model(ABC):
    """Abstract base class for language models"""

    def __init__(self, **kwargs: Any):
        # here you define the configs required to initialize the model
        # any settings or config required
        pass

    @abstractmethod
    def load(self) -> None:
        # optional: for local inferencing, this is an extra step
        # at this step, the model is loaded into the gpu, not before it
        pass

    @abstractmethod
    def unload(self) -> None:
        # optional: incase of local inferencing, the model class should provide a way to unload/free memory
        pass

    @abstractmethod
    def generate(self, context: str, images: list[Path] | None) -> Any:
        # decode the model's output, and just return the new string, post the input
        # we assume the model can me text -> text, or text + image -> text
        # although the language model will output a str, we are not forcing you to return str, from this func
        # you can parse it, and return custom objects suitable for your use case's implementation
        # example using outlines.txt for structured generation
        # the response type can be any application specific class
        pass


class ToolCallingModel(ABC):
    @abstractmethod
    def generate(
        self, context: str, images: list[Path] | None = None
    ) -> ToolCallingModelResponse:
        pass


# see Model is just about calling the model
# the things like force_final etc, does not make sense
#


@dataclass
class ModelCall:
    model: str
    input: str  # the input context to Model.generate()
    images: list[str]  # list of file paths used as images for multimodal input
    output: str  # the output serialized as string
    # (although Model.generate can return complex objects as output, for tracing purposes, we keep record of a str)

    def json(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)


# TODO: add more metadata like time of execution, input and output tokens


@dataclass
class ToolCall:
    tool: str
    # the name of the tool, unique identifier scoped to applications's context

    input: dict[str, Any]  # the input args to the tool call
    output: dict[str, Any]  # the output of the tool

    def json(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)


type TraceLayer = list[ModelCall | ToolCall]


class Trace:
    # container for traces of interaction with tools and raw model
    def __init__(self):
        self.__stack: list[TraceLayer] = []

    def push(self, layer: TraceLayer):
        self.__stack.append(layer)

    def json(self):
        # Convert each item in the stack to a dict
        return json.dumps(self.obj())

    def obj(self) -> list[list[dict[str, Any]]]:
        return [[item.to_dict() for item in layer] for layer in self.__stack]


@dataclass
class AgentResponse:
    confidence: float
    trace: Trace
    answer: str


# how does trace.json() might look like
#
# what the model is called with + what the model replied
# what the tool is called with + what the tool replied
#
# the trace is fundamentally redundant, because,
# the models output + tool call response is again fed back to the model
# traces exist for the three-fold reason of reproducibility, resumability, and observability
# the trace must also capture the inherent flow of the model/tool calls, in case things are made parallely
# we can make our agent flexible as to do multiple pll tool and as well as model calls
#
# the trace can be visualized as an acyclic directed graph
#
# the trace is represented as an stack of array of calls
# in each stack layer, all the pll calls are represented. all items in a stack layer are independent of each other
# the next layer depends on the outputs of the previous
#
#
#
# [  [layer 1] [layer 2] [layer 3]  ]
# sequenciality: layer 1 --> layer 2 --> layer 3
# [  [ModelCall(in, out)]  [ToolCall(in, out) ] [ModelCall(in, out) ]                               ]
# [  [ModelCall(in, out)]  [Tool@A(in, out)  Tool@B(in, out)]   [Model@A(in, out) Model@B(in, out)] ]                                     ]
# json representation, is just json.dumps(python object)
# Trace object contains a single execution trace of the agent invocation, from beginning to end
# its recommended for a single execution to be written to a seperate json file, while checkpointing
# mutliple calls under a similar venture can be stored under a directory
#
#


class Checkpointer:
    dir: Path

    def __init__(self, dir: str):
        self.dir = Path(dir)
        os.makedirs(self.dir, exist_ok=True)

    def write(self, file_path: str, trace: Trace):
        with open(file=self.dir / file_path, mode="w", encoding="utf-8") as file:
            file.write(trace.json())


# what is an agent ? its the flow or graph of model invocations
# now, models cant be invoked from air, model invocations happen tightly coupled with ToolEngine
# if the model outputs final response its terminated, but if it outputs a tool call, the tool engine is responsible for executing it
# the tool engine is where we define the model -- tool interaction loop
#
#
#
@dataclass
class Agent:
    # customizable, parallizable, interruptible agents
    # an agent is a graph of object functions
    # for now, we are implementing a mono-objective function agent
    # later we will expand to supports graphs (which by default includes linear chains)
    #
    model: Model
    tools: list[Tool]
    router: ToolEngine
    checkpointer: Checkpointer | None = None
    max_calls: int = 5

    def __post_init__(self):
        """Initialize agent after dataclass creation"""
        self.__state = {}
        self.trace = Trace()

        # Bind model and router
        self.model.bind(self.router)

    @property
    def state(self):
        return self.__state

    def invoke(self, question: str, image: str = None) -> AgentResponse:
        """
        Main execution loop for the agent
        Iteratively calls model and tools until final answer is reached
        """
        # Initialize context with tools and question
        tools_context = self.router.get_tools_context(self.tools)
        context = f"{tools_context}\n\nQuestion: {question}"

        if image:
            context = f"Image: {image}\n\n{context}"

        calls = 0

        # Initial model call
        response = self.model.generate(context, force_final=False)
        self.trace.push(
            [
                ModelCall(
                    model=self.model.__class__.__name__,
                    input=context,
                    output=response.content,
                )
            ]
        )

        # Tool execution loop
        while not response.is_final and calls < self.max_calls:
            if response.tool_name and response.tool_args:
                # Find the tool
                tool = next(
                    (t for t in self.tools if t.name == response.tool_name), None
                )

                if tool is None:
                    # Tool not found, force final answer
                    context += f"\n\nError: Tool '{response.tool_name}' not found. Please provide final answer."
                    response = self.model.generate(context, force_final=True)
                    self.trace.push(
                        [
                            ModelCall(
                                model=self.model.__class__.__name__,
                                input=context,
                                output=response.content,
                            )
                        ]
                    )
                    break

                # Execute tool
                tool_output = tool.call(**response.tool_args)

                # Record tool call
                self.trace.push(
                    [
                        ToolCall(
                            tool=response.tool_name,
                            input=response.tool_args,
                            output=tool_output,
                        )
                    ]
                )

                # Update context with tool result
                context += f"\n\nTool: {response.tool_name}\nInput: {json.dumps(response.tool_args)}\nOutput: {json.dumps(tool_output)}"

                # Call model again with updated context
                response = self.model.generate(context, force_final=False)
                self.trace.push(
                    [
                        ModelCall(
                            model=self.model.__class__.__name__,
                            input=context,
                            output=response.content,
                        )
                    ]
                )

                calls += 1
            else:
                # No tool call in response, must be final
                break

        # When max calls have expired, force the model to generate final answer
        if calls >= self.max_calls and not response.is_final:
            context += (
                "\n\nMax tool calls reached. Please provide your final answer now."
            )
            response = self.model.generate(context, force_final=True)
            self.trace.push(
                [
                    ModelCall(
                        model=self.model.__class__.__name__,
                        input=context,
                        output=response.content,
                    )
                ]
            )

        # Checkpoint if available
        if self.checkpointer:
            self.checkpoint()

        return AgentResponse(
            confidence=1.0,  # TODO: implement confidence scoring
            trace=self.trace,
            answer=response.content,
        )

    def checkpoint(self, filename: str = None):
        """Write the current trace to disk"""
        if self.checkpointer is None:
            raise Exception(
                "Checkpointer not found! Failed to write current trace to disk"
            )

        if filename is None:
            # Generate filename from timestamp
            from datetime import datetime

            filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        self.checkpointer.write(filename, self.trace)

    def update_state(self, key: str, value: any):
        """Update agent state"""
        self.__state[key] = value

    def get_state(self, key: str, default=None):
        """Get value from agent state"""
        return self.__state.get(key, default)


# usage example
#
#

# model = SomeModel("model_path", other_args)

# tool_a = ToolA()
# tool_b = ToolB()

# std_router = ToolEngine(tool_a, tool_b)

# agent = Agent(
#     model=model, router=std_router, tools=[tool_a, tool_b], checkpointer=checkpointer
# )


# answer = agent.invoke(question="What is 2+2?")


# print(answer.answer)  # prints the final answer

# print(answer.trace.json())  # prints the full trace that lead to this answer


# TODO: structured output validation
#
#
#
#
