from abc import ABC
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os


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

    def __init__():
        pass

    def call(*args, **kwargs):
        pass


class ToolEngine(ABC):
    # provides mechanism to decide which tool to call
    #
    __allow_pll_calls: bool = False
    # whether to allow multiple parallel tool calls, before the model is invoked again

    def __init__():
        pass

    def decide_tools(self, tools: list[Tool], call_model) -> Tool:
        # it can choose to call multiple tools at once
        return self.call_model(
            "Given context: {context} which tool should you call or generate final answer "
        )


class Model(ABC):
    __router: ToolEngine

    def __init__():
        pass

    def bind(self, router: ToolEngine):
        # binds the model with a tool router
        # thus allowing us to generate a final answer, or choose a tool
        # the generate method will return the model's output
        # the execution environment is concerned with the implementation of the tool call
        # the model needs to be called again with updated context
        router.call_model = self.generate

        pass

    def generate():
        pass


@dataclass
class ModelCall:
    model: str
    input: str
    # todo handle multimodal input where image is present
    output: str

    def json(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)


# TODO: add more metadata like time of execution, input and output tokens


@dataclass
class ToolCall:
    tool: str
    input: dict[str, any]
    output: dict[str, any]

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

    def obj(self) -> list[TraceLayer]:
        return [[asdict(item) for item in layer] for layer in self.__stack]


@dataclass
class Response:
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


@dataclass
class Agent:
    # customizable, parallizable, interruptible agents
    model: Model
    tools: list[Tool]
    checkpointer: Checkpointer

    __state: dict[str, any]
    __max_calls: int = 5

    def bind(model, tool_engine):
        # the binding of model and tool engine happens at the agent level
        # one model can be used with multiple different tool engines, and similarly
        # a tool engine can be used with multiple different models
        # the instantiation of joint(model, tool_engine) is the agent
        #
        pass

    def invoke(self, image: str, question: str) -> Response:
        context = self.router.get_context(self.tools)
        answer = self.model.generate(context)

        calls = 0

        while not answer.is_final and calls < self.max_calls:
            tool, args = answer.tool_call
            context.update(tool.call(args))
            answer = self.model.generate(context)
            calls += 1

        # when max calls have expired, force the model to generate final answer,
        # instead of giving it option to call tool
        return answer

    def checkpoint(self):
        # write the current state to disk
        self.checkpointer.write(self.state)


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


# answer = agent.invoke(image, question)


# print(answer)  # prints the final answer

# print(answer.trace)  # prints the full trace that lead to this answer


# TODO: structured output validation
#
#
