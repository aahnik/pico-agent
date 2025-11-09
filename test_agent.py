"""
Comprehensive test suite for agent.py
Tests every class, method, execution path, and edge case
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from typing import Type
from pydantic import BaseModel, Field
from agent import (
    Tool,
    ToolResult,
    Model,
    ToolEngine,
    Agent,
    Checkpointer,
    ModelCall,
    ToolCall,
    Trace,
    AgentResponse,
    ToolCallingModelResponse,
)

# test change

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpointer(temp_dir):
    """Create a checkpointer with temp directory"""
    return Checkpointer(temp_dir)


@pytest.fixture
def mock_trace():
    """Create a mock trace with sample data"""
    trace = Trace()
    trace.push([ModelCall(model="test-model", input="hello", output="hi")])
    trace.push([ToolCall(tool="test-tool", input={"x": 1}, output={"y": 2})])
    return trace


# ============================================================================
# CONCRETE IMPLEMENTATIONS FOR TESTING
# ============================================================================


class MockTool(Tool):
    """Mock tool for testing"""

    class Parameters(BaseModel):
        x: int = Field(default=0, description="An integer parameter")
        y: int = Field(default=0, description="Another parameter")

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters_schema(self) -> Type[BaseModel]:
        return self.Parameters

    def execute(self, x: int = 0, y: int = 0) -> dict:
        return {
            "result": f"Sum: {x + y}, Product: {x * y}",
            "sum": x + y,
            "product": x * y,
        }


class FailingTool(Tool):
    """Tool that always raises an exception"""

    class Parameters(BaseModel):
        pass

    @property
    def description(self) -> str:
        return "A tool that fails"

    @property
    def parameters_schema(self) -> Type[BaseModel]:
        return self.Parameters

    def execute(self, **kwargs) -> dict:
        raise ValueError("Tool execution failed")
        # Note: This will never return, but if it did, it would need:
        # return {"result": "error"}


class CacheableTool(Tool):
    """Tool that tracks execution count"""

    class Parameters(BaseModel):
        value: int = Field(description="input value")

    def __init__(self):
        super().__init__()
        self.execution_count = 0

    @property
    def description(self) -> str:
        return "Cacheable tool"

    @property
    def parameters_schema(self) -> Type[BaseModel]:
        return self.Parameters

    def execute(self, value: int) -> dict:
        self.execution_count += 1
        return {
            "result": f"Doubled value: {value * 2}",
            "computed_result": value * 2,
            "count": self.execution_count,
        }


class MockModel(Model):
    """Mock model that follows a script"""

    def __init__(self, script: list[ToolCallingModelResponse] = None):
        super().__init__()
        self.script = script or []
        self.call_count = 0
        self.contexts_received = []

    def generate(
        self, context: str, force_final: bool = False
    ) -> ToolCallingModelResponse:
        self.contexts_received.append(context)
        self.call_count += 1

        if force_final:
            return ToolCallingModelResponse(
                content="Forced final answer", is_final=True
            )

        if self.script and self.call_count <= len(self.script):
            return self.script[self.call_count - 1]

        return ToolCallingModelResponse(content="Default response", is_final=True)


class SimplestModel(Model):
    """Simplest possible model - always returns final answer"""

    def generate(
        self, context: str, force_final: bool = False
    ) -> ToolCallingModelResponse:
        return ToolCallingModelResponse(content="Simple answer", is_final=True)


class ToolCallingModel(Model):
    """Model that calls a specific tool once then finishes"""

    def __init__(self, tool_name: str, tool_args: dict):
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.called = False
        self.contexts_received = []

    def generate(
        self, context: str, force_final: bool = False
    ) -> ToolCallingModelResponse:
        self.contexts_received.append(context)

        if force_final or self.called:
            return ToolCallingModelResponse(
                content="Final answer after tool", is_final=True
            )

        self.called = True
        return ToolCallingModelResponse(
            content="Calling tool",
            is_final=False,
            tool_name=self.tool_name,
            tool_args=self.tool_args,
        )


class SimpleRouter(ToolEngine):
    """Simple router for testing"""

    def decide_tools(self, tools, context):
        # Not used in basic agent implementation
        return []


# ============================================================================
# TOOL CLASS TESTS
# ============================================================================


class TestTool:
    """Test the Tool abstract class and its implementations"""

    def test_tool_initialization(self):
        """Test tool initializes with correct name and empty cache"""
        tool = MockTool()
        assert tool.name == "MockTool"
        assert tool.cache == {}

    def test_tool_description_property(self):
        """Test tool description property"""
        tool = MockTool()
        assert tool.description == "A mock tool for testing"
        assert isinstance(tool.description, str)

    def test_tool_parameters_schema_property(self):
        """Test tool parameters schema property"""
        tool = MockTool()
        schema = tool.parameters_schema
        # Should return a Pydantic BaseModel class
        assert issubclass(schema, BaseModel)
        # Check that the schema has the expected fields
        json_schema = schema.model_json_schema()
        assert "x" in json_schema["properties"]
        assert "y" in json_schema["properties"]

    def test_tool_execute_method(self):
        """Test tool execute method"""
        tool = MockTool()
        result = tool.execute(x=5, y=3)
        assert "result" in result
        assert result["sum"] == 8
        assert result["product"] == 15

    def test_tool_call_without_cache(self):
        """Test tool call executes and returns result"""
        tool = MockTool()
        result = tool.call(x=10, y=5)
        assert "result" in result
        assert result["sum"] == 15
        assert result["product"] == 50

    def test_tool_call_with_caching(self):
        """Test tool call uses cache on repeated calls"""
        tool = CacheableTool()

        # First call - should execute
        result1 = tool.call(value=5)
        assert "result" in result1
        assert result1["computed_result"] == 10
        assert result1["count"] == 1
        assert tool.execution_count == 1

        # Second call with same args - should use cache
        result2 = tool.call(value=5)
        assert result2["computed_result"] == 10
        assert result2["count"] == 1  # Same as first call
        assert tool.execution_count == 1  # No new execution

        # Call with different args - should execute
        result3 = tool.call(value=7)
        assert result3["computed_result"] == 14
        assert result3["count"] == 2
        assert tool.execution_count == 2

    def test_tool_cache_key_generation(self):
        """Test cache keys are consistent for same arguments"""
        tool = MockTool()

        tool.call(x=1, y=2)
        tool.call(y=2, x=1)  # Different order, same args

        # Should only have one cache entry
        assert len(tool.cache) == 1

    def test_tool_load_cache(self):
        """Test loading cache from dictionary"""
        tool = MockTool()

        # Pre-warm cache
        cache_data = {
            '{"x": 5, "y": 3}': {
                "result": "Sum: 8, Product: 15",
                "sum": 8,
                "product": 15,
            }
        }
        tool.load_cache(cache_data)

        # Call should use pre-warmed cache
        result = tool.call(x=5, y=3)
        assert "result" in result
        assert result["sum"] == 8
        assert result["product"] == 15

    def test_tool_format_for_model(self):
        """Test tool formatting for model prompt"""
        tool = MockTool()
        formatted = tool.format_for_model()

        assert "MockTool" in formatted
        assert "A mock tool for testing" in formatted
        # Check that the formatted string contains schema information
        assert "x" in formatted or "integer" in formatted.lower()
        assert "y" in formatted or "parameter" in formatted.lower()
        assert "Tool:" in formatted
        assert "Description:" in formatted
        assert "Parameters:" in formatted

    def test_tool_with_no_parameters(self):
        """Test tool with empty parameters schema"""

        class NoParamTool(Tool):
            class Parameters(BaseModel):
                pass

            @property
            def description(self):
                return "No params"

            @property
            def parameters_schema(self):
                return self.Parameters

            def execute(self, **kwargs):
                return {"result": "ok", "status": "ok"}

        tool = NoParamTool()
        result = tool.call()
        assert result["status"] == "ok"

    def test_tool_execute_not_implemented(self):
        """Test that Tool is abstract and execute must be implemented"""

        class IncompleteToolClass(Tool):
            class Parameters(BaseModel):
                pass

            @property
            def description(self):
                return "test"

            @property
            def parameters_schema(self):
                return self.Parameters

        # Should not be able to instantiate without implementing execute
        with pytest.raises(TypeError):
            tool = IncompleteToolClass()


# ============================================================================
# MODEL CLASS TESTS
# ============================================================================


class TestModel:
    """Test the Model abstract class and implementations"""

    def test_model_initialization(self):
        """Test model initializes correctly"""
        model = SimplestModel()
        assert model is not None

    def test_model_generate_method(self):
        """Test model generate method"""
        model = SimplestModel()
        response = model.generate("test context")
        assert isinstance(response, ToolCallingModelResponse)
        assert response.is_final is True

    def test_model_generate_with_force_final(self):
        """Test model respects force_final parameter"""
        model = MockModel(
            script=[
                ToolCallingModelResponse(
                    content="tool call",
                    is_final=False,
                    tool_name="TestTool",
                    tool_args={},
                )
            ]
        )

        # Without force_final
        response1 = model.generate("context", force_final=False)
        assert response1.is_final is False

        # With force_final
        response2 = model.generate("context", force_final=True)
        assert response2.is_final is True

    def test_model_bind_to_router(self):
        """Test model binding to router"""
        model = SimplestModel()
        router = SimpleRouter()

        model.bind(router)

        # Router should have reference to model's generate method
        assert router.call_model is not None
        assert callable(router.call_model)

    def test_model_tracks_calls(self):
        """Test mock model tracks its calls"""
        model = MockModel()

        model.generate("first context")
        model.generate("second context")

        assert model.call_count == 2
        assert len(model.contexts_received) == 2
        assert "first context" in model.contexts_received[0]

    def test_model_with_script(self):
        """Test mock model follows a script"""
        script = [
            ToolCallingModelResponse(
                content="step 1", is_final=False, tool_name="Tool1", tool_args={"x": 1}
            ),
            ToolCallingModelResponse(
                content="step 2", is_final=False, tool_name="Tool2", tool_args={"y": 2}
            ),
            ToolCallingModelResponse(content="final", is_final=True),
        ]
        model = MockModel(script=script)

        r1 = model.generate("context")
        assert r1.content == "step 1"
        assert r1.tool_name == "Tool1"

        r2 = model.generate("context")
        assert r2.content == "step 2"

        r3 = model.generate("context")
        assert r3.content == "final"
        assert r3.is_final is True


# ============================================================================
# MODEL RESPONSE TESTS
# ============================================================================


class TestModelResponse:
    """Test ModelResponse dataclass"""

    def test_model_response_final_answer(self):
        """Test creating a final answer response"""
        response = ToolCallingModelResponse(content="final answer", is_final=True)
        assert response.content == "final answer"
        assert response.is_final is True
        assert response.tool_name is None
        assert response.tool_args is None

    def test_model_response_tool_call(self):
        """Test creating a tool call response"""
        response = ToolCallingModelResponse(
            content="calling tool",
            is_final=False,
            tool_name="TestTool",
            tool_args={"x": 1, "y": 2},
        )
        assert response.is_final is False
        assert response.tool_name == "TestTool"
        assert response.tool_args == {"x": 1, "y": 2}

    def test_model_response_defaults(self):
        """Test ModelResponse default values"""
        response = ToolCallingModelResponse(content="test")
        assert response.is_final is False
        assert response.tool_name is None
        assert response.tool_args is None


# ============================================================================
# TOOL ENGINE TESTS
# ============================================================================


class TestToolEngine:
    """Test ToolEngine abstract class"""

    def test_tool_engine_initialization(self):
        """Test tool engine initializes correctly"""
        engine = SimpleRouter()
        assert engine is not None

    def test_tool_engine_initialization_with_parallel_calls(self):
        """Test tool engine with parallel calls flag"""
        engine = SimpleRouter(allow_parallel_calls=True)
        assert engine is not None

    def test_tool_engine_get_tools_context(self):
        """Test formatting tools context for model"""
        engine = SimpleRouter()
        tools = [MockTool(), CacheableTool()]

        context = engine.get_tools_context(tools)

        assert "MockTool" in context
        assert "CacheableTool" in context
        assert "mock tool" in context.lower()
        assert isinstance(context, str)

    def test_tool_engine_get_tools_context_empty_list(self):
        """Test get_tools_context with empty tools list"""
        engine = SimpleRouter()
        context = engine.get_tools_context([])
        assert context == ""

    def test_tool_engine_call_model_binding(self):
        """Test that call_model is set to None initially"""
        engine = SimpleRouter()
        assert engine.call_model is None


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================


class TestModelCall:
    """Test ModelCall dataclass"""

    def test_model_call_creation(self):
        """Test creating a ModelCall"""
        mc = ModelCall(model="test-model", input="hello", output="hi")
        assert mc.model == "test-model"
        assert mc.input == "hello"
        assert mc.output == "hi"

    def test_model_call_json(self):
        """Test ModelCall JSON serialization"""
        mc = ModelCall(model="test-model", input="hello", output="hi")
        json_str = mc.json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["model"] == "test-model"
        assert data["input"] == "hello"
        assert data["output"] == "hi"

    def test_model_call_to_dict(self):
        """Test ModelCall to_dict method"""
        mc = ModelCall(model="test-model", input="hello", output="hi")
        data = mc.to_dict()

        assert isinstance(data, dict)
        assert data["model"] == "test-model"


class TestToolCall:
    """Test ToolCall dataclass"""

    def test_tool_call_creation(self):
        """Test creating a ToolCall"""
        tc = ToolCall(tool="calc", input={"x": 5}, output={"y": 10})
        assert tc.tool == "calc"
        assert tc.input == {"x": 5}
        assert tc.output == {"y": 10}

    def test_tool_call_json(self):
        """Test ToolCall JSON serialization"""
        tc = ToolCall(tool="calc", input={"x": 5}, output={"y": 10})
        json_str = tc.json()

        data = json.loads(json_str)
        assert data["tool"] == "calc"
        assert data["input"]["x"] == 5
        assert data["output"]["y"] == 10

    def test_tool_call_to_dict(self):
        """Test ToolCall to_dict method"""
        tc = ToolCall(tool="calc", input={}, output={})
        data = tc.to_dict()
        assert isinstance(data, dict)


class TestTrace:
    """Test Trace class"""

    def test_trace_initialization(self):
        """Test trace initializes with empty stack"""
        trace = Trace()
        assert trace.obj() == []

    def test_trace_push_single_layer(self):
        """Test pushing a single layer to trace"""
        trace = Trace()
        layer = [ModelCall(model="m", input="i", output="o")]
        trace.push(layer)

        assert len(trace.obj()) == 1
        assert len(trace.obj()[0]) == 1

    def test_trace_push_multiple_layers(self):
        """Test pushing multiple layers"""
        trace = Trace()
        trace.push([ModelCall(model="m1", input="i1", output="o1")])
        trace.push([ToolCall(tool="t1", input={}, output={})])
        trace.push([ModelCall(model="m2", input="i2", output="o2")])

        assert len(trace.obj()) == 3

    def test_trace_push_parallel_calls(self):
        """Test pushing layer with multiple parallel calls"""
        trace = Trace()
        layer = [
            ToolCall(tool="t1", input={"x": 1}, output={"y": 1}),
            ToolCall(tool="t2", input={"x": 2}, output={"y": 2}),
        ]
        trace.push(layer)

        assert len(trace.obj()) == 1
        assert len(trace.obj()[0]) == 2

    def test_trace_json_serialization(self):
        """Test trace JSON serialization"""
        trace = Trace()
        trace.push([ModelCall(model="m", input="i", output="o")])
        trace.push([ToolCall(tool="t", input={"x": 1}, output={"y": 2})])

        json_str = trace.json()
        assert isinstance(json_str, str)

        data = json.loads(json_str)
        assert len(data) == 2
        assert data[0][0]["model"] == "m"
        assert data[1][0]["tool"] == "t"

    def test_trace_obj_returns_list(self):
        """Test trace obj returns proper list structure"""
        trace = Trace()
        trace.push([ModelCall(model="m", input="i", output="o")])

        obj = trace.obj()
        assert isinstance(obj, list)
        assert isinstance(obj[0], list)
        assert isinstance(obj[0][0], dict)


class TestResponse:
    """Test Response dataclass"""

    def test_response_creation(self):
        """Test creating a Response"""
        trace = Trace()
        response = AgentResponse(confidence=0.95, trace=trace, answer="test answer")

        assert response.confidence == 0.95
        assert response.trace is trace
        assert response.answer == "test answer"

    def test_response_with_full_trace(self, mock_trace):
        """Test response with complete trace"""
        response = AgentResponse(confidence=1.0, trace=mock_trace, answer="final")

        assert len(response.trace.obj()) == 2


# ============================================================================
# CHECKPOINTER TESTS
# ============================================================================


class TestCheckpointer:
    """Test Checkpointer class"""

    def test_checkpointer_initialization(self, temp_dir):
        """Test checkpointer creates directory"""
        cp = Checkpointer(temp_dir)
        assert cp.dir == Path(temp_dir)
        assert os.path.exists(temp_dir)

    def test_checkpointer_creates_nonexistent_directory(self, temp_dir):
        """Test checkpointer creates directory if it doesn't exist"""
        new_dir = os.path.join(temp_dir, "subdir", "nested")
        cp = Checkpointer(new_dir)
        assert os.path.exists(new_dir)

    def test_checkpointer_write_trace(self, temp_dir, mock_trace):
        """Test writing trace to file"""
        cp = Checkpointer(temp_dir)
        cp.write("test_trace.json", mock_trace)

        filepath = Path(temp_dir) / "test_trace.json"
        assert os.path.exists(filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0][0]["model"] == "test-model"

    def test_checkpointer_write_multiple_traces(self, temp_dir):
        """Test writing multiple traces"""
        cp = Checkpointer(temp_dir)

        trace1 = Trace()
        trace1.push([ModelCall(model="m1", input="i1", output="o1")])

        trace2 = Trace()
        trace2.push([ModelCall(model="m2", input="i2", output="o2")])

        cp.write("trace1.json", trace1)
        cp.write("trace2.json", trace2)

        assert os.path.exists(Path(temp_dir) / "trace1.json")
        assert os.path.exists(Path(temp_dir) / "trace2.json")

    def test_checkpointer_overwrites_existing_file(self, temp_dir, mock_trace):
        """Test that writing to existing file overwrites it"""
        cp = Checkpointer(temp_dir)

        # Write first trace
        cp.write("test.json", mock_trace)

        # Write different trace to same file
        new_trace = Trace()
        new_trace.push([ModelCall(model="new", input="new", output="new")])
        cp.write("test.json", new_trace)

        # Verify it was overwritten
        with open(Path(temp_dir) / "test.json") as f:
            data = json.load(f)

        assert data[0][0]["model"] == "new"


# ============================================================================
# AGENT CLASS TESTS
# ============================================================================


class TestAgent:
    """Test Agent class - the core orchestrator"""

    def test_agent_initialization(self, checkpointer):
        """Test agent initializes correctly"""
        model = SimplestModel()
        tool = MockTool()
        router = SimpleRouter()

        agent = Agent(
            model=model,
            tools=[tool],
            router=router,
            checkpointer=checkpointer,
            max_calls=5,
        )

        assert agent.model is model
        assert agent.tools == [tool]
        assert agent.router is router
        assert agent.checkpointer is checkpointer
        assert agent.max_calls == 5

    def test_agent_initialization_without_checkpointer(self):
        """Test agent can be created without checkpointer"""
        agent = Agent(
            model=SimplestModel(),
            tools=[MockTool()],
            router=SimpleRouter(),
            checkpointer=None,
            max_calls=3,
        )

        assert agent.checkpointer is None

    def test_agent_post_init_creates_trace(self):
        """Test agent creates trace during initialization"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        assert isinstance(agent.trace, Trace)
        assert len(agent.trace.obj()) == 0

    def test_agent_post_init_binds_model_and_router(self):
        """Test agent binds model and router during init"""
        model = SimplestModel()
        router = SimpleRouter()

        agent = Agent(model=model, tools=[], router=router)

        # Router should have model's generate method
        assert router.call_model is not None

    def test_agent_state_property(self):
        """Test agent state property"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        assert isinstance(agent.state, dict)
        assert len(agent.state) == 0

    def test_agent_update_state(self):
        """Test updating agent state"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        agent.update_state("key1", "value1")
        agent.update_state("key2", 123)

        assert agent.state["key1"] == "value1"
        assert agent.state["key2"] == 123

    def test_agent_get_state(self):
        """Test getting values from agent state"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        agent.update_state("test_key", "test_value")

        value = agent.get_state("test_key")
        assert value == "test_value"

    def test_agent_get_state_with_default(self):
        """Test get_state returns default for missing key"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        value = agent.get_state("nonexistent", default="default_value")
        assert value == "default_value"

    def test_agent_get_state_without_default(self):
        """Test get_state returns None for missing key without default"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        value = agent.get_state("nonexistent")
        assert value is None

    def test_agent_invoke_simple_question(self):
        """Test agent invoke with simple question - no tools"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        response = agent.invoke("What is 2+2?")

        assert isinstance(response, AgentResponse)
        assert response.answer == "Simple answer"
        assert response.confidence == 1.0
        assert len(response.trace.obj()) > 0

    def test_agent_invoke_with_tool_call(self, checkpointer):
        """Test agent invoke with single tool call"""
        model = ToolCallingModel(tool_name="MockTool", tool_args={"x": 5, "y": 3})
        tool = MockTool()

        agent = Agent(
            model=model, tools=[tool], router=SimpleRouter(), checkpointer=checkpointer
        )

        response = agent.invoke("Calculate something")

        assert response.answer == "Final answer after tool"
        assert len(response.trace.obj()) >= 3  # Model -> Tool -> Model

    def test_agent_invoke_records_trace(self):
        """Test that invoke records complete trace"""
        model = SimplestModel()
        agent = Agent(model=model, tools=[], router=SimpleRouter())

        response = agent.invoke("test question")

        trace_obj = response.trace.obj()
        assert len(trace_obj) > 0
        assert "model" in trace_obj[0][0]
        assert "input" in trace_obj[0][0]
        assert "output" in trace_obj[0][0]

    def test_agent_invoke_with_image(self):
        """Test agent invoke with image parameter"""
        agent = Agent(model=MockModel(), tools=[], router=SimpleRouter())

        response = agent.invoke("Describe this", image="image_data_here")

        # Check that image was included in context
        assert "Image:" in agent.model.contexts_received[0]
        assert "image_data_here" in agent.model.contexts_received[0]

    def test_agent_invoke_tool_not_found(self):
        """Test agent handles tool not found gracefully"""
        model = ToolCallingModel(tool_name="NonExistentTool", tool_args={})

        agent = Agent(
            model=model,
            tools=[MockTool()],  # Different tool
            router=SimpleRouter(),
        )

        response = agent.invoke("test")

        # Should force final answer after error (ToolCallingModel returns this when forced)
        assert response.answer == "Final answer after tool"
        # Should contain error message in trace
        assert any(
            "Error: Tool" in agent.model.contexts_received[i]
            for i in range(len(agent.model.contexts_received))
        )

    def test_agent_invoke_max_calls_reached(self):
        """Test agent stops at max_calls and forces final answer"""
        # Model that keeps calling tools
        script = [
            ToolCallingModelResponse(
                content="call 1",
                is_final=False,
                tool_name="MockTool",
                tool_args={"x": 1, "y": 1},
            ),
            ToolCallingModelResponse(
                content="call 2",
                is_final=False,
                tool_name="MockTool",
                tool_args={"x": 2, "y": 2},
            ),
            ToolCallingModelResponse(
                content="call 3",
                is_final=False,
                tool_name="MockTool",
                tool_args={"x": 3, "y": 3},
            ),
            ToolCallingModelResponse(
                content="call 4",
                is_final=False,
                tool_name="MockTool",
                tool_args={"x": 4, "y": 4},
            ),
            ToolCallingModelResponse(
                content="call 5",
                is_final=False,
                tool_name="MockTool",
                tool_args={"x": 5, "y": 5},
            ),
        ]
        model = MockModel(script=script)

        agent = Agent(
            model=model,
            tools=[MockTool()],
            router=SimpleRouter(),
            max_calls=3,  # Limit to 3 calls
        )

        response = agent.invoke("test")

        # Should have forced final answer
        assert "Max tool calls reached" in model.contexts_received[-1]
        assert response.answer == "Forced final answer"

    def test_agent_invoke_context_accumulation(self):
        """Test that context accumulates with each tool call"""
        model = MockModel(
            script=[
                ToolCallingModelResponse(
                    content="c1",
                    is_final=False,
                    tool_name="MockTool",
                    tool_args={"x": 1, "y": 1},
                ),
                ToolCallingModelResponse(
                    content="c2",
                    is_final=False,
                    tool_name="MockTool",
                    tool_args={"x": 2, "y": 2},
                ),
                ToolCallingModelResponse(content="final", is_final=True),
            ]
        )

        agent = Agent(model=model, tools=[MockTool()], router=SimpleRouter())

        response = agent.invoke("test")

        contexts = model.contexts_received
        # Each subsequent context should be longer
        assert len(contexts[1]) > len(contexts[0])
        assert len(contexts[2]) > len(contexts[1])
        # Should contain tool results
        assert "Output:" in contexts[1]

    def test_agent_invoke_no_tool_call_in_response(self):
        """Test agent stops when model doesn't request tool"""
        model = MockModel(
            script=[
                ToolCallingModelResponse(
                    content="answer", is_final=False
                )  # is_final=False but no tool
            ]
        )

        agent = Agent(model=model, tools=[MockTool()], router=SimpleRouter())

        response = agent.invoke("test")

        # Should stop after first call
        assert model.call_count == 1

    def test_agent_checkpoint_called_when_provided(self, temp_dir):
        """Test that checkpoint is called when checkpointer provided"""
        checkpointer = Checkpointer(temp_dir)
        agent = Agent(
            model=SimplestModel(),
            tools=[],
            router=SimpleRouter(),
            checkpointer=checkpointer,
        )

        response = agent.invoke("test")

        # Should have created a checkpoint file
        files = os.listdir(temp_dir)
        assert len(files) > 0
        assert any(f.startswith("trace_") and f.endswith(".json") for f in files)

    def test_agent_checkpoint_not_called_without_checkpointer(self):
        """Test checkpoint not called when checkpointer is None"""
        agent = Agent(
            model=SimplestModel(), tools=[], router=SimpleRouter(), checkpointer=None
        )

        # Should not raise error
        response = agent.invoke("test")
        assert response is not None

    def test_agent_checkpoint_manual_call(self, temp_dir):
        """Test manually calling checkpoint"""
        checkpointer = Checkpointer(temp_dir)
        agent = Agent(
            model=SimplestModel(),
            tools=[],
            router=SimpleRouter(),
            checkpointer=checkpointer,
        )

        agent.invoke("test")
        agent.checkpoint("manual_checkpoint.json")

        assert os.path.exists(Path(temp_dir) / "manual_checkpoint.json")

    def test_agent_checkpoint_with_custom_filename(self, temp_dir):
        """Test checkpoint with custom filename"""
        checkpointer = Checkpointer(temp_dir)
        agent = Agent(
            model=SimplestModel(),
            tools=[],
            router=SimpleRouter(),
            checkpointer=checkpointer,
        )

        agent.trace.push([ModelCall(model="m", input="i", output="o")])
        agent.checkpoint("custom_name.json")

        assert os.path.exists(Path(temp_dir) / "custom_name.json")

    def test_agent_checkpoint_without_checkpointer_does_nothing(self):
        """Test checkpoint does nothing when no checkpointer"""
        agent = Agent(
            model=SimplestModel(), tools=[], router=SimpleRouter(), checkpointer=None
        )

        # Should not raise error
        agent.checkpoint()

    def test_agent_multiple_tool_calls_in_sequence(self):
        """Test agent can make multiple sequential tool calls"""
        script = [
            ToolCallingModelResponse(
                content="1",
                is_final=False,
                tool_name="MockTool",
                tool_args={"x": 1, "y": 1},
            ),
            ToolCallingModelResponse(
                content="2",
                is_final=False,
                tool_name="CacheableTool",
                tool_args={"value": 5},
            ),
            ToolCallingModelResponse(content="final", is_final=True),
        ]
        model = MockModel(script=script)

        agent = Agent(
            model=model, tools=[MockTool(), CacheableTool()], router=SimpleRouter()
        )

        response = agent.invoke("test")

        # Should have 5 layers: Model -> Tool -> Model -> Tool -> Model
        assert len(response.trace.obj()) >= 5

    def test_agent_default_max_calls(self):
        """Test agent uses default max_calls of 5"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        assert agent.max_calls == 5

    def test_agent_with_empty_tools_list(self):
        """Test agent works with empty tools list"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        response = agent.invoke("test")
        assert response is not None
        assert response.answer is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_calculation(self, temp_dir):
        """Test complete workflow: question -> tool -> answer -> checkpoint"""
        checkpointer = Checkpointer(temp_dir)
        model = ToolCallingModel(tool_name="MockTool", tool_args={"x": 10, "y": 20})
        tool = MockTool()

        agent = Agent(
            model=model, tools=[tool], router=SimpleRouter(), checkpointer=checkpointer
        )

        response = agent.invoke("Calculate 10 + 20")

        # Verify response
        assert response.answer == "Final answer after tool"
        assert response.confidence == 1.0

        # Verify trace
        trace_obj = response.trace.obj()
        assert len(trace_obj) >= 3

        # Verify checkpoint
        files = os.listdir(temp_dir)
        assert len(files) == 1

        # Verify checkpoint content
        with open(Path(temp_dir) / files[0]) as f:
            saved_trace = json.load(f)

        assert len(saved_trace) == len(trace_obj)

    def test_tool_caching_across_calls(self):
        """Test tool caching works across multiple invocations"""
        tool = CacheableTool()
        model = MockModel(
            script=[
                ToolCallingModelResponse(
                    content="1",
                    is_final=False,
                    tool_name="CacheableTool",
                    tool_args={"value": 5},
                ),
                ToolCallingModelResponse(content="final", is_final=True),
            ]
        )

        agent = Agent(model=model, tools=[tool], router=SimpleRouter())

        # First invocation
        agent.invoke("test 1")
        assert tool.execution_count == 1

        # Second invocation with same args (new model, same tool)
        model2 = MockModel(
            script=[
                ToolCallingModelResponse(
                    content="2",
                    is_final=False,
                    tool_name="CacheableTool",
                    tool_args={"value": 5},
                ),
                ToolCallingModelResponse(content="final", is_final=True),
            ]
        )
        agent2 = Agent(model=model2, tools=[tool], router=SimpleRouter())

        agent2.invoke("test 2")
        # Should use cache, no new execution
        assert tool.execution_count == 1

    def test_state_persistence_across_invocations(self):
        """Test state persists across multiple invoke calls"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        agent.update_state("counter", 0)

        for i in range(3):
            agent.invoke(f"question {i}")
            counter = agent.get_state("counter")
            agent.update_state("counter", counter + 1)

        assert agent.get_state("counter") == 3

    def test_multiple_agents_with_shared_tools(self):
        """Test multiple agents can share the same tool instances"""
        shared_tool = CacheableTool()

        agent1 = Agent(
            model=ToolCallingModel("CacheableTool", {"value": 10}),
            tools=[shared_tool],
            router=SimpleRouter(),
        )

        agent2 = Agent(
            model=ToolCallingModel("CacheableTool", {"value": 10}),
            tools=[shared_tool],
            router=SimpleRouter(),
        )

        agent1.invoke("test 1")
        assert shared_tool.execution_count == 1

        # Agent 2 should use cached result
        agent2.invoke("test 2")
        assert shared_tool.execution_count == 1


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_question(self):
        """Test agent handles empty question"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())
        response = agent.invoke("")
        assert response is not None

    def test_very_long_question(self):
        """Test agent handles very long question"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())
        long_question = "x" * 10000
        response = agent.invoke(long_question)
        assert response is not None

    def test_special_characters_in_question(self):
        """Test agent handles special characters"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())
        response = agent.invoke("Test: {special} [chars] <tags> & symbols!")
        assert response is not None

    def test_tool_with_special_characters_in_output(self):
        """Test tool that returns special characters"""

        class SpecialTool(Tool):
            class Parameters(BaseModel):
                pass

            @property
            def description(self):
                return "Returns special chars"

            @property
            def parameters_schema(self):
                return self.Parameters

            def execute(self, **kwargs):
                return {
                    "result": "<tag> & {special} [chars]",
                    "raw_data": "<tag> & {special} [chars]",
                }

        tool = SpecialTool()
        result = tool.call()
        assert "<tag>" in result["result"]

    def test_tool_returning_none(self):
        """Test tool that returns None"""

        class NoneTool(Tool):
            class Parameters(BaseModel):
                pass

            @property
            def description(self):
                return "Returns None"

            @property
            def parameters_schema(self):
                return self.Parameters

            def execute(self, **kwargs):
                return {"result": "None value returned", "value": None}

        tool = NoneTool()
        result = tool.call()
        assert result["value"] is None

    def test_tool_with_nested_dict_output(self):
        """Test tool with complex nested output"""

        class NestedTool(Tool):
            class Parameters(BaseModel):
                pass

            @property
            def description(self):
                return "Returns nested data"

            @property
            def parameters_schema(self):
                return self.Parameters

            def execute(self, **kwargs):
                return {
                    "result": "Nested data structure",
                    "level1": {"level2": {"level3": ["a", "b", "c"]}},
                }

        tool = NestedTool()
        result = tool.call()
        assert result["level1"]["level2"]["level3"] == ["a", "b", "c"]

    def test_max_calls_zero(self):
        """Test agent with max_calls=0"""
        agent = Agent(
            model=ToolCallingModel("MockTool", {"x": 1, "y": 1}),
            tools=[MockTool()],
            router=SimpleRouter(),
            max_calls=0,
        )

        response = agent.invoke("test")
        # Should force final immediately
        assert "Max tool calls reached" in agent.model.contexts_received[-1]

    def test_max_calls_one(self):
        """Test agent with max_calls=1"""
        agent = Agent(
            model=ToolCallingModel("MockTool", {"x": 1, "y": 1}),
            tools=[MockTool()],
            router=SimpleRouter(),
            max_calls=1,
        )

        response = agent.invoke("test")
        # Should allow exactly one tool call
        assert response is not None

    def test_unicode_in_inputs(self):
        """Test handling unicode characters"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())
        response = agent.invoke("Hello 世界 مرحبا мир")
        assert response is not None

    def test_json_like_strings_in_parameters(self):
        """Test tool parameters that look like JSON"""

        class JsonParamTool(Tool):
            class Parameters(BaseModel):
                data: str = Field(default="", description="json string")

            @property
            def description(self):
                return "test"

            @property
            def parameters_schema(self):
                return self.Parameters

            def execute(self, data: str = ""):
                return {"result": f"Received: {data}", "received": data}

        tool = JsonParamTool()
        result = tool.call(data='{"nested": "json"}')
        assert result["received"] == '{"nested": "json"}'


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================


class TestPerformance:
    """Test performance and resource handling"""

    def test_large_trace(self):
        """Test handling large trace with many layers"""
        trace = Trace()
        for i in range(100):
            trace.push([ModelCall(model=f"m{i}", input=f"i{i}", output=f"o{i}")])

        json_str = trace.json()
        assert isinstance(json_str, str)
        assert len(trace.obj()) == 100

    def test_cache_with_many_entries(self):
        """Test tool cache with many entries"""
        tool = MockTool()

        for i in range(100):
            tool.call(x=i, y=i * 2)

        assert len(tool.cache) == 100

    def test_agent_state_with_many_entries(self):
        """Test agent state with many keys"""
        agent = Agent(model=SimplestModel(), tools=[], router=SimpleRouter())

        for i in range(100):
            agent.update_state(f"key{i}", f"value{i}")

        assert len(agent.state) == 100
        assert agent.get_state("key50") == "value50"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
