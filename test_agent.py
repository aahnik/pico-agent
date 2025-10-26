from agent import Checkpointer, Trace, TraceLayer, ToolCall, ModelCall
import json
from pathlib import Path


def mock_generate():
    return "hello world"


def test_generate():
    assert mock_generate() == "hello world"


def test_tool_call():
    tc = ToolCall(tool="tool_a", input={"cats": 2, "sofa": 2}, output={"sum": 4})
    assert tc.json()


def test_model_call():
    mc = ModelCall(model="org/abc-model", input="how are you", output="i am fine")
    assert mc.json()


def get_mock_trace():
    # simulate a dummy trace
    trace = Trace()
    trace.push(
        [ModelCall(model="org/abc-model", input="how are you", output="i am fine")]
    )
    trace.push(
        [ToolCall(tool="tool_a", input={"cats": 2, "sofa": 2}, output={"sum": 4})]
    )
    trace.push(
        [
            ModelCall(
                model="org/abc-model",
                input="how are you ... i am fine ... cats 2 sofa 2 ... sum 4",
                output="ok",
            )
        ]
    )
    return trace


def test_trace():
    trace = get_mock_trace()
    assert isinstance(trace, Trace)
    assert trace.json()
    assert trace.obj()


def test_checkpointing():
    checkpointer = Checkpointer("test")
    trace = get_mock_trace()
    checkpointer.write("test.json", trace)
    with open(checkpointer.dir / "test.json", mode="r") as file:
        json_string = file.read()
        obj = json.loads(json_string)
        assert trace.obj() == obj
