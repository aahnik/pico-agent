from pico_agent import Model
from huggingface import inference, AutoModelLLM

class QwenModel(Model):
    def __init__():
        self.model = AutoModelLLM('qwen-llm')
    def generate(**params):
        self.model.generate(**params)


class MockModel(Model):
    def generate():
        return "asfsdjfasijf"

from anthropic import generate
class ClaudeModel(Model):
    def __init__():
        self.api_key = api_key
    def generate():
        generate()



my_model = MyModel()

answer = my_model.generate(image=image, question=question)

from huggingface import XYZModel


class CounterTool:
    "This tool counts given object in image"

    def __init__(afdf):
        self.some = 10

    @cache
    def call(image='file.png', obj_desc:'ball') -->int:
        hugginface.call()

    def cache():




counter_tool = CounterTool(afadf)

my_checkpointer = FileCheckpointer("traces.json")

agent = MyAgent(
    model = my_model,
    tools = [counter_tool, color_picker],
    tool_engine = my_tool_engine
    checkpointer = my_checkpointer
)




if __name__ == '__main__':
    answer = agent.invoke(question="how many balls are there in yellow color", image = "/file/path.png")


