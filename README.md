# pico-agent

> pico (p) = $10^-12$ (trillionth)

A very small and simple framework(defines the way of thinking/interacting) for building language model agents.
The primary purpose is to lay out a clean and quickly understandable abstraction for agents. You don't need to read large docs, or fight with the framework to customize it.

This is suited for when you will be probably working on some custom experiment, where you define your own agentic flows, and tools. You can implement your own model by inheriting the `Model` interface. To call any llm model you can call via their API, inference locally, by using any popular library.


If you need integrations with lots of popular tools, models and agentic workflows then something like langchain/langraph is more suitable. But if you are working on some custom project, where you need to hack lots of internal bits, to customize everything, welcome to pico-agent!

Contribute your ideas/feedback via creating issues. I plan to hand-write this small library to have absolute control and a solid mental map of everything. _Not accepting code from LLMs or other humans_.
