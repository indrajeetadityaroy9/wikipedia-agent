# Wikipedia Agent

An LLM backed conversational agent equipped with access to Wikipedia, allowing responses based on a knowledge base built from selected Wikipedia pages.

The agent operates with the ReAct prompt framework. This framework enables the agent to use tools step by step to answer questions. Essentially, it understands the question, selects a tool, reviews the tool’s result, and then decides whether to answer or try the tool again based on that result.
