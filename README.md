# Wikipedia Agent

A conversational agent that utilizes Wikipedia as its knowledge base, enabling responses based on selected Wikipedia pages.

The agent operates with the ReAct prompt framework. This framework enables the agent to use tools step by step to answer questions. Essentially, it understands the question, selects a tool, reviews the tool’s result, and then decides whether to answer or try the tool again based on that result.

**Performance Optimizations:**
  - Optimized text splitting parameters (`chunk_size` and `chunk_overlap`) for better retrieval performance and relevance.
  - Limited the number of retrieved documents (`k`) to balance response relevance and performance.
