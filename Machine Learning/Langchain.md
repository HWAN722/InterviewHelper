# LangChain Common APIs and Their Purposes

LangChain is a powerful framework designed to help developers build applications with language models more easily and effectively. Here are some of its key APIs:

## 1. `LLMChain`
- **Purpose**: Simplifies calling a language model with a specific prompt.
- **Use Case**: When you have a template prompt and want to fill in variables dynamically to generate responses.
```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI()
prompt = PromptTemplate(template="What is the capital of {country}?", input_variables=["country"])
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run(country="France")
print(response)
```

## 2. `PromptTemplate`
- **Purpose**: Defines a reusable and parameterized prompt template.
- **Use Case**: Create consistent prompts where only certain fields change at runtime.
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Tell me a {adjective} joke about {topic}.",
    input_variables=["adjective", "topic"]
)

print(prompt.format(adjective="funny", topic="cats"))
```

## 3. `ChatOpenAI`
- **Purpose**: Interface for interacting with OpenAI's chat models (like GPT-3.5, GPT-4).
- **Use Case**: Send structured conversations to chat models instead of plain text prompts.
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI()
messages = [HumanMessage(content="What's the weather today?")]
response = chat(messages)
print(response.content)

```

## 4. `Memory`
- **Purpose**: Stores the conversation history or other contextual information.
- **Types**: 
  - `ConversationBufferMemory` (stores full conversation)
  - `ConversationSummaryMemory` (summarizes and stores)
- **Use Case**: Maintain state across multiple turns of conversation.
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})

print(memory.load_memory_variables({}))
```

## 5. `Agent`
- **Purpose**: Allows the LLM to make decisions about which actions to take based on intermediate steps.
- **Use Case**: Build multi-step reasoning applications (e.g., answering a query by searching and calculating).
```python
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI

llm = OpenAI()
tools = load_tools(["serpapi"])  # Example: web search tool
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

agent.run("What's the capital of Germany?")
```

## 6. `Tool`
- **Purpose**: External functions or services that an agent can call.
- **Use Case**: Plug in APIs like web search, database lookup, math solvers into an LLM-driven workflow.
```python
from langchain.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    return a + b

print(add_numbers.run(a=3, b=4))
```

## 7. `Retriever`
- **Purpose**: Fetch relevant documents based on a query.
- **Use Case**: Build retrieval-augmented generation (RAG) systems, where the model retrieves facts before answering.
```python
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("path_to_faiss", embeddings)

retriever = db.as_retriever()
docs = retriever.get_relevant_documents("What is LangChain?")
print(docs)
```

## 8. `Document`
- **Purpose**: A standard structure for text chunks with associated metadata.
- **Use Case**: Store and organize large text corpora for search and retrieval.
```python
from langchain.schema import Document

doc = Document(page_content="LangChain enables LLM apps.", metadata={"source": "documentation"})
print(doc.page_content, doc.metadata)
```

## 9. `Vectorstore`
- **Purpose**: Store and search documents based on their vector embeddings.
- **Common Vectorstores**: FAISS, Chroma, Pinecone.
- **Use Case**: Semantic search, where similar meanings (not just keywords) are matched.
```python
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(["Hello world", "Hi there"], embedding=embeddings)

query_result = db.similarity_search("greetings", k=1)
print(query_result)
```

## 10. `Chain`
- **Purpose**: A generic interface for any workflow involving LLMs.
- **Use Case**: Compose multiple steps (prompt → LLM → memory → retriever → output) into a single pipeline.
```python
from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI

llm = OpenAI()
chain1 = LLMChain(llm=llm, prompt=PromptTemplate.from_template("Translate {text} to French."))
chain2 = LLMChain(llm=llm, prompt=PromptTemplate.from_template("Summarize: {text}"))

overall_chain = SimpleSequentialChain(chains=[chain1, chain2])

result = overall_chain.run("I love studying AI.")
print(result)
```
---

# Quick Summary Table

| API             | Purpose                                         | Example Use Case                         |
|-----------------|--------------------------------------------------|------------------------------------------|
| `LLMChain`      | Run a model with a prompt template               | Q&A chatbot                             |
| `PromptTemplate`| Create structured prompts                       | Filling in variables in prompts         |
| `ChatOpenAI`    | Interface with OpenAI chat models                | Conversational agents                   |
| `Memory`        | Keep conversation history                       | Multi-turn chat applications            |
| `Agent`         | Enable decision making and tool calling         | Search-then-answer bots                 |
| `Tool`          | External APIs callable by models                | Math calculator, search tool            |
| `Retriever`     | Fetch relevant documents                        | Knowledge base question answering       |
| `Document`      | Structure text and metadata                     | Organizing retrieved search results     |
| `Vectorstore`   | Semantic search with embeddings                 | Search documents by meaning             |
| `Chain`         | Create multi-step LLM workflows                 | Summarization + Q&A pipeline             |

