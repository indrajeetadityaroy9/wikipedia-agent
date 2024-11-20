import os
import warnings
from bs4 import GuessedAtParserWarning
# Suppress the specific warning from BeautifulSoup
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
import openai
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.agent.react.base import ReActAgent                   
from llama_index.llms.openai.base import OpenAI
import chainlit as cl
from chainlit.input_widget import Select, TextInput
from index_wikipages import create_index
from utils import get_apikey
from wikipedia.exceptions import DisambiguationError
from llama_index.vector_stores.faiss import FaissVectorStore

agent = None
index = None

class WikiPageList(BaseModel):
    "Data model for WikiPageList"
    pages: list

def wikipage_list(query):
    openai.api_key = get_apikey()

    prompt_template_str = """
    Given the user input, identify the Wikipedia pages to be indexed.
    The user may mention pages after the phrase "please index:".
    Extract these page titles and return them as a list.
    If only one page is mentioned, return a single-element list.
    Examples:
    1. "I want to learn about Texas. Please index: Texas (U.S. state)"
       Output: ["Texas (U.S. state)"]
    2. "Tell me about Python. Please index: Python (programming language), Pythonidae"
       Output: ["Python (programming language)", "Pythonidae"]
    """
    program = OpenAIPydanticProgram.from_defaults(
        output_cls=WikiPageList,
        prompt_template_str=prompt_template_str,
        verbose=True,
    )
    wikipage_requests = program(query=query)
    return wikipage_requests.pages

def prioritize_option(options, query):
    for option in options:
        if query.lower() in option.lower():
            return option
    return options[0]

def create_react_agent(MODEL):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=wikisearch_engine(index),
            metadata=ToolMetadata(
                name="Wikipedia Search",
                description="Useful for performing searches on the Wikipedia knowledgebase",
            ),
        )
    ]

    openai.api_key = get_apikey()
    llm = OpenAI(model=MODEL)
    agent_instance = ReActAgent.from_tools(
        tools=query_engine_tools,
        llm=llm,
        verbose=True,
    )
    return agent_instance

def create_wikidocs(wikipage_requests, original_query):
    reader = WikipediaReader()
    documents = []
    for page in wikipage_requests:
        try:
            wiki_pages = reader.load_data(pages=[page])
            documents.extend(wiki_pages)
        except DisambiguationError as e:
            selected_page = prioritize_option(e.options, original_query)
            print(f"DisambiguationError: '{page}' may refer to multiple pages. Selecting '{selected_page}'.")
            try:
                wiki_pages = reader.load_data(pages=[selected_page])
                documents.extend(wiki_pages)
            except Exception as inner_e:
                print(f"Failed to retrieve page '{selected_page}': {inner_e}")
        except Exception as e:
            print(f"An error occurred while fetching page '{page}': {e}")
    return documents

def initialize_faiss_vector_store():
    """
    Initializes the FAISS vector store.

    Returns:
        FaissVectorStore: An instance of the FAISS vector store.
    """
    faiss_vector_store = FaissVectorStore()
    return faiss_vector_store

def save_faiss_index(index, file_path="faiss_index.bin"):
    """
    Saves the FAISS index to a binary file.

    Args:
        index (VectorStoreIndex): The vector store index to save.
        file_path (str): The path to save the FAISS index file.
    """
    faiss_vector_store = index.vector_store
    faiss_vector_store.save_index(file_path)
    print(f"FAISS index saved to {file_path}")

def load_faiss_index(faiss_vector_store, file_path="faiss_index.bin"):
    """
    Loads the FAISS index from a binary file.

    Args:
        faiss_vector_store (FaissVectorStore): The FAISS vector store instance.
        file_path (str): The path to load the FAISS index file.

    Returns:
        bool: True if loaded successfully, False otherwise.
    """
    if os.path.exists(file_path):
        faiss_vector_store.load_index(file_path)
        print(f"Loaded existing FAISS index from {file_path}")
        return True
    else:
        print(f"No existing FAISS index found at {file_path}")
        return False

def create_index(query, index_file_path="faiss_index.bin"):
    """
    Creates or loads a vector store index using FAISS based on the provided query.

    Args:
        query (str): The user query containing Wikipedia page requests.
        index_file_path (str): The path to load/save the FAISS index file.

    Returns:
        VectorStoreIndex: The created or loaded vector store index.
    """
    global index
    wikipage_requests = wikipage_list(query)
    if not wikipage_requests:
        print("No Wikipedia pages found to index.")
        return None
    documents = create_wikidocs(wikipage_requests, query)
    if not documents:
        print("No documents were created from the Wikipedia pages.")
        return None
    text_splits = SentenceSplitter(chunk_size=150, chunk_overlap=45)
    nodes = text_splits.get_nodes_from_documents(documents)         
    # Initialize FAISS vector store
    faiss_vector_store = initialize_faiss_vector_store()
    
    # Check if index file exists
    if os.path.exists(index_file_path):
        faiss_vector_store.load_index(index_file_path)
        print(f"Loaded existing FAISS index from {index_file_path}")
    else:
        # Add documents to FAISS
        faiss_vector_store.add_documents(nodes)
        faiss_vector_store.build_index()
        faiss_vector_store.save_index(index_file_path)
        print(f"Created and saved new FAISS index to {index_file_path}")
    # Create VectorStoreIndex with FAISS
    index = VectorStoreIndex(nodes, vector_store=faiss_vector_store)
    
    return index

def wikisearch_engine(index):
    query_engine = index.as_query_engine(
        response_mode="compact", verbose=True, similarity_top_k=10
    )
    return query_engine

@cl.on_chat_start
async def on_chat_start():
    global index
    settings = await cl.ChatSettings(
        [
            Select(
                id="MODEL",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo"],
                initial_index=0,
            ),
            TextInput(id="WikiPageRequest", label="Request Wikipage"),
        ]
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    global agent
    global index
    query = settings["WikiPageRequest"]
    if not isinstance(query, str):
        query = str(query)
    index = create_index(query)
    if index is None:
        await cl.Message(
            author="Agent", content=f"""Failed to index Wikipage(s) "{query}"."""
        ).send()
        return
    print("Index created for query:", query)

    print("on_settings_update", settings)
    MODEL = settings["MODEL"]
    if not isinstance(MODEL, str):
        MODEL = str(MODEL)
    agent = create_react_agent(MODEL)
    await cl.Message(
        author="Agent", content=f"""Wikipage(s) "{query}" successfully indexed"""
    ).send()

@cl.on_message
async def main(message: str):
    global agent
    if agent:
        print("Agent is available, processing message.")
        response = await cl.make_async(agent.chat)(message)
        await cl.Message(author="Agent", content=response).send()
    else:
        print("Agent is not available.")
        await cl.Message(author="Agent", content="Agent is not initialized yet. Please set up the agent first.").send()