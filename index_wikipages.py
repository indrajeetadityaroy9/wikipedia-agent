import os
import warnings
from bs4 import GuessedAtParserWarning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
import wikipedia
from llama_index.readers.wikipedia import WikipediaReader
from wikipedia.exceptions import DisambiguationError
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel
from llama_index.program.openai import OpenAIPydanticProgram
import openai
from utils import get_apikey
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

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
    return wikipage_requests.pages  # Ensure to return the list of pages

def prioritize_option(options, original_query):
    """
    Prioritizes disambiguation options based on the original query.

    Args:
        options (list): List of possible Wikipedia page titles.
        original_query (str): The original user query.

    Returns:
        str: The selected Wikipedia page title.
    """

    normalized_query = original_query.lower()
    
    for option in options:
        if normalized_query in option.lower():
            return option
    
    return options[0]

def create_wikidocs(wikipage_requests, original_query):
    """
    Creates documents from Wikipedia pages, handling disambiguations.

    Args:
        wikipage_requests (list): List of Wikipedia page titles to index.
        original_query (str): The original user query.

    Returns:
        list: List of Document objects.
    """
    reader = WikipediaReader()
    documents = []
    for page in wikipage_requests:
        try:
            wiki_pages = reader.load_data(pages=[page])  # Returns a list of Document objects
            documents.extend(wiki_pages)  # Flatten the list
        except DisambiguationError as e:
            selected_page = prioritize_option(e.options, original_query)  # Prioritize based on query
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
    Initializes the FAISS vector store with a FAISS index.

    Returns:
        FaissVectorStore: An instance of the FAISS vector store.
    """
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore

    dimension = 768

    # Create a FAISS index
    faiss_index = faiss.IndexFlatL2(dimension)
    # Initialize the FaissVectorStore with the created FAISS index
    faiss_vector_store = FaissVectorStore(faiss_index=faiss_index)

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
    
    if os.path.exists(index_file_path):
        faiss_vector_store.load_index(index_file_path)
        print(f"Loaded existing FAISS index from {index_file_path}")
    else:
        faiss_vector_store.add_documents(nodes)
        faiss_vector_store.build_index()
        faiss_vector_store.save_index(index_file_path)
        print(f"Created and saved new FAISS index to {index_file_path}")
    
    index = VectorStoreIndex(nodes, vector_store=faiss_vector_store)
    return index