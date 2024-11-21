# index_wikipages.py
import os
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_apikey

INDEX_DIR = "faiss_index"
OPENAI_API_KEY = get_apikey()


def parse_wikipage_list(query):
    """
    Parses a comma-separated string of Wikipedia page titles into a list.
    """
    page_titles = [title.strip() for title in query.split(",") if title.strip()]
    return page_titles


def fetch_wikipedia_content(page_titles):
    """
    Fetches the content of the specified Wikipedia pages.
    """
    documents = []
    for title in page_titles:
        try:
            # Prevent Wikipedia from auto-suggesting alternate titles
            page = wikipedia.page(title, auto_suggest=False)
            content = page.content
            documents.append({"title": title, "content": content})
            print(f"Fetched content for '{title}'.")
        except DisambiguationError as e:
            print(f"DisambiguationError: The title '{title}' is ambiguous. Suggestions: {e.options}")
        except PageError:
            print(f"PageError: The page '{title}' does not exist. Please check the title and try again.")
        except Exception as e:
            print(f"Unexpected error fetching '{title}': {e}")
    return documents


def split_documents(documents):
    """
    Splits the fetched Wikipedia content into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    metadatas = []
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        docs.extend(splits)
        metadatas.extend([{"title": doc["title"]}] * len(splits))
        print(f"Split document '{doc['title']}' into {len(splits)} chunk(s).")
    return docs, metadatas


def create_or_update_vectorstore(docs, metadatas):
    """
    Creates a new FAISS vector store or updates an existing one with new documents.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, 'faiss.index')) and os.path.exists(os.path.join(INDEX_DIR, 'vectorstore.pkl')):
        # Load existing index and add new documents
        try:
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_texts(docs, metadatas=metadatas)
            print(f"Added {len(docs)} new document chunk(s) to the existing index.")
        except Exception as e:
            print(f"Error updating existing vectorstore: {e}")
            raise e
    else:
        # Create a new vectorstore
        try:
            vectorstore = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
            print(f"Created a new index with {len(docs)} document chunk(s).")
        except Exception as e:
            print(f"Error creating new vectorstore: {e}")
            raise e

    # Save the vectorstore to disk
    try:
        vectorstore.save_local(INDEX_DIR)
        print(f"Vectorstore saved to '{INDEX_DIR}'.")
    except Exception as e:
        print(f"Error saving vectorstore: {e}")
        raise e

    return vectorstore


def index_wikipedia_pages(user_input):
    """
    Main function to index Wikipedia pages based on user input.
    """
    page_titles = parse_wikipage_list(user_input)
    if not page_titles:
        print("No valid Wikipedia pages to index.")
        return None
    documents = fetch_wikipedia_content(page_titles)
    if not documents:
        print("Failed to fetch any Wikipedia content.")
        return None
    docs, metadatas = split_documents(documents)
    if not docs:
        print("No documents to index after splitting.")
        return None
    vectorstore = create_or_update_vectorstore(docs, metadatas)
    print(f"Indexed {len(docs)} document chunk(s).")
    return vectorstore
