# index_wikipages.py

import re
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def parse_wikipage_list(query):
    match = re.search(r'please index:\s*(.*)', query, re.IGNORECASE)
    if match:
        index_part = match.group(1).strip()
        pages = [page.strip() for page in index_part.split(",")]
        return pages
    else:
        return []


def fetch_wikipedia_content(page_titles):
    documents = []
    for title in page_titles:
        try:
            page = wikipedia.page(title)
            content = page.content
            documents.append({"title": title, "content": content})
        except (DisambiguationError, PageError) as e:
            print(f"Error fetching '{title}': {e}")
    return documents


def split_documents(documents):
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
    return docs, metadatas


def create_vectorstore(docs, metadatas):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    vectorstore.save_local("faiss_index")
    return vectorstore


def index_wikipedia_pages(query):
    page_titles = parse_wikipage_list(query)
    if not page_titles:
        print("No valid Wikipedia pages to index.")
        return None
    documents = fetch_wikipedia_content(page_titles)
    if not documents:
        print("Failed to fetch any Wikipedia content.")
        return None
    docs, metadatas = split_documents(documents)
    vectorstore = create_vectorstore(docs, metadatas)
    return vectorstore
