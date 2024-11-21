# chat_agent.py

import chainlit as cl
import os
import shutil
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from index_wikipages import index_wikipedia_pages, parse_wikipage_list
from utils import get_apikey

# Configure logging with timestamp and log level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
INDEX_DIR = "faiss_index"
OPENAI_API_KEY = get_apikey()


@cl.on_chat_start
async def main():
    """
    Initializes the chat session by setting up an empty history.
    Sends a welcome message with instructions and available commands.
    """
    # Initialize chat history in the user session
    cl.user_session.set("history", [])

    await cl.Message(
        content=(
            "👋 Hello! Welcome to the Wikipedia Chatbot.\n\n"
            "🔍 **How to Get Started:**\n"
            "Please enter the titles of the Wikipedia pages you want to index, separated by commas.\n"
            "**Example:** `Python (programming language), Artificial intelligence`\n\n"
            "📌 **Once indexed**, you can ask me questions about these topics!\n\n"
            "💡 **Available Commands:**\n"
            "- `/index [page1], [page2], ...` : Index specified Wikipedia pages.\n"
            "- `/reset` : Reset the current index and clear chat history.\n"
            "- `/help` : Show this help message.\n\n"
            "**Tip:** You can index more pages anytime by simply entering their titles or using the `/index` command."
        )
    ).send()


@cl.on_message
async def handle_message(message):
    """
    Handles incoming messages from the user.
    Supports indexing pages, resetting the index, showing help, and answering queries.
    """
    try:
        user_input = message.content.strip()

        if not user_input:
            await cl.Message(content="❌ Please enter a valid Wikipedia page title or ask a question.").send()
            return

        if user_input.startswith("/index"):
            pages_to_index = user_input.replace("/index", "").strip()
            if not pages_to_index:
                await cl.Message(content="❌ Please specify page titles after the `/index` command.").send()
                return
            await cl.Message(content="🔄 Indexing the provided Wikipedia pages... This may take a few moments.").send()
            try:
                index_wikipedia_pages(pages_to_index)
                num_pages = len(parse_wikipage_list(pages_to_index))
                await cl.Message(
                    content=f"✅ Successfully indexed {num_pages} Wikipedia page(s). You can now ask questions about them.").send()
            except Exception as e:
                logger.error(f"Error during indexing: {e}", exc_info=True)
                await cl.Message(content=f"❌ An error occurred while indexing: {str(e)}").send()
            return

        elif user_input.startswith("/reset"):
            try:
                if os.path.exists(INDEX_DIR):
                    shutil.rmtree(INDEX_DIR)
                    logger.info(f"Index directory '{INDEX_DIR}' has been removed.")
                cl.user_session.set("history", [])
                await cl.Message(
                    content="🧹 The index has been reset. Please index new pages using the `/index` command.").send()
            except Exception as e:
                logger.error(f"Error during reset: {e}", exc_info=True)
                await cl.Message(content=f"❌ An error occurred while resetting: {str(e)}").send()
            return

        elif user_input.startswith("/help"):
            await cl.Message(
                content=(
                    "📚 **Available Commands:**\n"
                    "- `/index [page1], [page2], ...` : Index specified Wikipedia pages.\n"
                    "- `/reset` : Reset the current index and clear chat history.\n"
                    "- `/help` : Show this help message."
                )
            ).send()
            return

        # Check if the FAISS index directory exists
        if not os.path.exists(INDEX_DIR):
            await cl.Message(content="🔄 Indexing the provided Wikipedia pages... This may take a few moments.").send()
            try:
                index_wikipedia_pages(user_input)
                await cl.Message(content="✅ Wikipedia pages have been indexed. You can now ask questions.").send()
            except Exception as e:
                logger.error(f"Error during indexing: {e}", exc_info=True)
                await cl.Message(content=f"❌ An error occurred while indexing: {str(e)}").send()
            return
        else:
            # Load the vectorstore with deserialization allowed
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                logger.info("Vectorstore loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading vectorstore: {e}", exc_info=True)
                await cl.Message(content=f"❌ An error occurred while loading the index: {str(e)}").send()
                return

            # Create the QA chain using ChatOpenAI
            try:
                qa = ConversationalRetrievalChain.from_llm(
                    ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                )
                logger.info("ConversationalRetrievalChain initialized successfully.")
            except Exception as e:
                logger.error(f"Error creating QA chain: {e}", exc_info=True)
                await cl.Message(content=f"❌ An error occurred while setting up the assistant: {str(e)}").send()
                return

            # Get chat history
            history = cl.user_session.get("history")
            if history is None:
                history = []

            # Run the chain
            try:
                result = qa({"question": user_input, "chat_history": history})
                answer = result["answer"]
                logger.info(f"User Question: {user_input}")
                logger.info(f"Assistant Answer: {answer}")
            except Exception as e:
                logger.error(f"Error during QA chain execution: {e}", exc_info=True)
                answer = f"❌ An error occurred while processing your request: {str(e)}"

            # Update chat history
            history.append((user_input, answer))
            cl.user_session.set("history", history)

            await cl.Message(content=answer).send()

    except Exception as e:
        logger.error(f"Unhandled error in handle_message: {e}", exc_info=True)
        await cl.Message(content=f"❌ An unexpected error occurred: {str(e)}").send()
