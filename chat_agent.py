# chat_agent.py
import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from index_wikipages import index_wikipedia_pages

@cl.on_chat_start
async def main():
    await cl.Message(
        content=(
            "Hello! Please specify the Wikipedia page(s) you want to index by typing "
            "'Please index: [page(s)]'. For example:\n"
            "Please index: Python (programming language), Artificial intelligence"
        )
    ).send()

@cl.on_message
async def handle_message(message):
    user_input = message

    if "please index:" in user_input.lower():
        # Index the Wikipedia pages
        index_wikipedia_pages(user_input)
        await cl.Message(
            content="Wikipedia pages have been indexed. You can now ask questions."
        ).send()
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local("faiss_index", embeddings)

        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
            retriever=vectorstore.as_retriever(),
        )

        history = cl.user_session.get("history")
        if history is None:
            history = []

        chat_history = []
        for human_msg, ai_msg in history:
            chat_history.append(("human", human_msg))
            chat_history.append(("ai", ai_msg))

        result = qa({"question": user_input, "chat_history": chat_history})

        history.append((user_input, result["answer"]))
        cl.user_session.set("history", history)

        await cl.Message(content=result["answer"]).send()
