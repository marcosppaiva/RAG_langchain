import os
import warnings
from typing import Any, List

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

warnings.filterwarnings("ignore")

load_dotenv()


def read_pdf(file_path: str) -> str:
    text = ""
    for pdf in os.listdir(file_path):
        pdf_reader = PdfReader(os.path.join(file_path, pdf))
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text: str) -> List[str]:
    text_splitter = text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=800, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(text_chunks: List[str]) -> Any:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vectorstore: FAISS):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

    return conversation_chain


if __name__ == "__main__":
    text = read_pdf("data/")

    text_chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(text_chunks)

    chat = get_conversation_chain(vectorstore)

    response = chat({"question": "Who is Marcos?"})
    # response = chat({"question": "What is your profession?"})
    # response = chat({"question": "What is your last work experience?"})
    # response = chat({"question": "What was the last company he worked for?"})
    # response = chat({"question": "Which company does he currently work for?"})
    # response = chat({"question": "What types of projects has he worked on so far?"})

    print(response)
