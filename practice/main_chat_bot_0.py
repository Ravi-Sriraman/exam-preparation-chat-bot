from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, List
from dotenv import load_dotenv


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def main():
    load_dotenv()
    llm = init_chat_model(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    pdf_loader = PyPDFLoader("../documents/MSc_PGd_DA BIBA Project 80_ Prov.pdf")
    documents = pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = splitter.split_documents(documents)
    _ = vector_store.add_documents(all_splits)
    prompt = hub.pull("rlm/rag-prompt")

    def retrieve(state: State):
        result = vector_store.similarity_search(state["question"])
        return {"context": result}

    def generate(state: State):
        context = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": context})
        answer = llm.invoke(messages)
        return {"answer": answer}

    graph = StateGraph(State).add_sequence([retrieve, generate]).add_edge(START, "retrieve").compile()
    response = graph.invoke({"question": "What is the percentage of marks awarded to live presentation?"})
    print(response["answer"])


if __name__ == '__main__':
    main()
