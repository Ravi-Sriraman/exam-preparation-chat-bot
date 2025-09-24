
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import UnstructuredPowerPointLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, List


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def main():
    load_dotenv()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = init_chat_model(model="gpt-4o-mini")
    loader = DirectoryLoader("./ppts/", loader_cls=UnstructuredPowerPointLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = splitter.split_documents(documents)
    vector_store = InMemoryVectorStore(embeddings)
    _ = vector_store.add_documents(all_splits)
    prompt = hub.pull("rlm/rag-prompt")

    def retrieve(state: State):
        docs = vector_store.similarity_search(state["question"])
        return {"context": docs}

    def generate(state: State):
        context = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": context})
        response = llm.invoke(messages)
        return {"answer": response}

    graph = StateGraph(State).add_sequence([retrieve, generate]).add_edge(START, "retrieve").compile()
    response = graph.invoke({"question": input("Ask your question: ")})
    print(response["answer"])


if __name__ == '__main__':
    main()