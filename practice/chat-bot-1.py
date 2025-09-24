import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
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
    vector_store = InMemoryVectorStore(embeddings)
    web_base_loader = WebBaseLoader(
        web_path = "https://lilianweng.github.io/posts/2023-06-23-agent/",
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    prompt = hub.pull("rlm/rag-prompt")
    documents = web_base_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents=documents)
    _ = vector_store.add_documents(all_splits)

    def retrieve(state: State):
        docs = vector_store.similarity_search(state["question"])
        return {"context": docs}

    def generate(state: State):
        context = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": context})
        response = llm.invoke(messages)
        return {"answer": response}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question": "What is task decomposition?"})
    print(response["answer"])

if __name__ == '__main__':
    main()