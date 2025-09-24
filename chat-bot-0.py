from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model

# Define State for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def main():
    load_dotenv()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    llm = init_chat_model(model="gpt-4o-mini")

    loader = WebBaseLoader(
        web_path = "https://lilianweng.github.io/posts/2023-06-23-agent/",
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load();
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)

    prompt = hub.pull("rlm/rag-prompt")

    # Define Application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = '\n\n'.join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question": "What is task decomposition"})
    print(response["answer"].content)

if __name__ == '__main__':
    main()