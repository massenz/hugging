# Created by M. Massenzio, 2024

from common import read_env
import os
import gradio as gr
from llama_index.core import (
    SimpleDirectoryReader, GPTVectorStoreIndex, PromptHelper,
    StorageContext, load_index_from_storage
)
from llama_index.legacy.llm_predictor import LLMPredictor
from langchain_openai import ChatOpenAI


# Load the environment variables
env = read_env()
if 'oai_token' not in env:
    raise ValueError("OpenAI API key not found in environment")
os.environ['OPENAI_API_KEY'] = env['oai_token']

MODEL = env.get('oai_model')
if not MODEL:
    raise ValueError("OpenAI model not found in environment")

INDEXES = os.sep.join([os.getenv("HOME"), ".cache", env.get("indexes_dir", "indexes")])


def index_documents(folder):
    max_input_size = 4096
    num_outputs = 512
    chunk_overlap_ratio = 0.1
    chunk_size_limit = 600

    # create the PromptHelper object
    prompt_helper = PromptHelper(
        max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit
        )

    # wraps around an LLM
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.7, model_name=MODEL, max_tokens=num_outputs
            )
    )

    # load the documents
    documents = SimpleDirectoryReader(folder).load_data()

    # perform the indexing
    index = GPTVectorStoreIndex.from_documents(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    # save the index in current directory
    index.storage_context.persist(persist_dir=INDEXES)


class Chatbot:
    """A simple chatbot that uses the indexed documents to answer questions

    It is implemented as a callable object, so it can be used as a function for Gradio.

    :param folder: the folder containing the documents to index
    """
    def __init__(self, folder):
        self.query_engine = self._make_query_engine(folder)

    def __call__(self, input_text):
        return self.query_engine.chat(input_text)

    @staticmethod
    def _make_query_engine(folder):
        storage_context = StorageContext.from_defaults(persist_dir=folder)
        index = load_index_from_storage(storage_context)
        return index.as_chat_engine()


def show_ui():
    chatbot = Chatbot(INDEXES)
    gr.Interface(
        fn=chatbot, title="Ask me about Banking & Finance",
        inputs="text", outputs="text").launch()


if __name__ == '__main__':
    docs = input("Folder with training docs (leave empty to skip): ")
    if not docs:
        print("Skipping indexing")
    else:
        try:
            if os.stat(docs).st_size == 0:
                print("Folder is empty")
                exit()
        except FileNotFoundError:
            print("Folder not found")
            exit()
        try:
            print("Indexing documents...")
            index_documents(docs)
            print("Indexing complete")
        except Exception as e:
            print(f"Indexing failed: {e}")
            exit()

    show_ui()
