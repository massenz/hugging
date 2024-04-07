# Created by M. Massenzio, 2024

from common import read_env, print_memory
import os

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

env = read_env()
if 'hf_token' not in env:
    raise ValueError("Hugging Face API token not found in environment")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = env['hf_token']

template = '''
Current conversation context: {history}
Question: {question}
AI: 
'''

prompt = PromptTemplate(
    template=template, input_variables=['question', 'history']
)

hub_llm = HuggingFaceEndpoint(
    repo_id=env['hf_model'],
    temperature=1,
    model_kwargs={"max_length": 64}
)

memory = ConversationBufferMemory(max_size=5)

llm_chain = ConversationChain(
    llm=hub_llm, verbose=False, memory=memory
)

question = ''
while True:
    try:
        # Prompt the user for a question
        question = input("Ask a question ([Ctrl-D] to quit): ")
        # Generate the answer
        # answer = llm_chain.invoke({'question': question})
        # print(f"Answer: {answer['text']}")
        print(llm_chain.invoke({'input': question})['response'])
    except EOFError:
        break

print("This is a record of the conversation:")
print_memory(memory)
print("Goodbye!")
