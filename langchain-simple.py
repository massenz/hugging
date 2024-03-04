# Created by M. Massenzio, 2024

from common import read_env
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

env = read_env()
if 'hf_token' not in env:
    raise ValueError("Hugging Face API token not found in environment")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = env['hf_token']

template = '''
Question: {question}
Answer: 
'''

prompt = PromptTemplate(
    template=template, input_variables=['question']
)

hub_llm = HuggingFaceEndpoint(
    repo_id=env['model'],
    temperature=1,
    model_kwargs={"max_length": 64}
)

llm_chain = LLMChain(
    prompt=prompt, llm=hub_llm
)

# Prompt the user for a question
question = input("Ask a question: ")

# Generate the answer
answer = llm_chain.invoke({'question': question})

print(f"Answer: {answer['text']}")
