import os
from dotenv import load_dotenv
from collections import deque
from typing import Dict, List, Optional, Any


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, PromptTemplate, SerpAPIWrapper, LLMChain
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.experimental import BabyAGI
from tqdm import tqdm


# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv("OPENAI_API_KEY")


# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

embeddings = OpenAIEmbeddings(disallowed_special=())

try:
    vectorstore.load_local('vectorstore_nodejs-task-app-restapi', embeddings)
    print("Loaded vectorstore from local")
except:
    print("Building vectorstore from scratch")
    root_dir = './nodejs-task-app-restapi'
    docs = []
    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), desc="Processing directories"):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    vectorstore.add_documents(texts)
    vectorstore.save_local('vectorstore_nodejs-task-app-restapi')

retriever = vectorstore.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10



model = ChatOpenAI(model_name='gpt-3.5-turbo') # 'ada' 'gpt-3.5-turbo' 'gpt-4',
qa_chain = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'), prompt=todo_prompt)

tools = [
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
    #     Tool(
    #     name="Code Queries",
    #     func=qa_chain.run,
    #     description="useful for when you need to ask questions about the code. Input: A question about the code. Output: An answer to that question. Please be very clear what the question is!",
    # ),
    WriteFileTool(),
    ReadFileTool(),
]

prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

OBJECTIVE = "Create a Swagger JSON File for the API in nodejs-task-app-restapi"

# Logging of LLMChains
verbose = True
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})