from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import DuckDuckGoSearchRun

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import os
import chainlit as cl
from chainlit.types import AskFileResponse

import logging

from langchain.vectorstores.cassandra import Cassandra

# configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#os.environ["OPENAI_API_KEY"] = "{Your_API_Key}"
SECURE_CONNECT_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), os.environ.get('SECURE_CONNECT_BUNDLE_PATH'))
ASTRA_DB_TOKEN_BASED_USERNAME=os.environ.get('ASTRA_DB_TOKEN_BASED_USERNAME')
ASTRA_DB_TOKEN_BASED_PASSWORD=os.environ.get('ASTRA_DB_TOKEN_BASED_PASSWORD')
ASTRA_DB_KEYSPACE=os.environ.get('ASTRA_DB_KEYSPACE')
ASTRA_DB_TABLE_NAME=os.environ.get('ASTRA_DB_TABLE_NAME')

cloud_config = {
   'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_TOKEN_BASED_USERNAME, ASTRA_DB_TOKEN_BASED_PASSWORD)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
myEmbedding = OpenAIEmbeddings()
retrieval_llm = OpenAI(temperature=0.5)

table_name = 'rag_chat_bot'

myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name=ASTRA_DB_TABLE_NAME,
)

index = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

query = "Restaurant personal preferences?"
print(index.query(query, llm=retrieval_llm))

qa = RetrievalQA.from_chain_type(
    llm=retrieval_llm,
    chain_type='stuff',
    retriever=myCassandraVStore.as_retriever(search_type="similarity", search_kwargs={'k': 1}),
    return_source_documents=False,
    verbose=False
)

welcome_message = """Welcome to this restaurant guide that considers your preferences! 
To get started:
1. Upload your preferences as a PDF or text file
2. Find a restaurant
"""

template = """Answer the following questions as best you can speaking as passionate restaurant advisor. 

You have access to the following tools:
{tools}

First understand my preference by using [Search profile] and the information in the input question 

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Your highest priority is to provide correct details, a correct name, a correct address, a correct phone number and a correct website url like this example: 
Provide the restaurant details structured like this:

Restaurant: [Restaurant Name]:
Address: [Street Address, Postal Code City, Country]
Phone Number: [Phone Number with Country Code]
Website: [URL of the restaurant's website]

The details must contain the the following details: name, address, phone number, webpage url
If you do not have all details than don't provide the restaurant information.

If you do not find a restaurant that matches my preference than say that you couldn't find a restaurant.

You should only answer questions about restaurants. 
If there is another question answer with "Can I help you with a restaurant question?". 

Begin! Remember to answer as a passionate and informative restaurant expert when giving your final answer.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print(f"CustomOutputParser: {llm_output}")
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            print(f"NOTMATCH")
            return AgentFinish({"output": llm_output}, llm_output)
#            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_profile(input_text):
  print(f"search_profile: {input_text}")
  return qa({"query": input_text})["result"]

def search_online(input_text):
    search = DuckDuckGoSearchRun().run(f"site:tripadvisor.com restaurants{input_text}")
    return search

def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search


def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tempfile:
        if file.type == "text/plain":
            tempfile.write(file.content)
        elif file.type == "application/pdf":
            with open(tempfile.name, "wb") as f:
                f.write(file.content)

        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    docsearch = Cassandra.from_documents(
        documents=docs,
        embedding=myEmbedding,
        session=session,
        keyspace=ASTRA_DB_KEYSPACE,
        table_name=ASTRA_DB_TABLE_NAME)

    return docsearch

@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        path="bot.png",
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
            disable_human_feedback=True,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Define a query to select all records from a table
    session.execute(f"DROP TABLE IF EXISTS {ASTRA_DB_KEYSPACE}.{ASTRA_DB_TABLE_NAME}")

    docsearch = await cl.make_async(get_docsearch)(file)
    tools = [

        Tool(
            name="Search profile",
            func=search_profile,
            description="useful to search for the user preferences/favourites/likes"
        ),
        Tool(
            name="Search general",
            func=search_general,
            description="useful for when you need to answer restaurant questions"
        )

    ]
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"]
    )

    output_parser = CustomOutputParser()
    llm = ChatOpenAI(temperature=0.2)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"],
                                 allowed_tools=tool_names)
    memory = ConversationBufferWindowMemory(k=2)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    cl.user_session.set("agent", agent_executor)

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    answer = await cl.make_async(agent.run)(message.content, callbacks=[cb])
    await cl.Message(content=answer).send()