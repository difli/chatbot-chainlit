import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Cassandra
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document

import chainlit as cl
from chainlit.types import AskFileResponse

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

SECURE_CONNECT_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), os.environ.get('SECURE_CONNECT_BUNDLE_PATH'))
ASTRA_DB_TOKEN_BASED_USERNAME=os.environ.get('ASTRA_DB_TOKEN_BASED_USERNAME')
ASTRA_DB_TOKEN_BASED_PASSWORD=os.environ.get('ASTRA_DB_TOKEN_BASED_PASSWORD')
ASTRA_DB_KEYSPACE=os.environ.get('ASTRA_DB_KEYSPACE')
ASTRA_DB_TABLE_NAME=os.environ.get('ASTRA_DB_TABLE_NAME')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

cloud_config = {
   'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_TOKEN_BASED_USERNAME, ASTRA_DB_TOKEN_BASED_PASSWORD)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

systemTemplate = """
Act as a friendly assistant that answers questions on travel and security information for travels to foreign countries. 
Your highest priority is to answer only in english language.
Only answer questions within the <tag> section 
You should only answer questions about travel and security. 
If there is another question answer with "Can I help you with something else about travel and security?". 

context:
{context}

chat history:
{chat_history}

Your highest priority is to answer only in english language.
"""
systemMessagePrompt = SystemMessagePromptTemplate.from_template(systemTemplate)
humanTemplate = "<tag>{question}<tag>"
humanMessagePrompt = HumanMessagePromptTemplate.from_template(humanTemplate)

chatPrompt = ChatPromptTemplate.from_messages(
    [systemMessagePrompt, humanMessagePrompt]
)
chain_type_kwargs = {"prompt": chatPrompt}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to this Travelbot powered by Astra DB! 
To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


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
        embedding=embeddings,
        session=session,
        keyspace=ASTRA_DB_KEYSPACE,
        table_name=ASTRA_DB_TABLE_NAME)

    return docsearch


@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        path="travel.png",
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

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True, combine_docs_chain_kwargs={"prompt": chatPrompt}
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    # Call the chain asynchronously
 #   res = await chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()