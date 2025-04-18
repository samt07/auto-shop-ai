#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr
import json


# In[ ]:


# imports for langchain, plotly and Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np 
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# In[ ]:


# price is a factor for our company, so we're going to use a low cost model

MODEL = "gpt-4o-mini"
db_name = "vector_db"


# In[ ]:


# Load environment variables in a file called .env

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')


# In[ ]:


##Load json
with open("knowledge-base/pinkys.json", 'r') as f:
    data = json.load(f)


# In[ ]:


#Convert to Langchain
documents = []
for item in data:
    content = item["content"]
    metadata = item.get("metadata", {})
    # Embed duration into content so it can be used as context
    duration = metadata.get("duration")
    if duration:
        content += f"\nDuration: {duration}"
    documents.append(Document(page_content=content, metadata=metadata))


# In[ ]:


#Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ",", " ", ""])
chunks = splitter.split_documents(documents)


# In[ ]:


doc_types = set(chunk.metadata['source'] for chunk in chunks)
#print(f"Document types found: {', '.join(doc_types)}")


# In[ ]:


embeddings = OpenAIEmbeddings()

# Delete if already exists

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create vectorstore

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
#print(f"Vectorstore created with {vectorstore._collection.count()} documents")


# In[ ]:


# # Let's investigate the vectors. Use for debugging if needed

# collection = vectorstore._collection
# count = collection.count()

# sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
# dimensions = len(sample_embedding)
# print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


# In[ ]:


# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)


memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# ## Now we will bring this up in Gradio using the Chat interface -

# In[ ]:


# Wrapping that in a function

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]


# In[ ]:


# And in Gradio:

view = gr.ChatInterface(chat, type="messages").launch(server_name="0.0.0.0", server_port=7860)
#demo.launch(server_name="0.0.0.0", server_port=7860)

