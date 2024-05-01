import streamlit as st
import torch
import os
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client import QdrantClient
from llama_index.core import Settings
from InstructorEmbedding import INSTRUCTOR


def stMain():

  open_api_key = ""
  os.environ['OPENAI_API_KEY'] = open_api_key 

  Settings.embed_model = 'local:hkunlp/instructor-large'

  qdrant_client = QdrantClient(
      url="https://097d6aff-312a-41fe-90e7-90219bf4a194.us-east4-0.gcp.cloud.qdrant.io",
      api_key="zeQHBdKQ5eZcgopVPI7uNVmisVDMJ4waGlfHjeAEU801klh-b35cIw",
  )

  vector_store = QdrantVectorStore(client=qdrant_client, collection_name="mycollection", enable_hybrid=True)
  index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

  chat_engine = index.as_chat_engine(chat_mode="context",response_mode="compact",max_new_tokens=1024,
                                        system_prompt=("You are a chatbot, able to have normal interactions, as well as talk about Franklin University")
                                        )
  
  st.title("Franklin Virtual Assistant")
  new_session = False
  end_chat = False
  new_session = st.sidebar.checkbox("Reset Chat History")
  end_chat = st.sidebar.checkbox("Exit Chat")

  count = 0

  while not end_chat:
    count+=1

    #initialize chat history
    if "messages" not in st.session_state:
      st.session_state.messages=[]

    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

    #request prompt from user
    prompt = None
    st_message = st.chat_message("user")
    prompt = st_message.chat_input("Ask me a question about Franklin University")
    more = True

    #wait for prompt from user
    with st.sidebar.status("Thinking..."):
      while more:
        if (prompt is None or len(prompt) <= 0):
          more = True
        else:
          more = False
    
    #write prompt on chat window
    if prompt is not None or len(prompt) > 0:
      prompt = str(prompt)
      st.session_state.messages.append(
        {"role": "user", "content": prompt}
      )
      with st.chat_message("user"):
        st.markdown(prompt)

    #display response in chat message container
    response = None
    with st.sidebar.status("Finding answers..."):
      response = chat_engine.chat(prompt)
    if response is not None or len(response) > 0:
      if prompt and response:
        st_message = st.chat_message("assistant")
        st_message.write(response.response)
      st.session_state.messages.append(
        {"role": "assistant", 
        "content": response}
      )

    if new_session:
      chat_engine.reset()
      new_session = False
    
    st.stop()
    st.rerun()
    st.title("Franklin Virtual Assistant")
    new_session=False
    end_chat=False
    new_session=st.sidebar.checkbox("Reset Chat History")
    end_chat=st.sidebar.checkbox("Exit Chat")
  st.stop()


if __name__ == "__main__":
  stMain()
