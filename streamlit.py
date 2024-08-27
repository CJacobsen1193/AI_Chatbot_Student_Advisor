import streamlit as st
import torch
import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client import QdrantClient
from llama_index.core import Settings
from InstructorEmbedding import INSTRUCTOR


def stMain():

  open_api_key = '****'
  os.environ['OPENAI_API_KEY'] = open_api_key 

  Settings.embed_model = 'local:hkunlp/instructor-large'

  qdrant_client = QdrantClient(
      url="https://097d6aff-312a-41fe-90e7-90219bf4a194.us-east4-0.gcp.cloud.qdrant.io",
      api_key="zeQHBdKQ5eZcgopVPI7uNVmisVDMJ4waGlfHjeAEU801klh-b35cIw",
  )

  vector_store = QdrantVectorStore(client=qdrant_client, collection_name="mycollection", enable_hybrid=True)
  index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

  chat_engine = index.as_chat_engine(chat_mode="context",response_mode="compact",max_new_tokens=1024,
                                        system_prompt=("You are a chatbot, able to have normal interactions, as well as talk about Franklin University using only the context provided")
                                        )
  
  st.title("Team 3 Virtual Advisor: AI from US")

    #initialize chat history
  if "messages" not in st.session_state:
      st.session_state.messages=[]

  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

    #request prompt from user
  if prompt := st.chat_input("Ask me about Franklin University!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)
    with st.chat_message("assistant"):
      response = chat_engine.chat(str(prompt))
      st.markdown(response.response)
    st.session_state.messages.append({"role": "assistant", "content": response.response}) 



if __name__ == "__main__":
  stMain()
