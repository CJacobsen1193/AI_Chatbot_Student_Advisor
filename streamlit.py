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
   llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=2000,
    model_kwargs={"token":"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"},
    tokenizer_kwargs={"token": "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"},
    device_map="auto",
  )

   Settings.llm = llm
   Settings.embed_model='local:hkunlp/instructor-large'
 
   qdrant_client = QdrantClient(
      url="https://1d752ae2-4e0f-4101-ae0f-b59cd212e480.us-east4-0.gcp.cloud.qdrant.io",
      api_key="ZEUHVnqv9sKXF1gHpY3u1pBKljE26BBoOqA2bkyAXKT7nEhCdq_xWA",
   )

   vector_store = QdrantVectorStore(client=qdrant_client, collection_name="mycollection", enable_hybrid=True)
   index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

   chat_engine = index.as_chat_engine(chat_mode="context",response_mode="compact",max_new_tokens=1024,
                                        system_prompt=("You are a chatbot, able to have normal interactions, as well as talk about Franklin University")
                                        )
  
   st.title("Franklin Virtual Assistant")

   client = HuggingFaceLLM(api_key="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")


   if "huggingface_model" not in st.session_state:
    st.session_state["huggingface_model"] = "meta-llama/Llama-2-7b-chat-hf"

   if "messages" not in st.session_state:
    st.session_state.messages = []

   for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

   if prompt := st.chat_input("Ask me a question about Franklin University!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      prompt = str(input("Ask me a question about Franklin University!  "))
      response = st.write_stream(chat_engine.chat(prompt))
      st.session_state.messages.append({"role": "assistant", "content": response})
      

if __name__ == "__main__":
  stMain()
