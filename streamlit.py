import streamlit as st
from dotenv import load_dotenv
import torch
import os
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM


def load_model():
  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
  )
  llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": "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", "quantization_config": quantization_config},
    tokenizer_kwargs={"token": "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"},
    device_map="auto",
  )

def load_db():
  from llama_index.core import VectorStoreIndex
  from llama_index.vector_stores.qdrant import QdrantVectorStore
  import qdrant_client
  from qdrant_client import QdrantClient
  from llama_index.core import Settings
  from InstructorEmbedding import INSTRUCTOR

  Settings.llm = llm
  Settings.embed_model='local:hkunlp/instructor-large'


  qdrant_client = QdrantClient(
    url="https://1d752ae2-4e0f-4101-ae0f-b59cd212e480.us-east4-0.gcp.cloud.qdrant.io",
    api_key="ZEUHVnqv9sKXF1gHpY3u1pBKljE26BBoOqA2bkyAXKT7nEhCdq_xWA",
  )

  vector_store = QdrantVectorStore(client=qdrant_client, collection_name="mycollection", enable_hybrid=True)
  index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

def main():
  load_model()
  load_db()
  


