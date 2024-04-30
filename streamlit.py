import streamlit as st
from dotenv import load_dotenv
import torch
import os
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

%pip install -i https://pypi.org/simple/ bitsandbytes

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

def main():
  load_model()
  


