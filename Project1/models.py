"""Model construction: the local embedding model and the Gemini chat model.

Kept in one small file because both build_database.py and retrievers.py need
the embedding model, and both app.py and evaluate.py need the LLM.
"""

import os

import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

import config


def get_device():
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embeddings():
    """Load the local BGE-M3 embedding model.

    BGE-M3 already returns L2-normalised vectors, so Chroma's default L2
    distance ranks results identically to cosine similarity.

    :return: HuggingFaceEmbeddings instance
    """
    device = get_device()
    print(f"Loading BGE-M3 embedding model on '{device}' from: {config.EMBEDDING_MODEL_DIR}")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_DIR,
        model_kwargs={"device": device},
    )


def get_api_key(api_key=None):
    """Find the Gemini API key.

    Looks at the supplied argument first, then Project1/.env, then the
    environment.

    :param api_key: key typed into the UI, or None
    :return: the API key as a string
    """
    if api_key and api_key.strip():
        return api_key.strip()

    load_dotenv(config.ENV_PATH)
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise ValueError(
            "No Gemini API key found. Add GEMINI_API_KEY=... to Project1/.env "
            "or paste a key into the app."
        )
    return key.strip()


def get_llm(api_key=None):
    """Create the Gemini chat model used for generation and query expansion.

    :param api_key: optional key; falls back to .env / environment
    :return: ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=get_api_key(api_key),
        temperature=config.TEMPERATURE,
    )
