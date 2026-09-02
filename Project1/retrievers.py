"""The four retrievers being compared, loaded from the prebuilt databases.

Nothing in this file builds an index - run build_database.py first. Every
retriever returns config.TOP_K documents, so the only difference between them
is *how* they choose those documents and *what* they hand back.
"""

import os
import time

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

import config
import preprocess
from build_database import load_parent_store
from models import get_embeddings


def check_database_exists():
    """Raise a clear error if build_database.py has not been run yet."""
    required = [config.STANDARD_DB_DIR, config.PARENT_DB_DIR, config.PARENT_STORE_PATH]

    if any(not os.path.exists(path) for path in required):
        raise FileNotFoundError(
            "The vector database has not been built yet.\n"
            "Run this first:  python Project1/build_database.py"
        )


def get_similarity_retriever(vectorstore):
    """Plain similarity search over the child chunks.

    BGE-M3 returns normalised vectors, so Chroma's L2 distance ranks the same
    way cosine similarity would.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K},
    )


def get_mmr_retriever(vectorstore):
    """Maximal Marginal Relevance.

    Fetches MMR_FETCH_K candidates, then greedily keeps the TOP_K that balance
    relevance to the query against being different from each other, which
    avoids returning three near-identical chunks.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K, "fetch_k": config.MMR_FETCH_K},
    )


def get_parent_retriever(vectorstore, docstore):
    """Search the small child chunks, return the large parent blocks.

    The chunks give precise matching; the blocks give the LLM enough
    surrounding context that an idea is not cut off mid-thought.
    """
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=preprocess.get_child_splitter(),
        parent_splitter=None,
        search_kwargs={"k": config.TOP_K},
    )


def get_multi_query_retriever(vectorstore, llm):
    """Ask the LLM to rephrase the question, search with every version.

    The results of all the searches are combined and de-duplicated, which
    widens coverage when the wording of the question does not match the
    wording of the book.
    """
    prompt = PromptTemplate.from_template(config.MULTI_QUERY_PROMPT).partial(
        count=config.MULTI_QUERY_COUNT
    )

    return MultiQueryRetriever.from_llm(
        retriever=get_similarity_retriever(vectorstore),
        llm=llm,
        prompt=prompt,
    )


def load_retrievers(llm):
    """Open the prebuilt databases and construct all four retrievers.

    :param llm: chat model, used by Multi-Query to rephrase the question
    :return: dict of {retriever name: retriever}
    """
    check_database_exists()
    embeddings = get_embeddings()

    standard_db = Chroma(
        collection_name=config.STANDARD_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.STANDARD_DB_DIR,
    )
    parent_db = Chroma(
        collection_name=config.PARENT_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.PARENT_DB_DIR,
    )
    parent_docs = load_parent_store()

    return {
        config.SIMILARITY: get_similarity_retriever(standard_db),
        config.MMR: get_mmr_retriever(standard_db),
        config.PARENT: get_parent_retriever(parent_db, parent_docs),
        config.MULTI_QUERY: get_multi_query_retriever(standard_db, llm),
    }


def retrieve(retrievers, name, query):
    """Run one retriever and time it.

    :param retrievers: dict from load_retrievers()
    :param name: one of config.RETRIEVER_NAMES
    :param query: the user's question
    :return: (documents, seconds taken)
    """
    if name not in retrievers:
        raise ValueError(f"Unknown retriever: {name}. Expected one of {config.RETRIEVER_NAMES}")

    start_time = time.time()
    documents = retrievers[name].invoke(query)
    return documents, time.time() - start_time
