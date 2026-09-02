"""Generation: turn the retrieved documents into an answer.

Retrieval and generation are timed separately so the comparison can show which
part of the pipeline each retriever actually costs. Multi-Query, for example,
spends an extra LLM call inside retrieval that the other three do not.
"""

import time

from langchain_core.prompts import PromptTemplate

import config
from retrievers import retrieve


def format_context(documents):
    """Join the retrieved documents into the context block for the prompt.

    :param documents: retrieved Documents
    :return: one string
    """
    return "\n\n".join(doc.page_content for doc in documents)


def format_sources(documents):
    """Pull the display fields out of the retrieved documents.

    Child chunks carry an exact "page". Parent blocks span several pages, so
    they carry page_start/page_end instead and are shown as a range.

    :param documents: retrieved Documents
    :return: list of dicts ready to display
    """
    sources = []

    for doc in documents:
        metadata = doc.metadata
        page = metadata.get("page")
        if page is None:
            page = f"{metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}"

        sources.append({
            "content": doc.page_content,
            "page": page,
            "chapter": metadata.get("chapter", "Unknown"),
            "source": metadata.get("source", "Unknown"),
            "words": len(doc.page_content.split()),
        })

    return sources


def generate_answer(llm, context, question):
    """Ask the LLM to answer the question using only the retrieved context.

    :param llm: the chat model
    :param context: text from format_context()
    :param question: the user's question
    :return: (answer, seconds taken)
    """
    prompt = PromptTemplate.from_template(config.ANSWER_PROMPT)
    message = prompt.format(context=context, question=question)

    start_time = time.time()
    response = llm.invoke(message)

    # .text flattens the reply to a plain string. Gemini 3 models return
    # .content as a list of content blocks rather than a bare string.
    return response.text, time.time() - start_time


def run_rag(retrievers, llm, name, question):
    """Run the full pipeline for one retriever: retrieve, then generate.

    :param retrievers: dict from retrievers.load_retrievers()
    :param llm: the chat model
    :param name: one of config.RETRIEVER_NAMES
    :param question: the user's question
    :return: dict with the answer, both latencies and the retrieved sources
    """
    documents, retrieval_seconds = retrieve(retrievers, name, question)
    context = format_context(documents)
    answer, generation_seconds = generate_answer(llm, context, question)

    return {
        "retriever": name,
        "question": question,
        "answer": answer,
        "retrieval_latency": round(retrieval_seconds, 3),
        "generation_latency": round(generation_seconds, 3),
        "total_latency": round(retrieval_seconds + generation_seconds, 3),
        "num_documents": len(documents),
        "context_words": len(context.split()),
        "sources": format_sources(documents),
    }
