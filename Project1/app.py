"""Gradio interface for the Atomic Habits retriever comparison.

    python Project1/app.py

Build the databases first with build_database.py. The Gemini API key is read
from Project1/.env.
"""

import gradio as gr

import config
from models import get_llm
from rag import run_rag
from retrievers import load_retrievers

# Loaded once on the first question and reused afterwards, so the embedding
# model and the vector stores are not rebuilt on every click.
LLM = None
RETRIEVERS = None


def get_system():
    """Load the LLM and the retrievers once, then reuse them.

    :return: (llm, retrievers)
    """
    global LLM, RETRIEVERS

    if RETRIEVERS is None:
        LLM = get_llm()
        RETRIEVERS = load_retrievers(LLM)

    return LLM, RETRIEVERS


def format_latency(result):
    """Small Markdown table of the two timings."""
    return (
        "| Step | Time |\n"
        "| --- | --- |\n"
        f"| Retrieval | **{result['retrieval_latency']}s** |\n"
        f"| Generation | **{result['generation_latency']}s** |\n"
        f"| Total | **{result['total_latency']}s** |\n\n"
        f"Retrieved **{result['num_documents']}** document(s), "
        f"**{result['context_words']}** words of context."
    )


def format_sources(result):
    """Markdown list of what the retriever actually returned."""
    if not result["sources"]:
        return "_No sources returned._"

    parts = []
    for i, source in enumerate(result["sources"], 1):
        snippet = " ".join(source["content"].split())[:300]
        parts.append(
            f"**{i}. Page {source['page']}** - {source['chapter']}  \n"
            f"_{source['words']} words, from {source['source']}_\n\n"
            f"> {snippet}..."
        )

    return "\n\n---\n\n".join(parts)


def answer_question(retriever_name, question):
    """Run one retriever, for the single-retriever tab.

    :return: (answer, latency table, sources)
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        llm, retrievers = get_system()
        result = run_rag(retrievers, llm, retriever_name, question.strip())
    except Exception as error:
        return f"**Error:** {error}", "", ""

    return result["answer"], format_latency(result), format_sources(result)


def compare_all(question):
    """Run the same question through all four retrievers.

    Each panel carries its own heading, so the columns stay identifiable.

    :return: one Markdown block per retriever, in config.RETRIEVER_NAMES order
    """
    blanks = [""] * (len(config.RETRIEVER_NAMES) - 1)

    if not question.strip():
        return ["Please enter a question."] + blanks

    try:
        llm, retrievers = get_system()
    except Exception as error:
        return [f"**Error:** {error}"] + blanks

    panels = []
    for name in config.RETRIEVER_NAMES:
        try:
            result = run_rag(retrievers, llm, name, question.strip())
            panels.append(
                f"## {name}\n\n"
                f"{result['retrieval_latency']}s retrieval + "
                f"{result['generation_latency']}s generation = "
                f"**{result['total_latency']}s**  \n"
                f"{result['num_documents']} document(s), "
                f"{result['context_words']} words of context\n\n"
                f"{result['answer']}\n\n"
                f"### Sources\n\n{format_sources(result)}"
            )
        except Exception as error:
            panels.append(f"## {name}\n\n**Error:** {error}")

    return panels


INTRO = f"""
# Atomic Habits - RAG Retriever Comparison

Four LangChain retrievers over the same book, the same index, the same prompt and
the same model. The only thing that changes is **how the context is retrieved**.

| Component | Choice |
| --- | --- |
| Book | *Atomic Habits*, body pages {config.CONTENT_PAGE_RANGE[0]}-{config.CONTENT_PAGE_RANGE[1]} |
| Embeddings | BGE-M3 (local) |
| Vector store | ChromaDB |
| LLM | {config.LLM_MODEL} |
| Child chunks | ~{config.CHILD_CHUNK_WORDS} words, {config.CHILD_OVERLAP_WORDS} word overlap |
| Parent blocks | ~{config.PARENT_CHUNK_WORDS} words |
"""

ABOUT = f"""
## The four retrievers

**Simple Similarity** - embeds the question and returns the {config.TOP_K} nearest
chunks. The baseline every other method is measured against.

**MMR (Maximal Marginal Relevance)** - fetches {config.MMR_FETCH_K} candidates, then
picks {config.TOP_K} that are relevant to the question *and* different from each
other. Useful when plain similarity would return three chunks that all say the
same thing.

**Parent Document** - searches the small chunks for precision, but returns the
larger block each chunk belongs to. The model sees a whole section instead of a
fragment, at the cost of a much longer context. Note that several chunks often
share one parent, so this retriever can return fewer documents than the others.

**Multi-Query** - asks the LLM for {config.MULTI_QUERY_COUNT} extra phrasings of the
question, searches with all of them, and returns the combined results. Widens
coverage when your wording does not match the book's, but costs an extra LLM
call and returns more documents than the others.

## How the book is prepared

Pages are read as paragraph blocks, grouped into ~{config.PARENT_CHUNK_WORDS} word
parent blocks that never cross a chapter boundary, then split into
~{config.CHILD_CHUNK_WORDS} word child chunks. Chapter titles come from the PDF's own
table of contents. The Notes and Index are skipped: they are 18% of the book and
would compete with real content for the {config.TOP_K} retrieval slots.

Both collections index the *same* child chunks, so the only difference between
the retrievers is what they hand back.

## Metadata on each chunk

```json
{{
    "page": 137,
    "chapter": "Chapter 13: How to Stop Procrastinating by Using the Two-Minute Rule",
    "source": "AtomicHabits.pdf"
}}
```
"""


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title="Atomic Habits RAG Comparison") as demo:
        gr.Markdown(INTRO)

        with gr.Tab("Single retriever"):
            with gr.Row():
                retriever_choice = gr.Dropdown(
                    label="Retriever",
                    choices=config.RETRIEVER_NAMES,
                    value=config.SIMILARITY,
                    scale=1,
                )
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What is the Two-Minute Rule?",
                    scale=3,
                )

            gr.Examples(
                examples=config.EVAL_QUESTIONS,
                inputs=question_box,
                label="Example questions",
            )
            run_button = gr.Button("Run", variant="primary")

            with gr.Row():
                answer_panel = gr.Markdown()
                latency_panel = gr.Markdown()

            gr.Markdown("### Retrieved sources")
            sources_panel = gr.Markdown()

            run_button.click(
                fn=answer_question,
                inputs=[retriever_choice, question_box],
                outputs=[answer_panel, latency_panel, sources_panel],
            )

        with gr.Tab("Compare all four"):
            compare_box = gr.Textbox(
                label="Question to compare",
                placeholder="e.g. What are the four laws of behaviour change?",
            )
            gr.Examples(
                examples=config.EVAL_QUESTIONS,
                inputs=compare_box,
                label="Example questions",
            )
            compare_button = gr.Button("Compare all four", variant="primary")

            # Two panels per row, each labelled by its own heading.
            comparison_panels = []
            for _ in range(0, len(config.RETRIEVER_NAMES), 2):
                with gr.Row():
                    comparison_panels.append(gr.Markdown())
                    comparison_panels.append(gr.Markdown())

            compare_button.click(
                fn=compare_all,
                inputs=[compare_box],
                outputs=comparison_panels,
            )

        with gr.Tab("About"):
            gr.Markdown(ABOUT)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
