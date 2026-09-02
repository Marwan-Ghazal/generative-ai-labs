"""Compare what each retriever fetches, without generating any answers.

    python Project1/evaluate.py

Retrieval is the only thing that differs between the four setups, so this
script stops there: it records what each retriever returned, how long it took
and how much context it would hand to the model.

Only Multi-Query touches the API, because it asks the LLM to rephrase the
question before searching. That is one call per question rather than one call
per question per retriever.

Writes two files next to this one:

    evaluation_results.json   the raw results
    evaluation_report.md      a readable side-by-side comparison

Build the databases first with build_database.py.
"""

import json
import time

import config
from models import get_llm
from rag import format_context, format_sources
from retrievers import load_retrievers, retrieve


def failed_result(name, question, error):
    """Placeholder row so one failure does not lose the rest of the run."""
    return {
        "retriever": name,
        "question": question,
        "error": str(error),
        "retrieval_latency": 0.0,
        "num_documents": 0,
        "context_words": 0,
        "sources": [],
    }


def evaluate_one(retrievers, name, question):
    """Retrieve for one question and record what came back.

    :param retrievers: dict from retrievers.load_retrievers()
    :param name: one of config.RETRIEVER_NAMES
    :param question: the question to ask
    :return: a result dict
    """
    documents, seconds = retrieve(retrievers, name, question)
    context = format_context(documents)

    return {
        "retriever": name,
        "question": question,
        "error": None,
        "retrieval_latency": round(seconds, 3),
        "num_documents": len(documents),
        "context_words": len(context.split()),
        "sources": format_sources(documents),
    }


def run_evaluation(retrievers):
    """Run every retriever against every question in config.EVAL_QUESTIONS.

    :return: dict of {retriever name: list of result dicts}
    """
    results = {}
    total = len(config.EVAL_QUESTIONS)

    for name in config.RETRIEVER_NAMES:
        print(f"\nEvaluating: {name}")
        results[name] = []

        for i, question in enumerate(config.EVAL_QUESTIONS, 1):
            print(f"  Q{i}/{total}: {question[:60]}...")
            try:
                results[name].append(evaluate_one(retrievers, name, question))
            except Exception as error:
                print(f"    failed: {error}")
                results[name].append(failed_result(name, question, error))

    return results


def average(values):
    """Mean of a list, or 0.0 when it is empty."""
    return sum(values) / len(values) if values else 0.0


def page_list(run):
    """Comma separated pages for one result, e.g. 'p.137, p.131-134'."""
    return ", ".join(f"p.{source['page']}" for source in run["sources"]) or "none"


def chapters_touched(runs):
    """Every distinct chapter a retriever reached across all the questions."""
    return {source["chapter"] for run in runs for source in run["sources"]}


def write_summary(file, results):
    """Write the averaged comparison table."""
    file.write("## Summary\n\n")
    file.write(
        "| Retriever | Avg retrieval | Avg docs | Avg context words | Distinct chapters reached |\n"
    )
    file.write("| --- | --- | --- | --- | --- |\n")

    for name in config.RETRIEVER_NAMES:
        runs = results.get(name, [])
        file.write(
            f"| **{name}** "
            f"| {average([r['retrieval_latency'] for r in runs]):.3f}s "
            f"| {average([r['num_documents'] for r in runs]):.1f} "
            f"| {average([r['context_words'] for r in runs]):.0f} "
            f"| {len(chapters_touched(runs))} |\n"
        )

    file.write(
        "\n> Multi-Query spends an extra LLM call inside retrieval to rephrase the\n"
        "> question, so its time is not comparable with the other three. It also\n"
        "> returns the union of several searches, so it fetches more documents.\n"
        ">\n"
        "> Parent Document often returns fewer documents than the others: several\n"
        "> child chunks frequently belong to the same parent block, and the\n"
        "> duplicates collapse into one.\n\n"
    )


def write_comparison(file, results):
    """Write one table per question, comparing what each retriever fetched."""
    file.write("## What each retriever fetched\n\n")

    for i, question in enumerate(config.EVAL_QUESTIONS):
        file.write(f"### Q{i + 1}. {question}\n\n")
        file.write("| Retriever | Time | Docs | Context words | Pages |\n")
        file.write("| --- | --- | --- | --- | --- |\n")

        for name in config.RETRIEVER_NAMES:
            run = results[name][i]
            file.write(
                f"| {name} "
                f"| {run['retrieval_latency']}s "
                f"| {run['num_documents']} "
                f"| {run['context_words']} "
                f"| {page_list(run)} |\n"
            )

        file.write("\n")

        for name in config.RETRIEVER_NAMES:
            run = results[name][i]

            if run["error"]:
                file.write(f"**{name}** - failed: {run['error']}\n\n")
                continue

            file.write(f"**{name}**\n\n")
            for n, source in enumerate(run["sources"], 1):
                snippet = " ".join(source["content"].split())[:220]
                file.write(
                    f"{n}. **p.{source['page']}** - {source['chapter']} "
                    f"({source['words']} words)  \n"
                    f"   > {snippet}...\n"
                )
            file.write("\n")

        file.write("---\n\n")


def write_report(results, path):
    """Write the Markdown comparison report."""
    with open(path, "w", encoding="utf-8") as file:
        file.write("# RAG Retriever Comparison - Atomic Habits\n\n")
        file.write(
            "Retrieval only - no answers are generated. Four LangChain retrievers\n"
            "over the same book, the same index and the same embedding model\n"
            "(BGE-M3). The only thing that changes is how the context is chosen.\n\n"
        )
        file.write(
            f"Child chunks ~{config.CHILD_CHUNK_WORDS} words "
            f"({config.CHILD_OVERLAP_WORDS} word overlap), "
            f"parent blocks ~{config.PARENT_CHUNK_WORDS} words, "
            f"top {config.TOP_K} per retriever.\n\n"
        )
        write_summary(file, results)
        write_comparison(file, results)

    print(f"Wrote {path}")


def main():
    start_time = time.time()

    # The LLM is only used to build the Multi-Query retriever, which rephrases
    # the question. No answers are generated.
    retrievers = load_retrievers(get_llm())

    results = run_evaluation(retrievers)

    with open(config.RESULTS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nWrote {config.RESULTS_JSON_PATH}")

    write_report(results, config.REPORT_MD_PATH)
    print(f"Finished in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
