"""Central configuration for the Atomic Habits RAG retriever comparison.

Every tunable value lives here so the other files stay short and readable.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved from this file's location, so every script works no matter which
# directory you run it from.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_DIR)

PDF_PATH = os.path.join(PROJECT_DIR, "AtomicHabits.pdf")
ENV_PATH = os.path.join(PROJECT_DIR, ".env")
CHROMA_DIR = os.path.join(PROJECT_DIR, "chroma_db")

STANDARD_DB_DIR = os.path.join(CHROMA_DIR, "standard")
PARENT_DB_DIR = os.path.join(CHROMA_DIR, "parent")
PARENT_STORE_PATH = os.path.join(CHROMA_DIR, "parent_store.pkl")

STANDARD_COLLECTION = "standard_collection"
PARENT_COLLECTION = "parent_child_collection"

RESULTS_JSON_PATH = os.path.join(PROJECT_DIR, "evaluation_results.json")
REPORT_MD_PATH = os.path.join(PROJECT_DIR, "evaluation_report.md")

# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------
# The PDF has 285 pages. Pages 1-7 are front matter (title, copyright, contents)
# and pages 213-285 are the Notes and Index sections - 14,071 words of raw
# citations, 18% of the book. Indexing them would let reference lists compete
# with real content for the 3 retrieval slots, so only the body is indexed.
CONTENT_PAGE_RANGE = (8, 212)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Sizes are measured in WORDS (see preprocess.count_words), not characters.
CHILD_CHUNK_WORDS = 200
CHILD_OVERLAP_WORDS = 20
PARENT_CHUNK_WORDS = 1000

# A block or chunk smaller than this is not a useful thing to retrieve, so it
# gets folded into a neighbour (blocks) or dropped (chunks). This is what keeps
# the six single-page section dividers - "THE 1ST LAW / Make It Obvious" and
# friends - from becoming six-word entries in the index.
MIN_BLOCK_WORDS = 300
MIN_CHUNK_WORDS = 50

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_DIR = os.path.join(REPO_ROOT, "models", "bge-m3")
LLM_MODEL = "gemini-3.6-flash"
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 3            # documents returned by every retriever
MMR_FETCH_K = 20     # candidates MMR considers before picking TOP_K
MULTI_QUERY_COUNT = 2  # extra query variations the LLM writes

SIMILARITY = "Simple Similarity"
MMR = "MMR"
PARENT = "Parent Document"
MULTI_QUERY = "Multi-Query"

RETRIEVER_NAMES = [SIMILARITY, MMR, PARENT, MULTI_QUERY]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """You are an expert assistant answering questions about the book 'Atomic Habits' by James Clear.

Answer the question using ONLY the context below. Your answer must be:
1. Accurate and directly grounded in the provided context.
2. Clearly organised, using bullet points or short sections where it helps.
3. Concise and friendly.

If the answer is not in the context, reply exactly:
"I could not find the answer to this question in the retrieved passages of 'Atomic Habits'."

Context:
{context}

Question: {question}

Answer:"""

MULTI_QUERY_PROMPT = """You are helping search a vector database of the book 'Atomic Habits'.

Write exactly {count} alternative phrasings of the question below. Vary the wording
and the vocabulary so that together they cover the topic from different angles.

Original question: {question}

Output ONLY the {count} questions, one per line, with no numbering, bullets, or extra text."""

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    "What are the four steps of the habit loop, and in what order do they occur?",
    "Explain the 'Two-Minute Rule' for building new habits and give an example of how to apply it.",
    "How does James Clear define the 'Goldilocks Rule' for maintaining motivation?",
    "What is the difference between motion and action, according to the author?",
    "Why does the author argue that your identity is closely linked to your habits, "
    "and how can identity-based habits be established?",
]
