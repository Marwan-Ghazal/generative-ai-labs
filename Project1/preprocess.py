"""Turn the Atomic Habits PDF into parent blocks and child chunks.

The pipeline is:

    load_pages()        PDF    -> [(page_number, text), ...]
    build_chapter_map() PDF    -> {page_number: chapter title}
    build_blocks()      pages  -> ~2000 word parent Documents
    split_children()    blocks -> ~500 word child Documents

Why blocks are built before splitting: a LangChain text splitter only splits
*within* a document, it never merges across documents. A page of this book
averages 273 words, so feeding it one Document per page means a 500-word
splitter barely splits anything and a 2000-word splitter does nothing at all.
Grouping pages into blocks first is what makes the parent/child distinction
real.

Run this file directly to print chunk statistics. It loads no models, so it
is fast and safe to re-run while tuning the chunk sizes in config.py.
"""

import os

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def count_words(text):
    """Length function for the splitters, so chunk sizes are counted in words."""
    return len(text.split())


def clean_page_text(page):
    paragraphs = []
    for block in page.get_text("blocks"):
        if block[6] != 0:  # skip image blocks
            continue
        lines = [line.strip() for line in block[4].split("\n") if line.strip()]
        if lines:
            paragraphs.append(" ".join(lines))
    return "\n\n".join(paragraphs).strip()


def load_pages(pdf_path=config.PDF_PATH, page_range=config.CONTENT_PAGE_RANGE):
    doc = fitz.open(pdf_path)
    first_page, last_page = page_range
    pages = []

    for page_number in range(first_page, last_page + 1):
        text = clean_page_text(doc[page_number - 1])
        if text:
            pages.append((page_number, text))

    doc.close()
    return pages


def format_chapter_title(title):
    """Turn a TOC title like '13: How to Stop...' into 'Chapter 13: How to Stop...'."""
    head = title.split(":", 1)[0].strip()
    return f"Chapter {title}" if head.isdigit() else title


def build_chapter_map(pdf_path=config.PDF_PATH):

    doc = fitz.open(pdf_path)
    entries = sorted((page, title) for _level, title, page in doc.get_toc() if page > 0)
    last_page = doc.page_count
    doc.close()

    chapter_map = {}
    for i, (start_page, title) in enumerate(entries):
        end_page = entries[i + 1][0] - 1 if i + 1 < len(entries) else last_page
        for page_number in range(start_page, end_page + 1):
            chapter_map[page_number] = format_chapter_title(title)

    return chapter_map


def make_block(page_group, chapter, source):
    """Join a group of consecutive pages into one parent Document.

    Also records where each page starts inside the joined text, so that
    split_children() can give every child chunk an exact page number.

    :param page_group: list of (page_number, text)
    :param chapter: chapter title these pages belong to
    :param source: file name of the book
    :return: a Document
    """
    texts = []
    page_offsets = []
    cursor = 0

    for page_number, text in page_group:
        page_offsets.append((cursor, page_number))
        texts.append(text)
        cursor += len(text) + 2  # + the "\n\n" separator

    return Document(
        page_content="\n\n".join(texts),
        metadata={
            "chapter": chapter,
            "source": source,
            "page_start": page_group[0][0],
            "page_end": page_group[-1][0],
            "page_offsets": page_offsets,  # internal, stripped before indexing
        },
    )


def group_words(page_group):
    """Total word count of a list of (page_number, text)."""
    return sum(count_words(text) for _, text in page_group)


def group_pages(pages, chapter_map, target_words):
    """Split the pages into chapter-pure runs of roughly target_words.

    :param pages: list of (page_number, text) from load_pages()
    :param chapter_map: lookup from build_chapter_map()
    :param target_words: approximate size of a parent block, in words
    :return: list of [chapter, page_group]
    """
    groups = []
    group = []
    group_chapter = None

    for page_number, text in pages:
        chapter = chapter_map.get(page_number, "Unknown")

        # Close the current group when the chapter changes or it is full enough.
        if group and (chapter != group_chapter or group_words(group) >= target_words):
            groups.append([group_chapter, group])
            group = []

        group.append((page_number, text))
        group_chapter = chapter

    if group:
        groups.append([group_chapter, group])

    return groups


def merge_small_groups(groups, min_words=config.MIN_BLOCK_WORDS):
    """Fold undersized groups into a neighbour.

    Two things produce them. The first is the short tail left over at the end
    of a chapter once a full-size block has been closed - page 30 of chapter 1
    is 33 words on its own. The second is the single-page section dividers such
    as "THE 1ST LAW / Make It Obvious", which the table of contents lists as
    chapters in their own right but which are only six words long.

    A tail is merged backwards into the rest of its own chapter. A divider has
    no earlier block of the same chapter, so it is merged forwards instead and
    becomes the opening line of the chapter it introduces.

    :param groups: list of [chapter, page_group] from group_pages()
    :param min_words: smallest acceptable block size
    :return: list of [chapter, page_group]
    """
    # Pass 1: short tail -> previous group of the same chapter.
    merged = []
    for chapter, page_group in groups:
        if merged and merged[-1][0] == chapter and group_words(page_group) < min_words:
            merged[-1][1].extend(page_group)
        else:
            merged.append([chapter, list(page_group)])

    # Pass 2: whatever is still too small -> the group that follows it.
    result = []
    carried = []
    for i, (chapter, page_group) in enumerate(merged):
        is_last = i == len(merged) - 1
        if not is_last and group_words(page_group) < min_words:
            carried.extend(page_group)
            continue
        result.append([chapter, carried + page_group])
        carried = []

    if carried and result:
        result[-1][1].extend(carried)

    return result


def build_blocks(pages, chapter_map, source, target_words=config.PARENT_CHUNK_WORDS):
    """Group consecutive pages into parent blocks of roughly target_words.

    A block never crosses a chapter boundary, so every block has exactly one
    chapter title.

    :param pages: list of (page_number, text) from load_pages()
    :param chapter_map: lookup from build_chapter_map()
    :param source: file name of the book
    :param target_words: approximate size of a parent block, in words
    :return: list of parent Documents
    """
    groups = group_pages(pages, chapter_map, target_words)
    groups = merge_small_groups(groups)
    return [make_block(page_group, chapter, source) for chapter, page_group in groups]


def page_for_offset(page_offsets, offset):
    """Find which page a character offset inside a block falls on.

    :param page_offsets: list of (char_offset, page_number) from make_block()
    :param offset: character offset of a chunk within the block
    :return: the page number
    """
    page = page_offsets[0][1]
    for char_offset, page_number in page_offsets:
        if char_offset > offset:
            break
        page = page_number
    return page


def get_child_splitter():
    """The splitter used for child chunks, by both collections.

    Shared so that the chunks searched by the Parent Document Retriever are
    identical to the ones searched by the other three retrievers.
    """
    return RecursiveCharacterTextSplitter(
        length_function=count_words,
        chunk_size=config.CHILD_CHUNK_WORDS,
        chunk_overlap=config.CHILD_OVERLAP_WORDS,
        add_start_index=True,
    )


def split_children(blocks):
    """Split parent blocks into child chunks carrying an exact page number.

    :param blocks: parent Documents from build_blocks()
    :return: list of child Documents with {page, chapter, source} metadata
    """
    splitter = get_child_splitter()
    children = []

    for block in blocks:
        page_offsets = block.metadata["page_offsets"]
        for chunk in splitter.split_documents([block]):
            if count_words(chunk.page_content) < config.MIN_CHUNK_WORDS:
                continue  # a stray remainder left at the end of a block
            start_index = chunk.metadata.get("start_index", 0)
            chunk.metadata = {
                "page": page_for_offset(page_offsets, start_index),
                "chapter": block.metadata["chapter"],
                "source": block.metadata["source"],
            }
            children.append(chunk)

    return children


def strip_page_offsets(blocks):
    """Return copies of the blocks without the internal page_offsets list.

    Chroma metadata values must be simple scalars, so the offsets table has to
    go before the blocks reach the vector store. Only split_children() needs it.

    :param blocks: parent Documents from build_blocks()
    :return: list of parent Documents safe to index
    """
    return [
        Document(
            page_content=block.page_content,
            metadata={k: v for k, v in block.metadata.items() if k != "page_offsets"},
        )
        for block in blocks
    ]


def prepare_documents(pdf_path=config.PDF_PATH):
    """Run the whole preprocessing pipeline.

    :param pdf_path: path to the book
    :return: (parent_blocks, child_chunks), both ready to index
    """
    source = os.path.basename(pdf_path)
    pages = load_pages(pdf_path)
    chapter_map = build_chapter_map(pdf_path)

    blocks = build_blocks(pages, chapter_map, source)
    children = split_children(blocks)

    return strip_page_offsets(blocks), children


def describe_chunks(documents, name):
    """Print size statistics for a list of chunks, as in the Course 1 lab."""
    word_counts = [count_words(doc.page_content) for doc in documents]
    total = len(documents)

    print(f"\n=== {name} ===")
    print(f"Count           : {total}")
    print(f"Words (avg)     : {sum(word_counts) / total:.0f}")
    print(f"Words (min/max) : {min(word_counts)} / {max(word_counts)}")
    print(f"Metadata keys   : {', '.join(sorted(documents[0].metadata.keys()))}")
    print("Example         :")
    example = documents[min(5, total - 1)]
    print(f"  metadata: {example.metadata}")
    print(f"  content : {example.page_content[:160].strip()}...")


if __name__ == "__main__":
    print(f"Preprocessing {config.PDF_PATH}")
    print(f"Indexing pages {config.CONTENT_PAGE_RANGE[0]}-{config.CONTENT_PAGE_RANGE[1]}")

    parent_blocks, child_chunks = prepare_documents()

    describe_chunks(parent_blocks, "Parent blocks")
    describe_chunks(child_chunks, "Child chunks")

    pages_seen = [doc.metadata["page"] for doc in child_chunks]
    chapters_seen = {doc.metadata["chapter"] for doc in child_chunks}
    print(f"\nPage range covered : {min(pages_seen)} - {max(pages_seen)}")
    print(f"Chapters covered   : {len(chapters_seen)}")
