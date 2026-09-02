"""Build the vector databases used by the retriever comparison.

Run this ONCE before starting the app:

    python Project1/build_database.py

It wipes and rebuilds Project1/chroma_db/, which holds two collections:

  standard   the ~500 word child chunks. Used by Simple Similarity, MMR and
             Multi-Query.
  parent     the SAME child chunks, but each one points back to the ~2000 word
             block it came from. Used by the Parent Document Retriever, which
             searches the chunks and returns the blocks.

Both collections search identical text, so the only thing that varies between
the four retrievers is what they hand back to the LLM.
"""

import os
import pickle
import shutil
import time

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore

import config
import preprocess
from models import get_embeddings


def save_parent_store(store, path):
    """Pickle the {doc_id: Document} mapping held by an InMemoryStore.

    :param store: the InMemoryStore filled in by ParentDocumentRetriever
    :param path: where to write the pickle
    :return: number of parent documents saved
    """
    keys = list(store.yield_keys())
    documents = dict(zip(keys, store.mget(keys)))

    with open(path, "wb") as file:
        pickle.dump(documents, file)

    return len(documents)


def load_parent_store(path=config.PARENT_STORE_PATH):
    """Rebuild an InMemoryStore from the pickle written by save_parent_store().

    :param path: path to the pickle
    :return: a populated InMemoryStore
    """
    with open(path, "rb") as file:
        documents = pickle.load(file)

    store = InMemoryStore()
    store.mset(list(documents.items()))
    return store


def build_standard_collection(children, embeddings):
    """Index the child chunks on their own.

    :param children: child Documents from preprocess.split_children()
    :param embeddings: the embedding model
    :return: the Chroma vector store
    """
    return Chroma.from_documents(
        documents=children,
        embedding=embeddings,
        collection_name=config.STANDARD_COLLECTION,
        persist_directory=config.STANDARD_DB_DIR,
    )


def build_parent_collection(blocks, embeddings):
    """Index the child chunks keyed back to their parent blocks.

    parent_splitter=None tells ParentDocumentRetriever that the documents it is
    given ARE the parents, so it only has to split them into children. Building
    the blocks in preprocess.py rather than letting the retriever do it is what
    makes the parents genuinely bigger than a single page.

    :param blocks: parent Documents from preprocess.build_blocks()
    :param embeddings: the embedding model
    :return: the InMemoryStore holding the parent documents
    """
    vectorstore = Chroma(
        collection_name=config.PARENT_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.PARENT_DB_DIR,
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=preprocess.get_child_splitter(),
        parent_splitter=None,
    )
    retriever.add_documents(blocks)

    return store


def main():
    start_time = time.time()

    if os.path.exists(config.CHROMA_DIR):
        print(f"Removing the existing database at {config.CHROMA_DIR}")
        shutil.rmtree(config.CHROMA_DIR)
    os.makedirs(config.CHROMA_DIR, exist_ok=True)

    print("Preprocessing the book...")
    blocks, children = preprocess.prepare_documents()
    print(f"  {len(blocks)} parent blocks, {len(children)} child chunks")

    embeddings = get_embeddings()

    print("Building the standard collection...")
    build_standard_collection(children, embeddings)

    print("Building the parent/child collection...")
    store = build_parent_collection(blocks, embeddings)
    saved = save_parent_store(store, config.PARENT_STORE_PATH)
    print(f"  stored {saved} parent documents")

    print(f"\nDone in {time.time() - start_time:.1f}s.")
    print(f"Databases written to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
