import argparse, os, shutil, hashlib
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

DEFAULT_CHROMA = "chroma/"
DEFAULT_DATA = "data/"

def load_pdfs(data_path: str) -> list[Document]:
    return PyPDFDirectoryLoader(data_path).load()

def split_documents(docs: list[Document], chunk_size=800, overlap=100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len
    )
    return splitter.split_documents(docs)

def stable_id(doc: Document) -> str:
    # for duplicates to not reinsert
    h = hashlib.sha1()
    h.update(doc.page_content.encode("utf-8"))
    src = str(doc.metadata.get("source", ""))
    pg = str(doc.metadata.get("page", ""))
    h.update(f"|{src}|{pg}".encode("utf-8"))
    return h.hexdigest()

def add_to_chroma(chunks: list[Document], chroma_path: str):
    db = Chroma(persist_directory=chroma_path, embedding_function=get_embedding_function())
    existing = set(db.get(include=[]).get("ids", []))  
    to_add, to_ids = [], []

    # Track per-(source,page) chunk indices to build display_source consistently
    counters = {}

    for c in chunks:
        # Build display_source like "data/file.pdf:12:2"
        src = str(c.metadata.get("source", ""))
        pg = str(c.metadata.get("page", "0"))
        key = (src, pg)
        counters[key] = counters.get(key, -1) + 1
        c.metadata["display_source"] = f"{src}:{pg}:{counters[key]}"
        hash_id = stable_id(c)
        c.metadata["hash_id"] = hash_id
        cid = hash_id
        if cid not in existing:
            to_add.append(c); to_ids.append(cid)

    if to_add:
        print(f"Adding {len(to_add)} new chunks to {chroma_path}")
        db.add_documents(to_add, ids=to_ids)
        # db.persist()
    else:
        print("No new documents to add.")

def clear_database(chroma_path: str):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"Cleared {chroma_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=DEFAULT_DATA)
    ap.add_argument("--chroma-path", default=DEFAULT_CHROMA)
    ap.add_argument("--reset", action="store_true", help="Reset the vector DB before ingest.")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=100)
    args = ap.parse_args()

    if args.reset:
        clear_database(args.chroma_path)

    print(f"Loading PDFs from {args.data_path} ...")
    docs = load_pdfs(args.data_path)
    print(f"Loaded {len(docs)} documents")

    print(f"Splitting into chunks (size={args.chunk_size}, overlap={args.chunk_overlap}) ...")
    chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Produced {len(chunks)} chunks")

    add_to_chroma(chunks, args.chroma_path)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
