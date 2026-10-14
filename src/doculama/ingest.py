from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "doculama" / "data" / "pdfs"
DB_DIR = BASE_DIR / "doculama" / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"

def ingest_pdfs():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(pdf_file)
            doc.metadata["file_name"] = pdf_file.name
            doc.metadata["page"] = doc.metadata.get("page", 0)
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="pdf_documents",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )
    vector_store.reset_collection()
    vector_store.add_documents(documents=chunks)

    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Database: {DB_DIR}")

if __name__ == "__main__":
    ingest_pdfs()