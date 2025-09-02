import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
for k in ("GOOGLE_API_KEY", "DATABASE_URL", "EMBEDDING_MODEL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

def ingest_pdf():
    current_dir = Path(__file__).parent.parent  # Go up one level to the root directory
    pdf_path = current_dir / "document.pdf"

    print(f"Loading PDF from: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path)).load()
    print(f"Loaded {len(docs)} pages from PDF")

    print("Splitting documents into chunks...")
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, add_start_index=False).split_documents(docs)
    if not splits:
        print("There is no content to split")
        raise SystemExit(0)

    print("Enriching document metadata...")
    enrichedDocs = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]


    print("Generating embeddings with its document IDs...")
    ids = [f"doc-{i}" for i in range(len(enrichedDocs))]
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL","models/embedding-001"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("Storing embeddings in PostgreSQL...")
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    print(f"Adding {len(enrichedDocs)} documents to the vector store...")
    store.add_documents(documents=enrichedDocs, ids=ids)
    print("Documents successfully added to the vector store!")


if __name__ == "__main__":
    ingest_pdf()