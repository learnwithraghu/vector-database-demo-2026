import os
import re
import lancedb
from pypdf import PdfReader
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

DATA_DIR = "share-holder-letters"
DB_URI = "./lancedb_data"
TABLE_NAME = "buffett_letters_multi"

def create_embeddings():
    print("Pre-loading VectorDB embeddings...")
    docs = []
    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}!")
        return

    # 1. Read PDFs
    for filename in pdf_files:
        print(f"Reading {filename}...")
        year = re.search(r"\d{4}", filename).group()
        reader = PdfReader(os.path.join(DATA_DIR, filename))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        docs.append({"year": int(year), "source": filename, "text": text})
            
    # 2. Chunk text
    print("Chunking documents...")
    chunk_size = 500
    overlap = 50
    chunks = []
    
    for doc in docs:
        text = doc['text']
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            chunks.append({
                "year": doc['year'],
                "source": doc['source'],
                "text": chunk_text
            })
            
    # 3. Setup LanceDB Schema
    print("Initializing Database and Model...")
    model = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")
    
    class Metadata(LanceModel):
        year: int
        source: str
        text: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()
        
    db = lancedb.connect(DB_URI)
    
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
        
    table = db.create_table(TABLE_NAME, schema=Metadata)
    
    # 4. Add data back to LanceDB
    print(f"Embedding and storing {len(chunks)} chunks...")
    table.add(chunks)
    print("✅ Pre-loading complete!")

if __name__ == "__main__":
    create_embeddings()
