import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.parsers import OpenParse

# 1. Configuration
# This points to where you put your books
DATA_DIR = "./data/novels/"
HOST = "0.0.0.0"
PORT = 8000

# 2. Ingestion: Watch the folder for .txt files
# 'mode="streaming"' means it keeps watching for new files if you add them later
documents = pw.io.fs.read(
    DATA_DIR,
    format="binary",
    mode="streaming",
    with_metadata=True
)

# 3. Text Decoding
# Converts the raw file bytes into readable text strings
table = documents.select(
    text=pw.this.data.decode("utf-8"),
    meta=pw.this.metadata
)

# 4. The Vector Store Server
# - OpenParse: Smartly splits text into paragraphs/chunks
# - Embedder: Turns text into numbers (vectors) for searching
server = VectorStoreServer(
    table,
    parser=OpenParse(table_map={"text": "text"}), 
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    port=PORT,
    host=HOST
)

if __name__ == "__main__":
    print(f"🔥 Pathway Indexer running on {HOST}:{PORT}")
    print(f"   Watching {DATA_DIR} for novels...")
    server.run()