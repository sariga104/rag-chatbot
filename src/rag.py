"""Core RAG logic: read a PDF, store it as searchable chunks, answer questions."""

import uuid

import chromadb
from anthropic import Anthropic
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = "claude-sonnet-5-5"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You answer questions about a document using only the excerpts provided. "
    "If the answer is not in the excerpts, say you couldn't find it in the document. "
    "Mention page numbers when they help."
)

_embedder = None


def get_embedder():
    """Load the embedding model once and reuse it (it's slow to load)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping pieces so answers aren't cut off mid-idea."""
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


class DocumentStore:
    """Holds one PDF as searchable chunks in an in-memory vector database."""

    def __init__(self):
        client = chromadb.EphemeralClient()
        self.collection = client.create_collection(f"doc_{uuid.uuid4().hex}")

    def add_pdf(self, file):
        """Read a PDF, chunk it, embed the chunks, and store them. Returns chunk count."""
        reader = PdfReader(file)
        texts, metadatas = [], []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk in chunk_text(page_text):
                texts.append(chunk)
                metadatas.append({"page": page_number})

        if not texts:
            raise ValueError("No readable text found. Is this a scanned PDF?")

        embeddings = get_embedder().encode(texts).tolist()
        ids = [f"chunk_{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )
        return len(texts)

    def search(self, question, k=4):
        """Return the k chunks most similar to the question."""
        query_embedding = get_embedder().encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
        )
        return [
            {"text": doc, "page": meta["page"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]


def answer_question(question, sources, history):
    """Ask the LLM to answer using the retrieved chunks.

    history: earlier turns as [{"role": "user"/"assistant", "content": "..."}]
    """
    context = "\n\n".join(f"[Page {s['page']}]\n{s['text']}" for s in sources)
    messages = history[-6:] + [
        {
            "role": "user",
            "content": f"Document excerpts:\n{context}\n\nQuestion: {question}",
        }
    ]

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text
