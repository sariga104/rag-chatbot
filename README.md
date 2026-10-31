# 📄 Chat with Your Documents

A RAG (retrieval-augmented generation) chatbot. Upload a PDF, ask questions in plain English, and get answers grounded in the document, with the source passages shown.

<!-- Add a screenshot or GIF here: ![Demo](demo.gif) -->

## How it works

1. **Read**: the PDF is split into small overlapping text chunks.
2. **Embed**: each chunk is converted into a vector with `sentence-transformers` and stored in ChromaDB.
3. **Retrieve**: your question is embedded and the 4 most similar chunks are found.
4. **Answer**: those chunks plus your question are sent to Claude, which answers using only that context.

## Tech stack

Python · Streamlit · Anthropic API · ChromaDB · sentence-transformers · pypdf

## Getting started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env            # Windows: copy .env.example .env
# then open .env and paste your key

# 5. Run the app
streamlit run app.py
```

## Project structure

```
rag-chatbot/
├── app.py            # Streamlit user interface
├── src/
│   └── rag.py        # chunking, embeddings, search, LLM call
├── requirements.txt
├── .env.example      # template for your API key
└── .gitignore
```

## Limitations and future ideas

- Scanned PDFs (images of text) aren't supported yet; adding OCR would fix this.
- Only one document at a time; multi-file support is a natural next step.
- Add an evaluation set to measure answer quality.
