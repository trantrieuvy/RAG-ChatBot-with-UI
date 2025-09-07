# Lecture Slide RAG Assistant
![ChatBot UI](/UI.jpg "ChatBot Interactive UI")
This project is a RAG ChatBot built with:
- Ollama for running local LLMs and embeddings  
- LangChain for data handling and retrieval  
- ChromaDB for vector storage  
- Streamlit for a simple chat interface  
The goal: Ask natural-language questions about your lecture slides (PDFs) and get grounded answers with direct links back to the original slide pages.

# Project Structure
````
├── chat\_ui.py               # Streamlit app (main chat interface)
├── populate\_database.py     # Build vector DB from lecture PDFs
├── get\_embedding\_function.py # Defines embedding model (Ollama + nomic-embed-text)
├── llama3.2\_tet.modelfile   # Custom Ollama model definition (tet\_bot)
├── data/                    # Put your lecture PDFs here
├── chroma/                  # ChromaDB will be stored here (auto-created)
````

# Setup Instructions

## 1. Install dependencies
Create a virtual environment and install requirements:
```bash
pip install -r requirements.txt
````
## 2. Install and run Ollama
Download Ollama and pull the required models:
```bash
# Embedding model
ollama pull nomic-embed-text
# Base LLM 
ollama pull llama3.2
```

## 3. Create custom LLM (`tet_bot`)
Use the provided `llama3.2_tet.modelfile`:
```bash
ollama create tet_bot -f llama3.2_tet.modelfile
```

## 4. Add your PDFs
Put all lecture slide PDFs in the `data/` folder.

## 5. Populate the database
Run this to build (or rebuild) the vector DB:
```bash
python populate_database.py --data-path data/ --chroma-path chroma/ --reset
```

## 6. Start the chat app
```bash
streamlit run chat_ui.py
```

# Usage
1. Select a topic in the sidebar:
   * **TET** → uses RAG on your lecture slides
   * **General questions** → plain LLM chat
2. Type a question (e.g., *"What is the Maxwell stress tensor?"*)
3. The assistant:
   * Retrieves relevant chunks from your slides
   * Builds a prompt with **context + history + your question**
   * Calls your local LLM (`tet_bot`)
   * Shows the answer and expandable **sources**
4. Expand a source to open the actual PDF page in-browser.
