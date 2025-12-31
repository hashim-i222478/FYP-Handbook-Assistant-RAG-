# FAST-NUCES FYP Handbook Assistant (RAG)

This project is a **Retrieval-Augmented Generation (RAG)** application designed to answer questions specifically related to the **FAST-NUCES BS Final Year Project (FYP) Handbook 2023**. 

It uses **Streamlit** for the user interface, **FAISS** for efficient similarity search, **Sentence Transformers** for generating embeddings, and **Google's Gemini** model for generating natural language answers based on retrieved context.

## 🚀 Features

- **Document Ingestion**: Extracts text from the handbook PDF, cleans it, and chunks it intelligently preserving page numbers and section headers.
- **Semantic Search**: Uses `all-MiniLM-L6-v2` embeddings and FAISS to find the most relevant sections of the handbook for a given query.
- **MMR Reranking**: Implements **Maximal Marginal Relevance (MMR)** to diversify the retrieved results and avoid redundancy.
- **GenAI Integration**: Connects to the **Google Gemini API** (specifically `gemini-2.5-flash`) to generate accurate, context-aware answers.
- **Source Citations**: Every answer provides specific page references from the handbook (`(p. X)`) to ensure traceability.

## 🛠️ Tech Stack

- **Python** 3.8+
- **Streamlit**: For the interactive web interface.
- **FAISS**: For high-performance similarity search of dense vectors.
- **Sentence Transformers**: For creating local embeddings of the text.
- **Google Generative AI (`google-generativeai`)**: For the LLM capabilities.
- **pdfplumber**: For robust PDF text extraction.
- **NLTK**: For sentence tokenization.
- **NumPy**: For vector operations.

## 📋 Prerequisites

1.  **Python**: Ensure you have Python installed.
2.  **Google Gemini API Key**: You will need an API key from Google AI Studio.
    - Set this in your environment variables or in the code (though environment variables are recommended for security).

## 📦 Installation

1.  **Clone the Repository** (if applicable):
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2.  **Install Dependencies**:
    You can install the required packages using pip.
    ```bash
    pip install streamlit faiss-cpu sentence-transformers google-generativeai pdfplumber nltk numpy tqdm python-dotenv
    ```

## ⚙️ Usage

### 1. Data Ingestion (Build the Index)

Before running the app, you need to process the PDF and build the vector index. This is done using `ingest.py`.

*   Place the handbook PDF (e.g., `3. FYP-Handbook-2023.pdf`) in the project root.
*   Run the ingestion script:

    ```bash
    python ingest.py --pdf "3. FYP-Handbook-2023.pdf"
    ```

    **Options:**
    - `--pdf`: Path to the input PDF file.
    - `--out_dir`: Directory to save the index (default: `./handbook_index`).
    - `--chunk_size`: Control chunk text size (defaults in code).

    *This process creates a `handbook_index` folder containing the FAISS index and chunk metadata.*

### 2. Run the Application

Once the index is built, you can launch the Streamlit app.

```bash
streamlit run app.py
```

*   The application will open in your default web browser (usually at `http://localhost:8501`).
*   Enter your question in the text box and click "Ask".

## 🔧 Configuration

You can adjust key parameters directly in `app.py`:

*   **`GEMINI_API_KEY`**: **CRITICAL**: You must set this environment variable. 
    *   Create a `.env` file in the project root (do not commit this file).
    *   Add `GEMINI_API_KEY=your_actual_api_key_here`.
    *   The app will read this automatically if you use `python-dotenv` or set it in your system environment.
*   **`TOP_K`** (Default: 7): Number of chunks to initially retrieve from FAISS.
*   **`FINAL_K`** (Default: 5): Number of chunks to keep after MMR reranking.
*   **`SIM_THRESHOLD`** (Default: 0.25): Minimum similarity score required to consider a chunk relevant.

## 📂 Project Structure

```
├── app.py                  # Main Streamlit application
├── ingest.py               # Script to process PDF and create embeddings
├── handbook_index/         # Generated directory containing index files
│   ├── faiss.index         # Vector store
│   └── chunks.jsonl        # Text chunks and metadata
├── 3. FYP-Handbook-2023.pdf # Source document
└── prompt_log.txt          # (Optional) Log of prompts used
```

## 🤝 Contributing

Contributions are welcome! If you find issues or want to improve the chunking logic or UI:

1.  Fork the repository.
2.  Create a feature branch.
3.  Commit your changes.
4.  Open a Pull Request.

## 📄 License

[Specify License Here, e.g., MIT, Proprietary, etc.]
