# YZV322E pgvector Demo: Semantic Search Engine

This repository contains a technical demonstration of vector similarity search using PostgreSQL and the `pgvector` extension. The project illustrates how high-dimensional vector embeddings can be used to perform semantic search, allowing for context-aware data retrieval that goes beyond traditional keyword matching.

## Tech Stack
* **Database:** PostgreSQL with `pgvector` extension
* **Infrastructure:** Docker & Docker Compose
* **ML Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
* **Application Framework:** Python & Streamlit

## System Requirements
* Docker Desktop (Ensure WSL 2 integration is enabled)
* Python 3.10 or higher
* Recommended environment: WSL 2 (Ubuntu) for optimal performance

## Installation and Execution

### 1. Database Setup
Start the PostgreSQL container in detached mode:
```bash
docker compose up -d
```

### 2. Environment Configuration
Create a virtual environment and install the necessary Python packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Data Ingestion
Run the setup script to initialize the database schema, enable the vector extension, and populate the table with real AI-generated embeddings:
```bash
python setup_db.py
```

### 4. Running the Web UI
Launch the Streamlit dashboard to interact with the semantic search engine:
```bash
streamlit run app.py
```

## Core Methodology
The application follows a standard RAG-like (Retrieval-Augmented Generation) workflow:
1. **Vectorization:** Input text is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` model.
2. **Storage:** These embeddings are stored in a PostgreSQL `vector(384)` column.

## Expected Output

When the Streamlit web application is launched and a query like "feline nutrition" is entered, the semantic search engine will bypass keyword limitations and return semantically similar results along with their calculated L2 distance scores. 

<img width="1031" height="556" alt="image" src="https://github.com/user-attachments/assets/99340258-aa85-4ead-8838-d6cb08b2796e" />


## AI Usage Disclosure

During the development of this project, artificial intelligence tools (specifically Large Language Models) were utilized as an assistant. The AI was used to brainstorm the architectural pipeline, debug Docker configurations, and refine the English text in the documentation and presentation scripts. However, the core conceptual integration of PostgreSQL, `pgvector`, and the vectorization logic was entirely reviewed, orchestrated, and tested by the student to ensure technical accuracy and course alignment.
4. **Similarity Search:** When a user enters a query, it is vectorized on-the-fly and compared against the database using the `<->` (L2 Distance) operator.
5. **Ranking:** The results are ordered by spatial proximity, where lower distance scores indicate higher semantic similarity.
