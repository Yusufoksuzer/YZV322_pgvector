import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

def main():
    print("1. Loading the AI model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("2. Connecting to the database...")
    conn = psycopg2.connect(
        dbname="vectordb", 
        user="admin", 
        password="password", 
        host="localhost", 
        port="5432"
    )
    cur = conn.cursor()

    print("3. Activating pgvector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)

    print("4. Creating vector table (384 dimensions)...")
    cur.execute("DROP TABLE IF EXISTS ai_documents;")
    cur.execute("""
        CREATE TABLE ai_documents (
            id serial PRIMARY KEY,
            content text,
            embedding vector(384)
        );
    """)
    conn.commit()

    print("5. Generating real embeddings and inserting data...")
    sentences = [
        "Deep learning and artificial intelligence are transforming the tech industry.",
        "A robust data engineering pipeline ensures smooth ETL operations.",
        "Computer vision algorithms can detect objects in high-resolution images.",
        "There is a huge discount on premium cat food today!"
    ]
    
    embeddings = model.encode(sentences)
    
    for sentence, embedding in zip(sentences, embeddings):
        cur.execute("INSERT INTO ai_documents (content, embedding) VALUES (%s, %s)", 
                    (sentence, np.array(embedding)))
    conn.commit()

    print("\n--- SEMANTIC SEARCH OPERATION ---")
    search_query = "How do neural networks process visual data?"
    print(f"Query: '{search_query}'\n")

    query_vector = model.encode(search_query)

    cur.execute("""
        SELECT content, embedding <-> %s AS distance
        FROM ai_documents
        ORDER BY distance ASC
        LIMIT 2;
    """, (np.array(query_vector),))

    rows = cur.fetchall()
    print("Top 2 most contextually similar results:")
    for row in rows:
        print(f"- Content: {row[0]} | Distance Score: {row[1]:.4f}")

    cur.close()
    conn.close()
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()