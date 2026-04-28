import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
conn = psycopg2.connect(dbname="vectordb", user="admin", password="password", host="localhost", port="5432")
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()
register_vector(conn)

cur.execute("DROP TABLE IF EXISTS ai_documents;")
cur.execute("""
    CREATE TABLE ai_documents (
        id serial PRIMARY KEY,
        content text,
        embedding vector(384)
    );
""")

sentences = [
    "Deep learning and artificial intelligence are transforming the tech industry.",
    "A robust data engineering pipeline ensures smooth ETL operations.",
    "Computer vision algorithms can detect objects in high-resolution images.",
    "There is a huge discount on premium cat food today!"
]
embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    cur.execute("INSERT INTO ai_documents (content, embedding) VALUES (%s, %s)", (sentence, np.array(embedding)))

conn.commit()
cur.close()
conn.close()
print("Database successfully populated with 384-dimensional real vectors!")
