import streamlit as st
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="pgvector AI Search", page_icon="🔍", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def get_db_connection():
    conn = psycopg2.connect(
        dbname="vectordb", user="admin", password="password", host="localhost", port="5432"
    )
    register_vector(conn)
    return conn

conn = get_db_connection()
cur = conn.cursor()
cur.execute("SELECT content FROM ai_documents ORDER BY id;")
all_docs = cur.fetchall()
cur.close()
conn.close()

with st.sidebar:
    st.header("📚 Database Contents")
    st.markdown("Available documents stored in the pgvector database:")
    for doc in all_docs:
        st.info(doc[0])

st.title("🔍 Semantic Search with pgvector")
st.markdown("Performs **semantic matching** between texts in the database, rather than exact keyword matching.")

query = st.text_input("Enter a sentence or concept to search:", placeholder="e.g., How do neural networks process visual data?")

if st.button("Search", type="primary"):
    if query:
        with st.spinner('Comparing AI vectors...'):
            query_vector = model.encode(query)
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT content, embedding <-> %s AS distance
                FROM ai_documents
                ORDER BY distance ASC
                LIMIT 2;
            """, (np.array(query_vector),))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            st.subheader("Top Matches:")
            for content, distance in results:
                st.success(f"**Distance Score:** {distance:.4f} \n\n**Document:** {content}")
    else:
        st.warning("Please enter a text to search.")
