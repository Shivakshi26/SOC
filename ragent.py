import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import sys
import os

# Configuration
openai.api_key = "API_KEY"
client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=openai.api_key)

# PDF Parsing
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Text Chunking
def chunk_text(text, max_chunk_size=500):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    chunks.append(current_chunk.strip())
    return chunks

# Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Summarizer
def generate_summary(text):
    prompt = f"Summarize the following research paper:\n\n{text[:4000]}\n\nSummary:"
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# FAISS Store and RAG
def build_faiss_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings))
    return index, chunks, embeddings

def get_top_k_contexts(query, chunks, index, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

def answer_query(query, chunks, index):
    contexts = get_top_k_contexts(query, chunks, index, k=3)
    combined_context = "".join(contexts)
    combined_context = combined_context[:8000]
    
    prompt = f"Based on the following research paper excerpts:\n\n{combined_context}\n\nAnswer the question:\n{query}"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Terminal Pipeline
def main():
    pdf_path = input("Enter the full path to the research paper PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("File not found. Please check the path and try again.")
        return

    print("\nExtracting and processing the PDF.\n")
    text = extract_text_from_pdf(pdf_path)
    summary = generate_summary(text)
    print("\n Summary of the paper:\n")
    print(summary)

    chunks = chunk_text(text)
    index, chunks, _ = build_faiss_index(chunks)

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting. Do Visit Again!")
            break
        answer = answer_query(query, chunks, index)
        print("\n Answer:\n")
        print(answer)

if __name__ == "__main__":
    main()
